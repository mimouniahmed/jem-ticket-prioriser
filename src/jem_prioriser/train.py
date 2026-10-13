"""Training (baseline and JEM) with experiment tracking in MLflow.

    uv run jem-train --alpha 0 --run-name baseline-mlp       # classifier only
    uv run jem-train --alpha 1 --run-name jem                # JEM
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # compare runs in the browser
"""

from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from .cleaning import CLASSES
from .dataset import DEFAULT_DB, TRAIN_END, VAL_END, load_bot_bugs, load_labeled, temporal_split
from .features import BugFeaturizer
from .metrics import classification_report, ood_auroc
from .model import MLP, ReplayBuffer, energy_from_logits, jem_loss
from .prioriser import Prioriser, PrioriserPyfunc

TRACKING_URI = "sqlite:///mlflow.db"  # local store: runs + Model Registry, no server needed
EXPERIMENT = "jem-prioriser"
MODEL_NAME = "jem-prioriser"


@dataclass
class Prepared:
    """Ready-to-use data, so the TF-IDF/SVD is not recomputed for every training run."""
    featurizer: BugFeaturizer
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    bots: pd.DataFrame  # bot-created bugs: out-of-distribution set
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    x_bots: np.ndarray


def prepare_data(db_path: str = DEFAULT_DB, svd_dim: int = 256, cache: str | None = "data/prepared.joblib") -> Prepared:
    if cache and Path(cache).exists():
        data = joblib.load(cache)
        if data.featurizer.svd.n_components == svd_dim:
            return data
    train, val, test = temporal_split(load_labeled(db_path))
    bots = load_bot_bugs(db_path)
    featurizer = BugFeaturizer(svd_dim=svd_dim).fit(train)
    data = Prepared(
        featurizer, train, val, test, bots,
        featurizer.transform(train), featurizer.transform(val),
        featurizer.transform(test), featurizer.transform(bots),
    )
    if cache:
        joblib.dump(data, cache)
    return data


@dataclass
class TrainConfig:
    alpha: float = 1.0           # weight of the generative term; 0 = standard classifier
    epochs: int = 20
    warmup_epochs: int = 2       # cross-entropy-only epochs before SGLD is enabled
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 0.05
    hidden: int = 256
    depth: int = 2
    energy_reg: float = 0.1
    sgld_steps: int = 20
    sgld_step_size: float = 1.0
    sgld_noise: float = 0.01
    buffer_size: int = 10000
    reinit_prob: float = 0.05
    seed: int = 0


@dataclass
class TrainResult:
    prioriser: Prioriser
    history: list[dict]        # one row per epoch
    metrics: dict[str, float]  # test metrics
    run_id: str
    model_uri: str | None      # model URI in MLflow (None when log_model=False)


@torch.no_grad()
def predict(model: MLP, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = model(torch.from_numpy(x))
    model.train()
    return torch.softmax(logits, 1).numpy(), energy_from_logits(logits).numpy()


def bots_of_period(data: Prepared, period: str) -> np.ndarray:
    """Bot bugs from the validation (2025) or test (2026) period: hyperparameter selection
    must never see test-period data, bot bugs included."""
    t = data.bots["creation_time"]
    if period == "val":
        mask = (t >= pd.Timestamp(TRAIN_END, tz="UTC")) & (t < pd.Timestamp(VAL_END, tz="UTC"))
    else:
        mask = t >= pd.Timestamp(VAL_END, tz="UTC")
    return data.x_bots[mask.to_numpy()]


def evaluate(model: MLP, data: Prepared) -> dict[str, float]:
    """Test metrics + detection of 2026 bot bugs (never seen) by energy and by confidence."""
    p_test, e_test = predict(model, data.x_test)
    p_bots, e_bots = predict(model, bots_of_period(data, "test"))
    metrics = {f"test_{k}": v for k, v in classification_report(p_test, data.test["y"].to_numpy()).items()}
    metrics["ood_auroc_energy"] = ood_auroc(e_test, e_bots)
    metrics["ood_auroc_maxprob"] = ood_auroc(-p_test.max(1), -p_bots.max(1))
    return metrics


def train(
    cfg: TrainConfig, data: Prepared, run_name: str | None = None,
    log_model: bool = True, register: bool = False, verbose: bool = True,
):
    """Train an MLP (alpha=0) or a JEM (alpha>0) and log everything to MLflow.
    Returns a TrainResult (model, per-epoch history, test metrics, MLflow identifiers)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    x = torch.from_numpy(data.x_train)
    y = torch.from_numpy(data.train["y"].to_numpy(copy=True))
    y_val = data.val["y"].to_numpy()
    x_val_bots = bots_of_period(data, "val")
    model = MLP(x.shape[1], len(CLASSES), cfg.hidden, cfg.depth)
    # AdamW: decoupled weight decay (proper L2 regularisation with Adam)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.buffer_size, x.shape[1], cfg.reinit_prob) if cfg.alpha > 0 else None

    history, best_nll, best_state = [], float("inf"), None
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({**asdict(cfg), "model": "jem" if cfg.alpha > 0 else "mlp", "input_dim": x.shape[1]})
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            use_buffer = buffer if epoch > cfg.warmup_epochs else None
            perm = torch.randperm(len(x))
            sums: dict[str, float] = {}
            n_batches = 0
            for i in range(0, len(x), cfg.batch_size):
                b = perm[i : i + cfg.batch_size]
                loss, stats = jem_loss(
                    model, x[b], y[b], use_buffer, cfg.alpha, cfg.energy_reg,
                    cfg.sgld_steps, cfg.sgld_step_size, cfg.sgld_noise,
                )
                if not torch.isfinite(loss) or abs(stats.get("gen", 0.0)) > 1e4:
                    raise RuntimeError(f"Divergence at epoch {epoch}: {stats}")
                opt.zero_grad()
                loss.backward()
                opt.step()
                for k, v in stats.items():
                    sums[k] = sums.get(k, 0.0) + v
                n_batches += 1

            p_val, e_val = predict(model, data.x_val)
            val = classification_report(p_val, y_val)
            _, e_val_bots = predict(model, x_val_bots)
            val["ood_auroc_energy"] = ood_auroc(e_val, e_val_bots)  # bot detection on the 2025 period
            row = {"epoch": epoch, **{f"train_{k}": v / n_batches for k, v in sums.items()},
                   **{f"val_{k}": v for k, v in val.items()}, "seconds": time.time() - t0}
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            # Keep the epoch with the best validation NLL (which also rewards calibration).
            # For a JEM, warm-up epochs are not eligible: they would select a plain classifier
            # trained before the generative term is active.
            selectable = cfg.alpha == 0 or epoch > cfg.warmup_epochs
            if selectable and val["nll"] < best_nll:
                best_nll, best_state = val["nll"], {k: v.clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"[{epoch:2d}/{cfg.epochs}] ce={row['train_ce']:.3f}"
                      + (f" gen={row['train_gen']:+.3f}" if "train_gen" in row else "")
                      + f" | val acc={val['accuracy']:.3f} f1={val['macro_f1']:.3f}"
                      f" nll={val['nll']:.3f} ece={val['ece']:.3f} ood={val['ood_auroc_energy']:.2f}"
                      f" ({row['seconds']:.1f}s)")

        model.load_state_dict(best_state)
        metrics = evaluate(model, data)
        mlflow.log_metrics(metrics)

        _, e_train = predict(model, data.x_train)
        prioriser = Prioriser(data.featurizer, model, e_train)
        if not log_model:  # hyperparameter search: keep parameters and metrics only
            return TrainResult(prioriser, history, metrics, run.info.run_id, None)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "prioriser.joblib")
            prioriser.save(path)
            info = mlflow.pyfunc.log_model(
                name="prioriser", python_model=PrioriserPyfunc(), artifacts={"prioriser": path},
                registered_model_name=MODEL_NAME if register else None,
            )
    return TrainResult(prioriser, history, metrics, run.info.run_id, info.model_uri)


def train_logreg(data: Prepared, C: float = 1.0) -> dict[str, float]:
    """Standard baseline: logistic regression on the full sparse TF-IDF."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"logreg-C{C}"):
        mlflow.log_params({"model": "logreg", "C": C})
        clf = LogisticRegression(C=C, max_iter=3000).fit(data.featurizer.transform_sparse(data.train), data.train["y"])
        probs = clf.predict_proba(data.featurizer.transform_sparse(data.test))
        metrics = {f"test_{k}": v for k, v in classification_report(probs, data.test["y"].to_numpy()).items()}
        mlflow.log_metrics(metrics)
    return metrics


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field, default in asdict(TrainConfig()).items():
        p.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    p.add_argument("--run-name", default=None)
    p.add_argument("--register", action="store_true", help="register the model in the Model Registry")
    args = vars(p.parse_args(argv))
    run_name, register = args.pop("run_name"), args.pop("register")
    result = train(TrainConfig(**args), prepare_data(), run_name, register=register)
    print(f"MLflow run: {result.run_id}")
    for k, v in result.metrics.items():
        print(f"  {k:28s} {v:.4f}")


if __name__ == "__main__":
    main()
