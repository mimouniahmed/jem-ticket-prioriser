"""Métriques : classification, calibration, qualité du classement et détection d'anomalies."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Score d'urgence = espérance sous p(y|x) de ces poids (P1 -> 1, P2 -> 2/3, P3 -> 1/3, Basse -> 0)
URGENCY_WEIGHTS = np.array([1.0, 2 / 3, 1 / 3, 0.0])


def urgency(probs: np.ndarray) -> np.ndarray:
    return probs @ URGENCY_WEIGHTS


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Écart moyen entre la confiance annoncée et la précision réelle, par tranche de confiance."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def top_k_recall(score: np.ndarray, is_target: np.ndarray, frac: float = 0.1) -> float:
    """Part des bugs cibles (ex. P1) retrouvés dans les `frac` % les mieux classés."""
    k = max(1, int(len(score) * frac))
    top = np.argsort(-score)[:k]
    return float(is_target[top].sum() / max(1, is_target.sum()))


def classification_report(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(1)
    u = urgency(probs)
    nll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()
    brier = ((probs - np.eye(probs.shape[1])[y]) ** 2).sum(1).mean()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "nll": float(nll),
        "brier": float(brier),
        "ece": expected_calibration_error(probs, y),
        "spearman_urgency": float(spearmanr(u, -y).statistic),  # y=0 est le plus urgent
        "p1_recall_top10": top_k_recall(u, y == 0, 0.10),
    }


def ood_auroc(score_in: np.ndarray, score_out: np.ndarray) -> float:
    """AUROC pour séparer « normal » (in) et « atypique » (out) ; score plus haut = plus atypique."""
    labels = np.r_[np.zeros(len(score_in)), np.ones(len(score_out))]
    return float(roc_auc_score(labels, np.r_[score_in, score_out]))
