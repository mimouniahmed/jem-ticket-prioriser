"""Fast tests, with no database or network access."""

import numpy as np
import pandas as pd
import torch

from jem_prioriser.cleaning import LABEL_OF_PRIORITY, human_sql, is_bot
from jem_prioriser.features import BugFeaturizer
from jem_prioriser.metrics import classification_report, expected_calibration_error, top_k_recall, urgency
from jem_prioriser.model import MLP, ReplayBuffer, energy_from_logits, jem_loss, sgld
from jem_prioriser.prioriser import Prioriser


def test_bot_rules():
    assert is_bot("intermittent-bug-filer@mozilla.bugs") and is_bot("update-bot@bmo.tld")
    assert not is_bot("u123@disabled.tld") and not is_bot("botond@mozilla.com")
    assert "NOT LIKE '%@mozilla.bugs'" in human_sql()
    assert LABEL_OF_PRIORITY["P4"] == LABEL_OF_PRIORITY["P5"] == "Low"


def test_energy_is_minus_logsumexp():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    assert torch.allclose(energy_from_logits(logits), -torch.logsumexp(logits, 1))
    # shifting all logits leaves p(y|x) unchanged but changes the energy
    assert torch.allclose(torch.softmax(logits + 10, 1), torch.softmax(logits, 1))
    assert energy_from_logits(logits + 10) < energy_from_logits(logits)


def test_sgld_lowers_energy():
    torch.manual_seed(0)
    model = MLP(8, 4, hidden=32)
    x0 = torch.randn(64, 8)
    x1 = sgld(model, x0, n_steps=30, step_size=0.5, noise=0.0)
    assert model.energy(x1).mean() < model.energy(x0).mean()


def test_jem_loss_terms():
    model = MLP(8, 4, hidden=32)
    x, y = torch.randn(16, 8), torch.randint(0, 4, (16,))
    ce, stats = jem_loss(model, x, y)  # no buffer = standard classifier
    assert set(stats) == {"ce"}
    loss, stats = jem_loss(model, x, y, ReplayBuffer(100, 8), sgld_steps=3)
    assert torch.isfinite(loss) and {"ce", "gen", "e_pos", "e_neg"} <= set(stats)


def test_metrics():
    y = np.array([0, 1, 2, 3])
    perfect = np.eye(4)
    assert classification_report(perfect, y)["accuracy"] == 1.0
    assert expected_calibration_error(perfect, y) == 0.0
    assert np.allclose(urgency(perfect), [1, 2 / 3, 1 / 3, 0])
    assert top_k_recall(np.array([0.9, 0.1, 0.2, 0.3]), np.array([True, False, False, False]), 0.25) == 1.0


def _fake_bugs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    words = ["crash", "freeze", "typo", "layout", "slow", "leak", "button", "tab"]
    return pd.DataFrame({
        "id": np.arange(n),
        "summary": [" ".join(rng.choice(words, 3)) for _ in range(n)],
        "description": [" ".join(rng.choice(words, 12)) for _ in range(n)],
        "product": rng.choice(["Firefox", "Core"], n),
        "component": rng.choice(["DOM", "Layout", "General"], n),
        "type": rng.choice(["defect", "task"], n),
        "op_sys": rng.choice(["Windows", "Linux"], n),
    })


def test_featurizer_and_prioriser():
    df = _fake_bugs()
    feat = BugFeaturizer(svd_dim=4, min_category_count=1).fit(df)
    x = feat.transform(df)
    assert x.dtype == np.float32 and x.shape == (len(df), feat.dim)
    assert feat.transform_sparse(df).shape[0] == len(df)

    model = MLP(feat.dim, 4, hidden=16)
    with torch.no_grad():
        energies = energy_from_logits(model(torch.from_numpy(x))).numpy()
    ranked = Prioriser(feat, model, energies).score(df)
    assert list(ranked["rank"]) == list(range(1, len(df) + 1))
    assert (np.diff(ranked["urgency"]) <= 0).all()
    assert ranked["needs_review"].mean() < 0.1  # ~5% above the 95th percentile
