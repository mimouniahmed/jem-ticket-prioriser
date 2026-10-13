"""2D toy example: the energy landscape can be PLOTTED here, which is impossible in 417 dimensions.

Three point clouds (3 classes); compares the E(x) landscape of a standard classifier with that
of a JEM, both trained with exactly the same code as on real bugs.
"""

from __future__ import annotations

import numpy as np
import torch

from .model import MLP, ReplayBuffer, energy_from_logits, jem_loss


def make_toy(n: int = 900, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    angles = np.array([90, 210, 330]) * np.pi / 180
    centers = 1.5 * np.stack([np.cos(angles), np.sin(angles)], 1)
    x = centers[y] + rng.normal(0, 0.35, (n, 2))
    return x.astype(np.float32), y.astype(np.int64)


def train_toy(
    x: np.ndarray, y: np.ndarray, alpha: float, epochs: int = 150, seed: int = 0,
    sgld_steps: int = 20, sgld_step_size: float = 0.05, sgld_noise: float = 0.05, energy_reg: float = 0.1,
) -> tuple[MLP, ReplayBuffer | None]:
    torch.manual_seed(seed)
    model = MLP(2, 3, hidden=64, depth=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    buffer = ReplayBuffer(2000, 2, reinit_prob=0.05) if alpha > 0 else None
    if buffer is not None:
        buffer.buffer = (torch.rand(2000, 2) * 8 - 4)  # uniform start over the whole [-4, 4]² square
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 100):
            b = perm[i : i + 100]
            loss, _ = jem_loss(model, xt[b], yt[b], buffer, alpha, energy_reg, sgld_steps, sgld_step_size, sgld_noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval(), buffer


@torch.no_grad()
def energy_grid(model: MLP, lim: float = 4.0, n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.linspace(-lim, lim, n, dtype=np.float32)
    xx, yy = np.meshgrid(g, g)
    pts = torch.from_numpy(np.stack([xx.ravel(), yy.ravel()], 1))
    return xx, yy, energy_from_logits(model(pts)).numpy().reshape(n, n)
