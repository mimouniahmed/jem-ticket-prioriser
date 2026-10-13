"""Joint Energy-based Model (Grathwohl et al., 2019, « Your classifier is secretly an EBM »).

Un classifieur f(x) -> logits définit aussi un modèle à énergie :
    p(y | x) = softmax(f(x))_y                       priorité du bug
    E(x)     = -logsumexp_y f(x)_y ,  p(x) ∝ e^-E(x)   « ce bug ressemble-t-il aux autres ? »

Perte jointe :  L = CE(f(x), y) + alpha * [E(x_data) - E(x_sgld)] + alpha * lambda * (E² régularisation)
où x_sgld sont des échantillons négatifs tirés par dynamique de Langevin (SGLD) depuis un
replay buffer persistant (Persistent Contrastive Divergence).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Petit réseau dense. Pas de BatchNorm ni de Dropout : ils rendraient l'énergie
    dépendante du batch ou aléatoire, ce qui casse l'échantillonnage SGLD."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]  # activation lisse : gradients SGLD réguliers
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return energy_from_logits(self(x))


def energy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=1)


class ReplayBuffer:
    """Mémoire d'échantillons négatifs : chaque chaîne SGLD repart de là où elle s'était arrêtée,
    ce qui permet des chaînes longues pour un coût de 20 pas par itération."""

    def __init__(self, size: int, dim: int, reinit_prob: float = 0.05, device: str = "cpu"):
        self.reinit_prob = reinit_prob
        self.buffer = torch.randn(size, dim, device=device)  # features standardisées -> N(0, 1)

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self.buffer), (n,), device=self.buffer.device)
        x = self.buffer[idx].clone()
        reinit = torch.rand(n, device=x.device) < self.reinit_prob  # garde de la diversité
        x[reinit] = torch.randn_like(x[reinit])
        return x, idx

    def update(self, idx: torch.Tensor, x: torch.Tensor) -> None:
        self.buffer[idx] = x.detach()


def sgld(model: MLP, x: torch.Tensor, n_steps: int = 20, step_size: float = 1.0, noise: float = 0.01) -> torch.Tensor:
    """Dynamique de Langevin : on descend l'énergie (vers les zones « plausibles ») en ajoutant du bruit."""
    x = x.detach()
    for _ in range(n_steps):
        x.requires_grad_(True)
        grad = torch.autograd.grad(model.energy(x).sum(), x)[0]
        x = (x - step_size * grad + noise * torch.randn_like(x)).detach()
    return x


def jem_loss(
    model: MLP,
    x: torch.Tensor,
    y: torch.Tensor,
    buffer: ReplayBuffer | None = None,
    alpha: float = 1.0,
    energy_reg: float = 0.1,
    sgld_steps: int = 20,
    sgld_step_size: float = 1.0,
    sgld_noise: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sans buffer (ou alpha=0) : simple entropie croisée = classifieur classique (baseline)."""
    logits = model(x)
    ce = F.cross_entropy(logits, y)
    stats = {"ce": ce.item()}
    if buffer is None or alpha == 0:
        return ce, stats

    x_neg, idx = buffer.sample(x.shape[0])
    x_neg = sgld(model, x_neg, sgld_steps, sgld_step_size, sgld_noise)
    buffer.update(idx, x_neg)

    e_pos = energy_from_logits(logits)
    e_neg = model.energy(x_neg)
    # Maximum de vraisemblance de p(x) : baisser l'énergie des vrais bugs, remonter celle des
    # échantillons du modèle. Le terme L2 empêche les énergies de partir à l'infini.
    gen = e_pos.mean() - e_neg.mean()
    reg = e_pos.pow(2).mean() + e_neg.pow(2).mean()
    loss = ce + alpha * (gen + energy_reg * reg)
    stats.update(gen=gen.item(), e_pos=e_pos.mean().item(), e_neg=e_neg.mean().item())
    return loss, stats
