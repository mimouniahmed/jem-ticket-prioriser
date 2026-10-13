"""Style commun des graphiques (palette catégorielle accessible, axes discrets)."""

from __future__ import annotations

import matplotlib.pyplot as plt
from cycler import cycler

from .cleaning import CLASSES

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # bleu, orange, turquoise, jaune (ordre fixe)
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

CLASS_COLORS = dict(zip(CLASSES, SERIES))
MODEL_COLORS = {"MLP (baseline)": SERIES[0], "JEM": SERIES[1], "Régression logistique": SERIES[2]}
HUMAN_BOT_COLORS = {"humains": SERIES[0], "bots": SERIES[1]}


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.figsize": (8, 4.2), "figure.dpi": 110,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
        "axes.prop_cycle": cycler(color=SERIES),
        "lines.linewidth": 2, "lines.markersize": 8,
        "legend.frameon": False, "font.size": 10,
    })
