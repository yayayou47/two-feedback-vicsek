"""Shared matplotlib style for journal figures.

White-background figures sized for PRE/PRL/Soft Matter
(single-column 3.4 in, two-column 7.0 in). Colorblind-safe
Wong palette as default. Cream is kept as an alternative for
slides via ``apply(bg="cream")``.
"""
from __future__ import annotations

import matplotlib as mpl

WHITE = "#ffffff"
CREAM = "#FFF8E1"

# Wong colorblind-safe palette
# https://www.nature.com/articles/nmeth.1618
WONG = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "sky":     "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "rpurple": "#CC79A7",
}

PARTICLE_BLUE = WONG["blue"]

SINGLE_COL = (3.4, 2.6)
DOUBLE_COL = (7.0, 3.0)
SQUARE = (3.4, 3.4)


def apply(bg: str = "white") -> None:
    """Activate the journal style. ``bg`` is ``"white"`` (default,
    for journal submission) or ``"cream"`` (for slides/blog)."""
    color = WHITE if bg == "white" else CREAM
    mpl.rcParams.update({
        "figure.facecolor": color,
        "axes.facecolor": color,
        "savefig.facecolor": color,
        "savefig.edgecolor": color,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    })
