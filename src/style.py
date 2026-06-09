"""Shared matplotlib style for journal figures.

Defines the project palette and rcParams for PRE/PRL/Soft Matter
figures: the colorblind-safe Wong colours, a fixed per-mode PALETTE
reused across analysis figures, PARTICLE_BLUE for snapshots, and
standard column sizes (single 3.4 in, double 7.0 in). apply() activates
serif 9 pt journal styling on a white (default) or cream background.
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

# Paper-wide mode palette. One saturated, distinctive hue per
# study mode, reused by every analysis figure (FSS pilot, g(r),
# autocorr, orderpdf, profile, ...). The same key set is used
# inside the snapshot npz files via the fixed_v / adaptive
# aliases. Importing PALETTE from this module is the canonical
# way to get a mode colour anywhere in the project.
PALETTE = {
    "vicsek_gauss":  "#000000",  # original Vicsek (Gaussian)   -- black
    "baseline":      "#1f4ea1",  # Cauchy fixed-parameter ref    -- blue
    "v2_limit":      "#2ca02c",  # noise-shape adaptation only  -- green
    "v3_limit":      "#d62728",  # motility adaptation only     -- red
    "full":          "#7e2aa0",  # double-adaptive (this work)  -- purple
    "fixed_v":       "#1f4ea1",  # alias used by snapshot npz
    "adaptive":      "#7e2aa0",  # alias used by snapshot npz
}

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
