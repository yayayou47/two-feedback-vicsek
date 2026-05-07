"""Shared matplotlib style for journal figures.

PRE/PRL-style sizing: single-column 3.4 in, two-column 7.0 in. Cream
background applied to figure and axes. Tick/label sizes kept at
8-9 pt so text remains legible at print size.
"""
from __future__ import annotations

import matplotlib as mpl

CREAM = "#FFF8E1"
PARTICLE_BLUE = "#1f4ea1"

SINGLE_COL = (3.4, 2.6)
DOUBLE_COL = (7.0, 3.0)
SQUARE = (3.4, 3.4)


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": CREAM,
        "axes.facecolor": CREAM,
        "savefig.facecolor": CREAM,
        "savefig.edgecolor": CREAM,
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
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    })
