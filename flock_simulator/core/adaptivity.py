"""
Shared-sigmoid coupling between the local alignment-zone neighbour
count ``n_i`` and the per-particle motility / noise-shape fields.

A single sigmoid in ``n_i`` gates both adaptations:

  sig_i  = 1 / (1 + exp(-(n_i - n_star) * slope)),
  v_i    = v_max     - (v_max - v_min)         * sig_i,
  alpha_i = alpha_min + (alpha_max - alpha_min) * sig_i.

Setting ``v_min = v_max`` freezes the motility channel; setting
``alpha_min = alpha_max`` freezes the noise-shape channel; the four
study modes (baseline, v2-limit, v3-limit, full) are recovered by
toggling these two pairs.

Boundary behaviour:
  - n_i = 0 -> sig_i = sigmoid(-n_star * slope) ~ 0 for typical
    n_star >= 1 and slope >= 1; v_i -> v_max, alpha_i -> alpha_min.
  - n_i large -> sig_i -> 1; v_i -> v_min, alpha_i -> alpha_max.
"""
from __future__ import annotations

import numpy as np


def shared_sigmoid_fields(
    n_ali: np.ndarray,
    n_star: float,
    slope: float,
    v_min: float,
    v_max: float,
    alpha_min: float,
    alpha_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-particle ``(v_i, alpha_i)`` from neighbour counts.

    Args:
      n_ali: (N,) int or float array, alignment-zone neighbour count.
      n_star: sigmoid threshold (typical occupancy at which the
        feedback engages).
      slope: sigmoid sharpness; larger means a more abrupt switch.
      v_min, v_max: speed bounds (must satisfy ``v_min <= v_max``).
        ``v_min == v_max`` freezes the motility channel.
      alpha_min, alpha_max: stability-index bounds in ``(0, 2]``
        (must satisfy ``alpha_min <= alpha_max``). ``alpha_min ==
        alpha_max`` freezes the noise-shape channel.

    Returns:
      v: (N,) float array of per-particle speeds.
      alpha: (N,) float array of per-particle stability indices.

    Complexity: O(N), no allocations beyond the two return arrays
    (intermediate ``sig`` is computed in-place via ``expit``-style
    NumPy expressions).
    """
    z = (n_ali.astype(np.float64) - n_star) * slope
    sig = 1.0 / (1.0 + np.exp(-z))
    v = v_max - (v_max - v_min) * sig
    alpha = alpha_min + (alpha_max - alpha_min) * sig
    return v, alpha
