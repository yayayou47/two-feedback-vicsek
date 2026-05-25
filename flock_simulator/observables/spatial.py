"""
Spatial observables.

Two families of diagnostic live here:

  * the coarse grid-based density-separation index
    ``density_separation_index`` (the standard MIPS diagnostic
    used throughout the paper), and
  * grid-free alternatives requested by the methodological
    review --- ``pair_correlation_separation`` (a
    pair-correlation-based separation index) and
    ``dbscan_cluster_sizes`` (DBSCAN cluster identification
    in place of the grid-threshold connected-component
    method).

The grid-free routines are O(N^2) and meant for a
cross-check against the cheap grid diagnostics on a handful of
snapshots, not for inline use in the long FSS sweeps.
"""
from __future__ import annotations

import numpy as np


def density_separation_index(
    x: np.ndarray, y: np.ndarray, L: float, n_bins: int = 10,
) -> float:
    """Coarse MIPS diagnostic: max-over-median density on a grid.

    Args:
      x, y: (N,) particle coordinates in ``[0, L)``.
      L: periodic box side length.
      n_bins: number of bins per side; default ``10``.

    Returns:
      The ratio of the densest grid cell to the median of all
      non-empty cells. ``s_sep ~ 1`` for a homogeneous
      configuration; ``s_sep`` significantly above 1 indicates a
      dense / dilute coexistence on the chosen grid.
    """
    H, _, _ = np.histogram2d(
        x, y,
        bins=[n_bins, n_bins],
        range=[[0.0, L], [0.0, L]],
    )
    H = H.flatten()
    H = H[H > 0]
    if H.size == 0:
        return 1.0
    return float(H.max() / np.median(H))


def _periodic_local_density(
    x: np.ndarray, y: np.ndarray, L: float, R: float,
) -> np.ndarray:
    """Per-particle neighbour count within radius ``R`` (periodic).

    O(N^2); intended for snapshot-level cross-checks only.
    """
    halfL = 0.5 * L
    R2 = R * R
    N = x.shape[0]
    rho = np.empty(N, dtype=np.float64)
    for i in range(N):
        dx = x - x[i]
        dy = y - y[i]
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        rho[i] = float(np.count_nonzero(dx * dx + dy * dy < R2) - 1)
    return rho


def pair_correlation_separation(
    x: np.ndarray, y: np.ndarray, L: float, R: float = 1.0,
) -> float:
    """Grid-free separation index from the local-density field.

    Instead of binning particles onto a fixed ``n_bins x n_bins``
    grid, this estimator measures the per-particle local density
    ``rho_i`` (neighbour count inside radius ``R``, periodic) and
    returns the ratio of its 95th percentile to its median:

      ``s_pc = percentile(rho, 95) / median(rho)``.

    For a homogeneous fluid the local-density field is narrow and
    ``s_pc`` sits near 1; a dense/dilute coexistence broadens the
    distribution and pushes ``s_pc`` well above 1. Unlike the
    grid index, ``s_pc`` has no binning artefact and no preferred
    length scale beyond ``R``.

    Args:
      x, y: (N,) particle coordinates in ``[0, L)``.
      L: periodic box side length.
      R: local-density radius (default ``1.0``).

    Returns:
      The 95th-percentile / median ratio of the local-density
      field. Returns ``1.0`` for an empty or single-particle
      configuration.
    """
    if x.shape[0] < 2:
        return 1.0
    rho = _periodic_local_density(x, y, L, R)
    med = float(np.median(rho))
    if med <= 0.0:
        # Median density is zero (very dilute): fall back to the
        # mean so the ratio is still finite and >= 1.
        mean = float(np.mean(rho))
        if mean <= 0.0:
            return 1.0
        return float(np.percentile(rho, 95) / mean)
    return float(np.percentile(rho, 95) / med)


def dbscan_cluster_sizes(
    x: np.ndarray,
    y: np.ndarray,
    L: float,
    eps: float = 1.0,
    min_samples: int = 4,
) -> np.ndarray:
    """DBSCAN cluster sizes on a periodic snapshot.

    A grid-free alternative to the legacy grid-threshold
    connected-component cluster finder. DBSCAN groups particles
    that are mutually density-reachable within ``eps``; a
    particle with fewer than ``min_samples`` neighbours inside
    ``eps`` is labelled as noise and excluded.

    Periodic boundaries are handled by feeding DBSCAN a
    precomputed periodic distance matrix (``metric="precomputed"``),
    which is O(N^2) in memory and time and therefore appropriate
    only for snapshot-level cross-checks.

    Args:
      x, y: (N,) particle coordinates in ``[0, L)``.
      L: periodic box side length.
      eps: DBSCAN neighbourhood radius (default ``1.0``, of the
        order of the alignment radius).
      min_samples: DBSCAN core-point threshold (default ``4``).

    Returns:
      A 1D int array of cluster sizes (number of particles per
      DBSCAN cluster), excluding noise points. Empty array if no
      cluster reaches ``min_samples``.
    """
    from sklearn.cluster import DBSCAN

    N = x.shape[0]
    if N < min_samples:
        return np.array([], dtype=int)
    halfL = 0.5 * L
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dx -= L * np.round(dx / L)
    dy -= L * np.round(dy / L)
    dist = np.sqrt(dx * dx + dy * dy)
    labels = DBSCAN(eps=eps, min_samples=min_samples,
                    metric="precomputed").fit_predict(dist)
    sizes = [int(np.count_nonzero(labels == c))
             for c in np.unique(labels) if c != -1]
    return np.array(sizes, dtype=int)
