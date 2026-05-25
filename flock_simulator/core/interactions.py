"""
Couzin priority interaction kernels for the omnidirectional
two-feedback Vicsek--Couzin simulator.

Two alignment kernels are provided:

  * ``zonal_update`` (metric): the alignment direction is the
    circular mean of every neighbour heading in the annulus
    ``R_r <= |r_ij| < R_a``. The number of alignment-zone
    neighbours ``n_ali`` (the field the shared sigmoid reads)
    is the metric count in the same annulus.

  * ``zonal_update_topological`` (topological): the alignment
    direction is the circular mean of the ``k`` nearest
    non-repulsion neighbours, selected from the same 5x5 cell
    stencil. The sigmoid input ``n_ali`` is still the metric
    annulus count, so the density-sensing channel is held fixed
    and only the alignment kernel changes -- a robustness test
    of the metric-alignment story.

Both kernels share the metric repulsion zone ``|r_ij| < R_r``
and the Couzin priority (repulsion > alignment > inertia).
Both are omnidirectional: no blind-sector filter is applied.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    def njit(*a: object, **k: object) -> object:
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda f: f

    def prange(*a: int, **k: int) -> range:
        return range(*a, **k)


@njit(cache=True, parallel=True)
def zonal_update(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    L: float,
    R_r: float,
    R_a: float,
    head: np.ndarray,
    nxt: np.ndarray,
    n_cell: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one step of the omnidirectional Couzin priority rule.

    Args:
      x, y, theta: (N,) arrays of positions and headings.
      L: periodic box side length.
      R_r, R_a: repulsion and alignment radii (must satisfy
        ``0 < R_r < R_a``).
      head, nxt: linked-cell-list arrays from
        ``build_cell_list(x, y, L, n_cell)``.
      n_cell: number of cells per side. The cell side ``L / n_cell``
        must be at least ``R_a`` so that the 3x3 stencil covers
        every interacting neighbour.

    Returns:
      new_theta: (N,) float array of headings *before* the angular
        kick is added. Range is the principal value returned by
        ``np.arctan2``, i.e. ``(-pi, pi]``.
      n_ali: (N,) int64 array, alignment-zone neighbour count for
        each particle. This is the field the shared sigmoid reads.

    Complexity: O(N) at fixed density (cell-list).
    """
    N = x.shape[0]
    new_theta = np.empty(N)
    n_ali_arr = np.zeros(N, dtype=np.int64)
    cell_size = L / n_cell
    Rr2 = R_r * R_r
    Ra2 = R_a * R_a
    halfL = 0.5 * L

    for i in prange(N):
        ci = int(x[i] / cell_size) % n_cell
        cj = int(y[i] / cell_size) % n_cell
        rep_x = 0.0
        rep_y = 0.0
        n_rep = 0
        ali_sx = 0.0
        ali_cx = 0.0
        n_ali = 0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni = (ci + di) % n_cell
                nj = (cj + dj) % n_cell
                k = head[ni * n_cell + nj]
                while k != -1:
                    if k != i:
                        dx = x[k] - x[i]
                        dy = y[k] - y[i]
                        if dx > halfL:
                            dx -= L
                        elif dx < -halfL:
                            dx += L
                        if dy > halfL:
                            dy -= L
                        elif dy < -halfL:
                            dy += L
                        d2 = dx * dx + dy * dy
                        if 0.0 < d2 < Ra2:
                            d = np.sqrt(d2)
                            ux = dx / d
                            uy = dy / d
                            if d2 < Rr2:
                                rep_x -= ux
                                rep_y -= uy
                                n_rep += 1
                            else:
                                ali_sx += np.sin(theta[k])
                                ali_cx += np.cos(theta[k])
                                n_ali += 1
                    k = nxt[k]
        if n_rep > 0:
            new_theta[i] = np.arctan2(rep_y, rep_x)
        elif n_ali > 0:
            new_theta[i] = np.arctan2(ali_sx, ali_cx)
        else:
            new_theta[i] = theta[i]
        n_ali_arr[i] = n_ali
    return new_theta, n_ali_arr


@njit(cache=True, parallel=True)
def zonal_update_topological(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    L: float,
    R_r: float,
    R_a: float,
    k_NN: int,
    head: np.ndarray,
    nxt: np.ndarray,
    n_cell: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Couzin priority with topological (k-NN) alignment.

    Repulsion is still metric (within R_r). Alignment is the
    circular mean of the ``k_NN`` nearest non-repulsion
    neighbours, scanned within the 5x5 cell stencil (search
    radius ~2.5 R_a). The sigmoid input ``n_ali`` is the metric
    alignment-zone count, identical to the metric kernel, so
    the density-sensing channel does not change between the
    two alignment kernels.

    Args:
      x, y, theta: (N,) arrays of positions and headings.
      L: periodic box side length.
      R_r, R_a: repulsion and metric alignment radii.
      k_NN: number of nearest neighbours used by the alignment
        kernel. Typical value 4 (Ginelli--Chat\\'e topological
        Vicsek). When a particle has fewer than ``k_NN``
        non-repulsion neighbours in the 5x5 stencil, it aligns
        with whatever it does have.
      head, nxt: linked-cell list from ``build_cell_list``;
        the caller must use ``cell_size >= R_a``.
      n_cell: number of cells per side.

    Returns:
      new_theta: (N,) headings before the angular kick.
      n_ali: (N,) metric annulus count, same definition as the
        metric kernel; this is the field the shared sigmoid
        reads.
    """
    N = x.shape[0]
    new_theta = np.empty(N)
    n_ali_arr = np.zeros(N, dtype=np.int64)
    cell_size = L / n_cell
    Rr2 = R_r * R_r
    Ra2 = R_a * R_a
    halfL = 0.5 * L
    # Per-thread buffer for k-NN selection. We allocate one extra
    # slot to host candidates before the truncation to k_NN.
    K_BUF = 64

    for i in prange(N):
        ci = int(x[i] / cell_size) % n_cell
        cj = int(y[i] / cell_size) % n_cell
        rep_x = 0.0
        rep_y = 0.0
        n_rep = 0
        n_ali = 0
        # Candidates for k-NN alignment: distance squared and partner
        # index, kept in two parallel small buffers; we will sort by
        # distance and pick the k_NN smallest.
        cand_d2 = np.empty(K_BUF, dtype=np.float64)
        cand_k = np.empty(K_BUF, dtype=np.int64)
        n_cand = 0
        for di in range(-2, 3):                  # 5x5 stencil
            for dj in range(-2, 3):
                ni = (ci + di) % n_cell
                nj = (cj + dj) % n_cell
                kk = head[ni * n_cell + nj]
                while kk != -1:
                    if kk != i:
                        dx = x[kk] - x[i]
                        dy = y[kk] - y[i]
                        if dx > halfL:
                            dx -= L
                        elif dx < -halfL:
                            dx += L
                        if dy > halfL:
                            dy -= L
                        elif dy < -halfL:
                            dy += L
                        d2 = dx * dx + dy * dy
                        if 0.0 < d2 < Rr2:
                            d = np.sqrt(d2)
                            rep_x -= dx / d
                            rep_y -= dy / d
                            n_rep += 1
                        else:
                            if d2 < Ra2:
                                n_ali += 1
                            # k-NN candidate (any non-repulsion neighbour
                            # within the 5x5 stencil that lies outside R_r).
                            if d2 >= Rr2 and n_cand < K_BUF:
                                cand_d2[n_cand] = d2
                                cand_k[n_cand] = kk
                                n_cand += 1
                    kk = nxt[kk]

        if n_rep > 0:
            new_theta[i] = np.arctan2(rep_y, rep_x)
        elif n_cand > 0:
            # Pick the k_NN smallest distances by simple selection sort;
            # n_cand is small (typically <= 20).
            n_pick = k_NN if n_cand >= k_NN else n_cand
            for s in range(n_pick):
                jmin = s
                for t in range(s + 1, n_cand):
                    if cand_d2[t] < cand_d2[jmin]:
                        jmin = t
                if jmin != s:
                    tmp_d = cand_d2[s]; cand_d2[s] = cand_d2[jmin]; cand_d2[jmin] = tmp_d
                    tmp_k = cand_k[s];  cand_k[s] = cand_k[jmin];  cand_k[jmin] = tmp_k
            ali_sx = 0.0
            ali_cx = 0.0
            for s in range(n_pick):
                ali_sx += np.sin(theta[cand_k[s]])
                ali_cx += np.cos(theta[cand_k[s]])
            new_theta[i] = np.arctan2(ali_sx, ali_cx)
        else:
            new_theta[i] = theta[i]
        n_ali_arr[i] = n_ali
    return new_theta, n_ali_arr
