"""
Periodic-boundary geometry helpers for the omnidirectional
two-feedback Vicsek--Couzin simulator.

Provides:
  - normalize_angle: wrap an angle to [-pi, pi].
  - build_cell_list: O(N) linked-cell list for a square periodic box.

The cell size is fixed at the alignment radius R_a (the largest
interaction range), which is the smallest value that lets a 3x3
cell stencil cover every neighbour without distance double-counting.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a: object, **k: object) -> object:
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda f: f


@njit(cache=True)
def normalize_angle(theta: float) -> float:
    """Wrap an angle to the half-open interval ``[-pi, pi)``.

    Args:
      theta: any real angle in radians.

    Returns:
      The same angle reduced modulo ``2*pi`` into ``[-pi, pi)``.

    Complexity: O(1).
    """
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


@njit(cache=True)
def build_cell_list(
    x: np.ndarray, y: np.ndarray, L: float, n_cell: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a periodic linked-cell list for ``N`` particles.

    Args:
      x: (N,) float array, particle x-coordinates in ``[0, L)``.
      y: (N,) float array, particle y-coordinates in ``[0, L)``.
      L: side length of the square periodic box.
      n_cell: number of cells per side. Cell side is ``L / n_cell``,
        and the caller must enforce ``L / n_cell >= R_a`` so that the
        3x3 stencil covers every interacting neighbour.

    Returns:
      head: (n_cell**2,) int64 array; ``head[c]`` is the index of the
        first particle in cell ``c`` (or -1 if empty).
      nxt:  (N,) int64 array; ``nxt[i]`` is the next particle index in
        the same cell as ``i`` (or -1 if ``i`` is the last).

    Complexity: O(N).

    Invariants:
      - traversing ``head[c]`` then ``nxt[k]`` until -1 visits every
        particle in cell ``c`` exactly once;
      - every particle index in ``[0, N)`` appears in exactly one cell.
    """
    N = x.shape[0]
    cell_size = L / n_cell
    head = -np.ones(n_cell * n_cell, dtype=np.int64)
    nxt = -np.ones(N, dtype=np.int64)
    for i in range(N):
        ci = int(x[i] / cell_size) % n_cell
        cj = int(y[i] / cell_size) % n_cell
        c = ci * n_cell + cj
        nxt[i] = head[c]
        head[c] = i
    return head, nxt
