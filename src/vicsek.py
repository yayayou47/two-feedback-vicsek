"""
Vicsek-Couzin 2D model with optional alpha-stable angular noise.

Each particle has three behavioural zones (Couzin et al., J. Theor.
Biol. 2002), within a vision cone (blind sector at the rear):

  * repulsion zone   d < R_r          : turn away from neighbours
  * alignment zone   R_r <= d < R_a   : align with neighbours' heading
  * outside R_a                       : no interaction

Decision order: repulsion is checked first; if at least one repulsion
neighbour is visible, the particle turns away from them (sum of unit
vectors pointing away). Otherwise, if at least one alignment neighbour
is visible, the particle copies their mean heading. Otherwise, the
particle keeps its previous heading. Symmetric alpha-stable noise is
then added in all cases.

xi_i ~ S_alpha(scale=eta, beta=0). alpha=2 recovers Gaussian; alpha<2
introduces heavy tails ("Levy-Vicsek-Couzin").

Spatial neighbour search uses a linked-cell list, O(N) per step. Hot
loops are JIT-compiled with Numba.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(f):
            return f

        return deco

    def prange(*args, **kwargs):
        return range(*args, **kwargs)

from noise import stable_rvs


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _zonal_update(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    L: float,
    R_r: float,
    R_a: float,
    blind_cos: float,
    head: np.ndarray,
    nxt: np.ndarray,
    n_cell: int,
) -> np.ndarray:
    """Repulsion / alignment / inertia update with vision cone."""
    N = x.shape[0]
    new_theta = np.empty(N)
    cell_size = L / n_cell
    Rr2 = R_r * R_r
    Ra2 = R_a * R_a
    halfL = 0.5 * L

    for i in prange(N):
        ci = int(x[i] / cell_size) % n_cell
        cj = int(y[i] / cell_size) % n_cell
        hcx = np.cos(theta[i])
        hcy = np.sin(theta[i])
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
                            if (hcx * ux + hcy * uy) >= blind_cos:
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
    return new_theta


@njit(cache=True)
def _build_cells(
    x: np.ndarray, y: np.ndarray, L: float, n_cell: int
) -> tuple[np.ndarray, np.ndarray]:
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


def _zonal_update_np(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    L: float,
    R_r: float,
    R_a: float,
    blind_cos: float,
) -> np.ndarray:
    """Vectorised O(N^2) NumPy fallback when Numba is unavailable."""
    halfL = 0.5 * L
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    dx -= L * np.round(dx / L)
    dy -= L * np.round(dy / L)
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    with np.errstate(invalid="ignore"):
        ux = dx / d
        uy = dy / d
    hcx = np.cos(theta)[:, None]
    hcy = np.sin(theta)[:, None]
    visible = (hcx * ux + hcy * uy) >= blind_cos
    rep_mask = visible & (d2 < R_r * R_r)
    ali_mask = visible & (d2 >= R_r * R_r) & (d2 < R_a * R_a)
    n_rep = rep_mask.sum(axis=1)
    rep_x = -(rep_mask * ux).sum(axis=1)
    rep_y = -(rep_mask * uy).sum(axis=1)
    s = np.sin(theta)
    c = np.cos(theta)
    ali_sx = ali_mask @ s
    ali_cx = ali_mask @ c
    n_ali = ali_mask.sum(axis=1)
    new_theta = theta.copy()
    use_rep = n_rep > 0
    use_ali = (~use_rep) & (n_ali > 0)
    new_theta[use_rep] = np.arctan2(rep_y[use_rep], rep_x[use_rep])
    new_theta[use_ali] = np.arctan2(ali_sx[use_ali], ali_cx[use_ali])
    return new_theta


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
@dataclass
class VicsekParams:
    N: int = 2000
    L: float = 15.0
    v0: float = 0.05
    R_r: float = 0.5         # repulsion radius (absolute)
    R_a: float = 0.7         # alignment outer radius (absolute)
    beta: float = 30.0       # blind-sector full angular width (degrees)
    eta: float = 0.3
    alpha: float = 2.0       # Levy stability index of the angular noise
    seed: int = 0


class Vicsek:
    def __init__(self, p: VicsekParams):
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        self.x = self.rng.uniform(0, p.L, size=p.N)
        self.y = self.rng.uniform(0, p.L, size=p.N)
        self.theta = self.rng.uniform(-np.pi, np.pi, size=p.N)
        self.R_r = p.R_r
        self.R_a = p.R_a
        # beta is the full angular width of the rear blind sector. A
        # neighbour at angular offset psi from the heading is visible iff
        # |psi| <= 180 - beta/2, i.e. cos(psi) >= -cos(beta/2).
        half = np.deg2rad(0.5 * p.beta)
        self.blind_cos = -np.cos(half)
        self.n_cell = max(1, int(p.L / max(self.R_a, 1e-9)))
        self.t = 0

    def step(self) -> None:
        p = self.p
        if _HAS_NUMBA:
            head, nxt = _build_cells(self.x, self.y, p.L, self.n_cell)
            target = _zonal_update(
                self.x, self.y, self.theta, p.L,
                self.R_r, self.R_a, self.blind_cos,
                head, nxt, self.n_cell,
            )
        else:
            target = _zonal_update_np(
                self.x, self.y, self.theta, p.L,
                self.R_r, self.R_a, self.blind_cos,
            )
        xi = stable_rvs(p.alpha, p.eta, p.N, self.rng)
        self.theta = (target + xi + np.pi) % (2 * np.pi) - np.pi
        self.x = (self.x + p.v0 * np.cos(self.theta)) % p.L
        self.y = (self.y + p.v0 * np.sin(self.theta)) % p.L
        self.t += 1

    def polarisation(self) -> float:
        return float(np.hypot(np.mean(np.cos(self.theta)), np.mean(np.sin(self.theta))))

    def state(self) -> dict:
        return dict(x=self.x.copy(), y=self.y.copy(), theta=self.theta.copy(), t=self.t)
