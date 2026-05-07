"""
Vicsek-Couzin variant with two density-adaptive feedbacks gated by
the same local neighbour count n_i:

    sig_i  = 1 / (1 + exp(-(n_i - n_star) * slope))
    v_i    = v_max   - (v_max - v_min)         * sig_i      (motility)
    alpha_i = alpha_min + (alpha_max - alpha_min) * sig_i   (noise shape)

The two channels are tied: a crowded particle is simultaneously slow
and noisy-Gaussian-like; an isolated particle is simultaneously fast
and noisy-Cauchy-like. Setting v_min = v_max disables the motility
channel; setting alpha_min = alpha_max disables the noise-shape
channel; one simulator therefore covers the four modes
{baseline, v2-limit, v3-limit, full} with a single code path.

Angular kicks are drawn particle-by-particle from S_{alpha_i}(eta).
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

from vicsek import _build_cells


@njit(cache=True, parallel=True)
def _zonal_update_with_count(
    x, y, theta, L, R_r, R_a, blind_cos, head, nxt, n_cell,
):
    """Couzin priority rule + visible alignment-zone neighbour count."""
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
        n_ali_arr[i] = n_ali
    return new_theta, n_ali_arr


def stable_rvs_vector(alpha_arr: np.ndarray, scale: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Symmetric alpha-stable variates, per-particle alpha_i.
    Chambers-Mallows-Stuck for alpha < 2; Gaussian fallback at 2."""
    a = np.clip(alpha_arr, 1e-3, 1.999)
    N = a.shape[0]
    U = rng.uniform(-np.pi / 2, np.pi / 2, size=N)
    W = rng.exponential(1.0, size=N)
    X = (np.sin(a * U) / np.cos(U) ** (1.0 / a)) * \
        (np.cos((1.0 - a) * U) / W) ** ((1.0 - a) / a)
    out = scale * X
    is_two = alpha_arr >= 2.0
    if is_two.any():
        gauss = rng.standard_normal(N) * (np.sqrt(2.0) * scale)
        out = np.where(is_two, gauss, out)
    return out


@dataclass
class DoubleAdaptiveParams:
    N: int = 500
    L: float = 15.0
    v_max: float = 0.05
    v_min: float = 0.005       # set v_min=v_max to freeze the motility channel
    alpha_min: float = 1.0     # set alpha_min=alpha_max to freeze the noise-shape channel
    alpha_max: float = 2.0
    R_r: float = 0.5
    R_a: float = 0.7
    beta: float = 30.0
    eta: float = 0.10
    n_star: float = 3.0
    slope: float = 2.0
    seed: int = 0


class DoubleAdaptiveVicsek:
    """Two-feedback Vicsek-Couzin: simultaneous v_i(n_i) and alpha_i(n_i)."""

    def __init__(self, p: DoubleAdaptiveParams):
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        self.x = self.rng.uniform(0, p.L, size=p.N)
        self.y = self.rng.uniform(0, p.L, size=p.N)
        self.theta = self.rng.uniform(-np.pi, np.pi, size=p.N)
        half = np.deg2rad(0.5 * p.beta)
        self.blind_cos = -np.cos(half)
        self.n_cell = max(1, int(p.L / max(p.R_a, 1e-9)))
        self.t = 0
        self.v_i = np.full(p.N, 0.5 * (p.v_max + p.v_min))
        self.alpha_i = np.full(p.N, 0.5 * (p.alpha_min + p.alpha_max))

    def step(self) -> None:
        p = self.p
        head, nxt = _build_cells(self.x, self.y, p.L, self.n_cell)
        target, n_ali = _zonal_update_with_count(
            self.x, self.y, self.theta, p.L,
            p.R_r, p.R_a, self.blind_cos,
            head, nxt, self.n_cell,
        )
        z = (n_ali - p.n_star) * p.slope
        sig = 1.0 / (1.0 + np.exp(-z))
        self.v_i = p.v_max - (p.v_max - p.v_min) * sig
        self.alpha_i = p.alpha_min + (p.alpha_max - p.alpha_min) * sig
        xi = stable_rvs_vector(self.alpha_i, p.eta, self.rng)
        self.theta = (target + xi + np.pi) % (2 * np.pi) - np.pi
        self.x = (self.x + self.v_i * np.cos(self.theta)) % p.L
        self.y = (self.y + self.v_i * np.sin(self.theta)) % p.L
        self.t += 1

    def polarisation(self) -> float:
        return float(np.hypot(np.mean(np.cos(self.theta)),
                              np.mean(np.sin(self.theta))))

    def density_separation_index(self, n_bins: int = 10) -> float:
        H, _, _ = np.histogram2d(
            self.x, self.y,
            bins=[n_bins, n_bins],
            range=[[0, self.p.L], [0, self.p.L]],
        )
        H = H.flatten()
        H = H[H > 0]
        if len(H) == 0:
            return 1.0
        return float(H.max() / np.median(H))
