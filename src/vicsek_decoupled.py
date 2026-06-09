"""
Decoupled-sigmoid Vicsek-Couzin model: the same two density-adaptive
feedbacks as DoubleAdaptiveVicsek, but with the motility and noise-shape
channels gated by two independent sigmoids of the neighbour count n_i,
parameterised by (n_star_v, slope_v) and (n_star_alpha, slope_alpha).
DecoupledParams configures the run and the DecoupledVicsek class exposes
step(), polarisation() and density_separation_index(). Setting both
sigmoid pairs equal recovers the shared-sigmoid DoubleAdaptiveVicsek.
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
from vicsek_double_adaptive import (_zonal_update_with_count,
                                     stable_rvs_vector)


@dataclass
class DecoupledParams:
    N: int = 500
    L: float = 15.0
    v_max: float = 0.05
    v_min: float = 0.005
    alpha_min: float = 1.0
    alpha_max: float = 2.0
    R_r: float = 0.5
    R_a: float = 0.7
    beta: float = 30.0
    eta: float = 0.10
    # Two independent sigmoids:
    n_star_v: float = 3.0
    slope_v: float = 2.0
    n_star_alpha: float = 3.0
    slope_alpha: float = 2.0
    seed: int = 0


class DecoupledVicsek:
    """Vicsek-Couzin with two independent sigmoid feedbacks."""

    def __init__(self, p: DecoupledParams):
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
        # Independent sigmoid evaluations.
        zv = (n_ali - p.n_star_v) * p.slope_v
        za = (n_ali - p.n_star_alpha) * p.slope_alpha
        sig_v = 1.0 / (1.0 + np.exp(-zv))
        sig_a = 1.0 / (1.0 + np.exp(-za))
        self.v_i = p.v_max - (p.v_max - p.v_min) * sig_v
        self.alpha_i = p.alpha_min + (p.alpha_max - p.alpha_min) * sig_a
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
