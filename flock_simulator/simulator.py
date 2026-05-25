"""
Top-level Simulator class for the omnidirectional two-feedback
Vicsek--Couzin model.

Orchestrates the per-step pipeline:

  1. build a periodic linked-cell list at cell side R_a;
  2. compute the Couzin priority heading and the alignment-zone
     neighbour count via core.interactions.zonal_update (no blind
     cone);
  3. apply the shared sigmoid to get per-particle (v_i, alpha_i)
     via core.adaptivity.shared_sigmoid_fields;
  4. draw N alpha-stable angular kicks via core.noise.stable_rvs_vector
     and add them to the heading;
  5. advect positions by v_i along the new heading, with periodic
     wrapping.

Per-particle state lives in pre-allocated NumPy arrays owned by the
Simulator (no per-step allocations beyond the two cell-list arrays
and the temporary kick vector). The PCG64 random generator is owned
by the Simulator and seeded once at construction so each
(seed, mode, L) tuple is reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core.adaptivity import shared_sigmoid_fields
from .core.geometry import build_cell_list, normalize_angle
from .core.interactions import zonal_update, zonal_update_topological
from .core.noise import stable_rvs_vector
from .observables.order import polarisation
from .observables.spatial import density_separation_index


@dataclass
class FlockParams:
    """Static parameters of one flock run.

    Defaults match the ``full`` mode of the manuscript at the
    pilot working point: shared sigmoid with ``n_star = 3``,
    ``slope = 2``, both feedbacks active in ``[v_min, v_max] =
    [0.005, 0.05]`` and ``[alpha_min, alpha_max] = [1, 2]``.

    Decoupled sigmoids: by default the speed and the
    stability index share one sigmoid through ``(n_star,
    slope)``. To run a *decoupled* variant (separate threshold
    or slope for the motility and noise-shape channels), set
    one or more of ``n_star_v``, ``slope_v``, ``n_star_alpha``,
    ``slope_alpha`` to a non-None value; whichever override is
    omitted falls back to the shared ``(n_star, slope)``.
    """
    N: int = 500
    L: float = 15.0
    v_max: float = 0.05
    v_min: float = 0.005
    alpha_min: float = 1.0
    alpha_max: float = 2.0
    R_r: float = 0.5
    R_a: float = 0.7
    eta: float = 0.10
    n_star: float = 3.0
    slope: float = 2.0
    seed: int = 0
    n_star_v: float | None = None
    slope_v: float | None = None
    n_star_alpha: float | None = None
    slope_alpha: float | None = None
    # Alignment kernel: "metric" (default, circular mean over the
    # alignment annulus) or "topological" (circular mean over the
    # ``k_NN`` nearest non-repulsion neighbours scanned in the 5x5
    # cell stencil). The shared-sigmoid input n_ali is always the
    # metric annulus count, so the density-sensing channel is
    # identical across the two kernels.
    alignment: str = "metric"
    k_NN: int = 4

    def resolved_v_sigmoid(self) -> tuple[float, float]:
        """``(n_star, slope)`` used by the motility channel."""
        return (self.n_star if self.n_star_v is None else self.n_star_v,
                self.slope if self.slope_v is None else self.slope_v)

    def resolved_alpha_sigmoid(self) -> tuple[float, float]:
        """``(n_star, slope)`` used by the noise-shape channel."""
        return (self.n_star if self.n_star_alpha is None else self.n_star_alpha,
                self.slope if self.slope_alpha is None else self.slope_alpha)


@dataclass
class FlockState:
    """Mutable per-step state of one flock run."""
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    v_i: np.ndarray
    alpha_i: np.ndarray
    n_ali: np.ndarray
    t: int = 0


class FlockSimulator:
    """Omnidirectional two-feedback Vicsek--Couzin simulator.

    Args:
      p: a ``FlockParams`` describing the run.

    Attributes:
      p: the input parameters.
      rng: the NumPy ``Generator`` (PCG64) seeded with ``p.seed``.
      state: the current ``FlockState``.
      n_cell: number of cells per side; chosen so the cell side is
        at least ``p.R_a``.
    """

    def __init__(self, p: FlockParams):
        if p.v_min > p.v_max:
            raise ValueError("v_min must be <= v_max")
        if p.alpha_min > p.alpha_max:
            raise ValueError("alpha_min must be <= alpha_max")
        if p.alpha_min <= 0.0 or p.alpha_max > 2.0:
            raise ValueError("alpha_min, alpha_max must be in (0, 2]")
        if p.R_r >= p.R_a:
            raise ValueError("R_r must be strictly < R_a")
        if p.L <= 0.0:
            raise ValueError("L must be positive")
        if p.alignment not in ("metric", "topological"):
            raise ValueError(
                "alignment must be 'metric' or 'topological'")
        if p.alignment == "topological" and p.k_NN < 1:
            raise ValueError("k_NN must be >= 1 for topological alignment")
        self.p = p
        self.rng = np.random.default_rng(np.random.SeedSequence(p.seed))
        x = self.rng.uniform(0.0, p.L, size=p.N)
        y = self.rng.uniform(0.0, p.L, size=p.N)
        theta = self.rng.uniform(-np.pi, np.pi, size=p.N)
        v_i = np.full(p.N, 0.5 * (p.v_max + p.v_min))
        alpha_i = np.full(p.N, 0.5 * (p.alpha_min + p.alpha_max))
        n_ali = np.zeros(p.N, dtype=np.int64)
        self.state = FlockState(x, y, theta, v_i, alpha_i, n_ali, t=0)
        self.n_cell = max(1, int(p.L / max(p.R_a, 1e-9)))

    # --- per-step pipeline -------------------------------------------------

    def step(self) -> None:
        """Advance the flock by one timestep.

        Pipeline: cell-list -> Couzin priority heading + neighbour
        count -> shared sigmoid -> alpha-stable kick -> advection.
        Mutates ``self.state`` in place; nothing is returned.
        """
        p = self.p
        s = self.state
        head, nxt = build_cell_list(s.x, s.y, p.L, self.n_cell)
        if p.alignment == "metric":
            target, n_ali = zonal_update(
                s.x, s.y, s.theta, p.L,
                p.R_r, p.R_a, head, nxt, self.n_cell,
            )
        else:   # topological
            target, n_ali = zonal_update_topological(
                s.x, s.y, s.theta, p.L,
                p.R_r, p.R_a, p.k_NN,
                head, nxt, self.n_cell,
            )
        ns_v, sl_v = p.resolved_v_sigmoid()
        ns_a, sl_a = p.resolved_alpha_sigmoid()
        if (ns_v == ns_a) and (sl_v == sl_a):
            v, alpha = shared_sigmoid_fields(
                n_ali,
                ns_v, sl_v,
                p.v_min, p.v_max,
                p.alpha_min, p.alpha_max,
            )
        else:
            v, _ = shared_sigmoid_fields(
                n_ali, ns_v, sl_v,
                p.v_min, p.v_max,
                p.alpha_min, p.alpha_max,
            )
            _, alpha = shared_sigmoid_fields(
                n_ali, ns_a, sl_a,
                p.v_min, p.v_max,
                p.alpha_min, p.alpha_max,
            )
        xi = stable_rvs_vector(alpha, p.eta, self.rng)
        new_theta = (target + xi + np.pi) % (2.0 * np.pi) - np.pi
        s.theta[:] = new_theta
        s.x[:] = (s.x + v * np.cos(new_theta)) % p.L
        s.y[:] = (s.y + v * np.sin(new_theta)) % p.L
        s.v_i[:] = v
        s.alpha_i[:] = alpha
        s.n_ali[:] = n_ali
        s.t += 1

    # --- snapshot observables ---------------------------------------------

    def polarisation(self) -> float:
        """Current value of ``|<exp(i theta)>|``."""
        return polarisation(self.state.theta)

    def density_separation_index(self, n_bins: int = 10) -> float:
        """Current value of ``s_sep`` on an ``n_bins x n_bins`` grid."""
        return density_separation_index(
            self.state.x, self.state.y, self.p.L, n_bins,
        )
