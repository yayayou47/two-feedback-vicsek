"""
Per-particle symmetric alpha-stable angular kicks via the
Chambers--Mallows--Stuck algorithm.

For each particle ``i`` we draw

  xi_i ~ S_{alpha_i}(eta) with skew = 0,

so a Cauchy-noise particle (``alpha = 1``) sees rare large jumps and
a Gaussian-noise particle (``alpha = 2``) sees standard kicks. The
formula is the symmetric Chambers--Mallows--Stuck representation:

  xi = scale * sin(alpha * U) / cos(U)^(1/alpha)
                * (cos((1 - alpha) * U) / W)^((1 - alpha)/alpha),

with ``U`` uniform on ``(-pi/2, pi/2)`` and ``W`` exponential of
mean 1 [Chambers, Mallows & Stuck, JASA 1976]. At ``alpha = 2`` the
formula is degenerate (``cos((1-alpha) U)^((1-alpha)/alpha)``
becomes ``cos(-U)^(-1/2)``); we substitute the exact Gaussian
``N(0, 2 * scale^2)`` instead.
"""
from __future__ import annotations

import numpy as np


def stable_rvs_vector(
    alpha_arr: np.ndarray,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one symmetric alpha-stable variate per particle.

    Args:
      alpha_arr: (N,) float array of stability indices, each in
        ``(0, 2]``. Values <= 1e-3 are clipped to 1e-3 to keep the
        algorithm well-defined; values >= 2 are routed to the exact
        Gaussian kernel.
      scale: positive float, the noise strength ``eta``.
      rng: a ``numpy.random.Generator`` (PCG64-backed by default).
        The same generator is used for all draws so seed propagation
        is unambiguous.

    Returns:
      (N,) float array of angular kicks.

    Notes:
      The Chambers--Mallows--Stuck call is fully vectorised in NumPy.
      We deliberately compute the kick *outside* the Numba loop because
      Numba's ``Generator`` support is incomplete and a global PCG64
      with vectorised draws is both faster and easier to seed.
    """
    a = np.clip(alpha_arr, 1e-3, 1.999)
    N = a.shape[0]
    U = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=N)
    W = rng.exponential(1.0, size=N)
    X = (np.sin(a * U) / np.cos(U) ** (1.0 / a)) * \
        (np.cos((1.0 - a) * U) / W) ** ((1.0 - a) / a)
    out = scale * X
    is_two = alpha_arr >= 2.0
    if is_two.any():
        gauss = rng.standard_normal(N) * (np.sqrt(2.0) * scale)
        out = np.where(is_two, gauss, out)
    return np.asarray(out, dtype=np.float64)
