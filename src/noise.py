"""
Alpha-stable (Levy) noise generators.

Reference: Chambers, Mallows & Stuck (1976), "A Method for Simulating
Stable Random Variables", JASA 71(354).

We generate symmetric alpha-stable variates S_alpha(scale=c, beta=0, mu=0).
For alpha = 2 this reduces to a Gaussian with std = sqrt(2)*c.
For alpha = 1 it is Cauchy with HWHM = c.

In the Vicsek model the noise is added to the angle modulo 2*pi. Heavy tails
allow large angular jumps -> "Levy flights in orientation space".
"""
from __future__ import annotations

import numpy as np


def stable_rvs(alpha: float, scale: float, size, rng: np.random.Generator) -> np.ndarray:
    """Symmetric alpha-stable random variates (beta=0, location=0).

    Parameters
    ----------
    alpha : float in (0, 2]
        Stability index. alpha=2 -> Gaussian, alpha=1 -> Cauchy.
    scale : float >= 0
        Scale parameter c.
    size : int or tuple
        Output shape.
    rng : np.random.Generator
        Source of randomness.
    """
    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2]")
    if alpha == 2.0:
        # Special case: Gaussian with std = sqrt(2)*scale (parametrization S2).
        return rng.standard_normal(size) * (np.sqrt(2.0) * scale)

    # Chambers-Mallows-Stuck for symmetric stable (beta = 0).
    U = rng.uniform(-np.pi / 2, np.pi / 2, size=size)
    W = rng.exponential(1.0, size=size)
    X = (np.sin(alpha * U) / np.cos(U) ** (1.0 / alpha)) * (
        np.cos((1.0 - alpha) * U) / W
    ) ** ((1.0 - alpha) / alpha)
    return scale * X


def wrapped_stable_rvs(
    alpha: float, scale: float, size, rng: np.random.Generator
) -> np.ndarray:
    """Stable variate wrapped to (-pi, pi]."""
    x = stable_rvs(alpha, scale, size, rng)
    return (x + np.pi) % (2 * np.pi) - np.pi
