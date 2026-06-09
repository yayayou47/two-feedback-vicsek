"""
Symmetric alpha-stable (Levy) noise generators via the Chambers-Mallows-
Stuck method (JASA 71(354), 1976). stable_rvs draws variates
S_alpha(scale=c, beta=0, mu=0); alpha=2 gives a Gaussian with
std=sqrt(2)*c and alpha=1 gives Cauchy with HWHM=c. wrapped_stable_rvs
wraps the output to (-pi, pi] for use as angular noise in the Vicsek
models, where alpha<2 produces heavy-tailed orientation kicks.
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
