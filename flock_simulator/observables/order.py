"""
Global polar-order observables: ``phi = |<exp(i theta)>|``, the
susceptibility ``chi = N * (var(phi)) / mean(phi)`` over a
trajectory, and the Binder cumulant ``U_4``.

Each function takes a 1D array of polarisation samples (one entry
per measurement step) and returns a single scalar.
"""
from __future__ import annotations

import numpy as np


def polarisation(theta: np.ndarray) -> float:
    """Magnitude of the mean orientation vector.

    Args:
      theta: (N,) array of headings.

    Returns:
      ``|<cos theta> + i <sin theta>|`` in ``[0, 1]``.
    """
    return float(np.hypot(np.mean(np.cos(theta)),
                          np.mean(np.sin(theta))))


def susceptibility(phi_trace: np.ndarray, N: int) -> float:
    """Vicsek susceptibility ``N * Var(phi)`` along a trajectory.

    Args:
      phi_trace: (n_meas,) sequence of polarisations.
      N: number of particles.

    Returns:
      ``N * variance(phi_trace)`` (population variance, divisor ``n``).
    """
    return float(N * np.var(phi_trace))


def binder_cumulant(phi_trace: np.ndarray) -> float:
    """Binder fourth-order cumulant ``1 - <phi^4>/(3 <phi^2>^2)``.

    Args:
      phi_trace: (n_meas,) sequence of polarisations.

    Returns:
      ``U_4``. Returns 0.0 (Gaussian limit) when ``<phi^2>`` is zero.
    """
    m2 = float(np.mean(phi_trace ** 2))
    if m2 <= 0.0:
        return 0.0
    m4 = float(np.mean(phi_trace ** 4))
    return 1.0 - m4 / (3.0 * m2 * m2)
