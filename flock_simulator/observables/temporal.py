"""
Temporal observables: heading autocorrelation ``C(tau)`` for a
single trajectory of unit-vector-valued angles.

For a per-particle heading trace ``theta[i, t]``, the per-particle
autocorrelation is

  c_i(tau) = <cos(theta_i(t + tau) - theta_i(t))>_t,

and ``C(tau)`` is the population mean over the chosen subset of
particles (e.g. dense quartile vs dilute quartile).
"""
from __future__ import annotations

import numpy as np


def heading_autocorrelation(
    theta_trace: np.ndarray,
    taus: np.ndarray,
) -> np.ndarray:
    """Per-particle, time-averaged heading autocorrelation.

    Args:
      theta_trace: (n_part, n_t) float array of headings.
      taus: 1D int array of lags in steps. Must satisfy
        ``0 <= max(taus) < n_t``.

    Returns:
      C: (len(taus),) float array, ``< <cos(d_theta)>_t >_part`` at
        each lag.

    Complexity: O(n_part * n_t * len(taus)). The implementation is
    a vectorised numpy diff so it is comfortable up to ~10^4 samples.
    """
    n_part, n_t = theta_trace.shape
    out = np.zeros(len(taus), dtype=np.float64)
    for k, tau in enumerate(taus):
        if tau == 0:
            out[k] = 1.0
            continue
        d = theta_trace[:, tau:] - theta_trace[:, :-tau]
        out[k] = float(np.mean(np.cos(d)))
    return out
