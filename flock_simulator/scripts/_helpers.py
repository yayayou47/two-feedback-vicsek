"""
Shared helpers for the no-cone Stage-3 re-runs.

Each legacy script lived in version4/src/run_double_*.py and used
the with-cone simulator. Their no-cone ports here all share the
same skeleton: build a FlockParams, instantiate FlockSimulator,
warm up, and measure ``phi`` and ``s_sep`` (and optionally the
population mean speed and stability index) over a trajectory.
The helpers in this module factor out that skeleton so the
ports are short.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flock_simulator import FlockParams, FlockSimulator      # noqa: E402

DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def measure_full(sim: FlockSimulator, n_meas: int, every: int = 1):
    """Step the simulator ``n_meas`` times and record observables.

    Returns four 1D arrays (phi, s_sep, v_mean, alpha_mean) of
    length ``n_meas // every``.
    """
    n = n_meas // every
    phi = np.empty(n)
    s_sep = np.empty(n)
    v = np.empty(n)
    a = np.empty(n)
    j = 0
    for k in range(n_meas):
        sim.step()
        if k % every == 0:
            phi[j] = sim.polarisation()
            s_sep[j] = sim.density_separation_index(n_bins=10)
            v[j] = float(sim.state.v_i.mean())
            a[j] = float(sim.state.alpha_i.mean())
            j += 1
    return phi[:j], s_sep[:j], v[:j], a[:j]


def measure_phi_sep(sim: FlockSimulator, n_meas: int):
    """Step ``n_meas`` times and record (phi, s_sep) only."""
    phi = np.empty(n_meas)
    s_sep = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        phi[k] = sim.polarisation()
        s_sep[k] = sim.density_separation_index(n_bins=10)
    return phi, s_sep


def measure_phi(sim: FlockSimulator, n_meas: int):
    """Step ``n_meas`` times and record ``phi`` only."""
    out = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        out[k] = sim.polarisation()
    return out


def warm(sim: FlockSimulator, n_warm: int, *, theta_zero: bool = True):
    """Warm up a freshly constructed simulator."""
    if theta_zero:
        sim.state.theta[:] = 0.0
    for _ in range(n_warm):
        sim.step()
