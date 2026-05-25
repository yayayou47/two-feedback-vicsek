"""
Side-by-side comparison of the legacy (with blind-cone) simulator
and the new omnidirectional simulator on the validation grid
(L=30, eta=0.10, N=2000, 500 warm + 1000 measurement steps,
10 seeds).

Reports the per-mode mean and std of <phi> and s_sep, the absolute
and relative differences, and writes a CSV summary that the
refactor report links to.

Run from version4/ as:
  ../.venv/bin/python flock_simulator/scripts/compare_cone_vs_nocone.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))           # for flock_simulator
sys.path.insert(0, str(ROOT / "src"))   # for legacy vicsek_double_adaptive

from flock_simulator import FlockParams, FlockSimulator                 # noqa: E402
from vicsek_double_adaptive import (                                     # noqa: E402
    DoubleAdaptiveParams, DoubleAdaptiveVicsek,
)

DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


N_WARM = 500
N_MEAS = 1000
SEEDS = list(range(10))


def _run_new(seed: int) -> tuple[float, float, float]:
    p = FlockParams(N=2000, L=30.0, eta=0.10, seed=seed)
    sim = FlockSimulator(p)
    t0 = perf_counter()
    for _ in range(N_WARM):
        sim.step()
    phis = np.empty(N_MEAS)
    seps = np.empty(N_MEAS)
    for k in range(N_MEAS):
        sim.step()
        phis[k] = sim.polarisation()
        seps[k] = sim.density_separation_index()
    return float(phis.mean()), float(seps.mean()), perf_counter() - t0


def _run_legacy(seed: int) -> tuple[float, float, float]:
    p = DoubleAdaptiveParams(N=2000, L=30.0, eta=0.10, seed=seed)
    sim = DoubleAdaptiveVicsek(p)
    t0 = perf_counter()
    for _ in range(N_WARM):
        sim.step()
    phis = np.empty(N_MEAS)
    seps = np.empty(N_MEAS)
    for k in range(N_MEAS):
        sim.step()
        phis[k] = sim.polarisation()
        seps[k] = sim.density_separation_index()
    return float(phis.mean()), float(seps.mean()), perf_counter() - t0


def main() -> None:
    n = len(SEEDS)
    new_phi = np.empty(n)
    new_sep = np.empty(n)
    new_t = np.empty(n)
    leg_phi = np.empty(n)
    leg_sep = np.empty(n)
    leg_t = np.empty(n)

    for i, s in enumerate(SEEDS):
        leg_phi[i], leg_sep[i], leg_t[i] = _run_legacy(s)
        new_phi[i], new_sep[i], new_t[i] = _run_new(s)
        print(f"seed {s:>2d}  "
              f"legacy  phi={leg_phi[i]:.4f}  s_sep={leg_sep[i]:.3f}  "
              f"t={leg_t[i]:.1f}s   "
              f"|   nocone  phi={new_phi[i]:.4f}  s_sep={new_sep[i]:.3f}  "
              f"t={new_t[i]:.1f}s")

    print()
    print("Aggregated (10 seeds, L=30, eta=0.10, N=2000, "
          f"{N_WARM}+{N_MEAS} steps):")
    print(f"  legacy  : <phi>={leg_phi.mean():.4f}+-{leg_phi.std(ddof=1):.4f}  "
          f"<s_sep>={leg_sep.mean():.3f}+-{leg_sep.std(ddof=1):.3f}  "
          f"<t>={leg_t.mean():.1f}s")
    print(f"  no-cone : <phi>={new_phi.mean():.4f}+-{new_phi.std(ddof=1):.4f}  "
          f"<s_sep>={new_sep.mean():.3f}+-{new_sep.std(ddof=1):.3f}  "
          f"<t>={new_t.mean():.1f}s")
    dphi = new_phi.mean() - leg_phi.mean()
    dsep = new_sep.mean() - leg_sep.mean()
    print(f"  delta   : phi {dphi:+.4f}  s_sep {dsep:+.3f}  "
          f"speedup {leg_t.mean()/new_t.mean():.2f}x")

    np.savez_compressed(
        DATA / "cone_vs_nocone_validation.npz",
        seeds=np.array(SEEDS, dtype=np.int64),
        legacy_phi=leg_phi, legacy_sep=leg_sep, legacy_time_s=leg_t,
        nocone_phi=new_phi, nocone_sep=new_sep, nocone_time_s=new_t,
        params=np.array([N_WARM, N_MEAS, 2000, 30.0, 0.10]),
    )
    print(f"\nWrote {DATA / 'cone_vs_nocone_validation.npz'}")


if __name__ == "__main__":
    main()
