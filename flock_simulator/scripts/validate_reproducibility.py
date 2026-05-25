"""
Reproducibility validator for the omnidirectional simulator.

Runs the canonical "validation grid" (L=30, eta=0.10, N=2000,
1000 measurement steps after 500 warm-up, 10 seeds) and either

  - writes a reference snapshot
    (``data/no_cone_reference.npz``) when invoked with ``--save``;
  - compares against the existing reference and exits non-zero on
    any divergence beyond ``--rtol`` (default 1e-12) when invoked
    without ``--save``.

The reference captures, per seed: the final ``polarisation()`` and
``density_separation_index(n_bins=10)``. With ``SeedSequence`` and
PCG64 those values must be bit-for-bit reproducible on the same
architecture; the validator therefore enforces a strict tolerance
by default.

Run from version4/ as:
  ../.venv/bin/python flock_simulator/scripts/validate_reproducibility.py --save
  ../.venv/bin/python flock_simulator/scripts/validate_reproducibility.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flock_simulator import FlockParams, FlockSimulator   # noqa: E402

DATA = ROOT / "data"
REF_PATH = DATA / "no_cone_reference.npz"


def _run_one(seed: int, *, n_warm: int, n_meas: int) -> tuple[float, float]:
    """Run one validation simulation and return (mean phi, mean s_sep)."""
    p = FlockParams(N=2000, L=30.0, eta=0.10, seed=seed)
    sim = FlockSimulator(p)
    for _ in range(n_warm):
        sim.step()
    phis = np.empty(n_meas)
    seps = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        phis[k] = sim.polarisation()
        seps[k] = sim.density_separation_index()
    return float(phis.mean()), float(seps.mean())


def _grid(n_warm: int, n_meas: int, n_seeds: int) -> dict[str, np.ndarray]:
    seeds = np.arange(n_seeds, dtype=np.int64)
    phi = np.empty(n_seeds)
    sep = np.empty(n_seeds)
    for i, s in enumerate(seeds):
        phi[i], sep[i] = _run_one(int(s), n_warm=n_warm, n_meas=n_meas)
    return dict(seeds=seeds, phi=phi, sep=sep,
                params=np.array([n_warm, n_meas, n_seeds],
                                 dtype=np.int64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="write the reference snapshot")
    parser.add_argument("--n-warm", type=int, default=500)
    parser.add_argument("--n-meas", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=1e-12)
    args = parser.parse_args()

    grid = _grid(args.n_warm, args.n_meas, args.n_seeds)
    print("validation grid:")
    print(f"  seeds: {grid['seeds'].tolist()}")
    print(f"  phi  : mean={grid['phi'].mean():+.6f}  "
          f"std={grid['phi'].std(ddof=1):.6f}")
    print(f"  s_sep: mean={grid['sep'].mean():+.6f}  "
          f"std={grid['sep'].std(ddof=1):.6f}")

    if args.save:
        DATA.mkdir(exist_ok=True)
        np.savez_compressed(REF_PATH, **grid)
        print(f"\nReference written to {REF_PATH}")
        return

    if not REF_PATH.exists():
        print(f"\nNo reference at {REF_PATH}; run with --save first.",
              file=sys.stderr)
        sys.exit(2)

    ref = np.load(REF_PATH)
    bad = []
    for k in ("phi", "sep"):
        max_abs_diff = float(np.max(np.abs(grid[k] - ref[k])))
        ref_max = float(np.max(np.abs(ref[k])))
        rel = max_abs_diff / max(ref_max, 1e-30)
        ok = rel <= args.rtol
        marker = "OK " if ok else "FAIL"
        print(f"  {marker}  {k:<5s} max|diff|={max_abs_diff:.3e}  "
              f"rel={rel:.3e}  rtol={args.rtol:.3e}")
        if not ok:
            bad.append(k)

    if bad:
        print(f"\nReproducibility broken on: {bad}", file=sys.stderr)
        sys.exit(1)
    print("\nAll observables match the saved reference.")


if __name__ == "__main__":
    main()
