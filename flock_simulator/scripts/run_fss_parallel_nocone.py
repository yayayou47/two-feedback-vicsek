"""Parallel finite-size scan for the revision runs A1 and D1.

The serial ``run_fss_homog10_nocone.py`` produced the published
seven-size homogeneous ten-seed series at rho0 = 2.22. The two
referee-driven extensions are too heavy to run serially:

  A1  extend the size grid to L = 180, 256 (rho0 = 2.22) to test
      whether the chi_max(L) excess slope survives larger boxes;
  D1  repeat chi_max(L) at rho0 in {1.0, 3.0} to separate a robust
      MIPS split from a parameter-window artefact.

Each (mode, eta, seed, L) run is fully determined by
``SeedSequence(seed)`` and is independent of every other, so the
sweep parallelises across processes with bit-for-bit the same
numbers as the serial driver. Work is distributed cell by cell with
a multiprocessing pool; the output array is written after every
flush of completed cells, and a rerun skips cells already finite, so
a multi-hour run survives interruptions (including a dead battery)
and resumes exactly where it stopped.

Usage:
  run_fss_parallel_nocone.py --rho 2.22 --sizes 180,256 \
      --out double_fss_homog10_largeL_nocone.npz [--workers 6]
  run_fss_parallel_nocone.py --rho 1.0  --sizes 22,30,45,64,90,128 \
      --out double_fss_density_rho1p0_nocone.npz
  run_fss_parallel_nocone.py --rho 3.0  --sizes 22,30,45,64,90,128 \
      --out double_fss_density_rho3p0_nocone.npz

Protocol mirrors run_fss_homog10_nocone.py exactly (same 10-eta
grid, same ten seeds, warm 1500 + measure 1000, R_r = 0.5,
R_a = 0.7, n_star = 3.0, slope = 2.0); only L, rho0 (= SIGMA) and
the output path change.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np

from _helpers import DATA, FlockParams, FlockSimulator

ETAS = np.array([0.005, 0.010, 0.020, 0.035, 0.050,
                 0.075, 0.100, 0.150, 0.200, 0.300])
SEEDS = list(range(11, 11 + 30, 3))          # 10 seeds: 11,14,...,38
N_WARM = 1500
N_MEAS = 1000

MODES = [
    ("vicsek_gauss", 0.05, 0.05, 2.0, 2.0),
    ("baseline",     0.05, 0.05, 1.0, 1.0),
    ("v2_limit",     0.05, 0.05, 1.0, 2.0),
    ("v3_limit",     0.005, 0.05, 1.0, 1.0),
    ("full",         0.005, 0.05, 1.0, 2.0),
]


def _u4(phi_series: np.ndarray) -> float:
    m2 = np.mean(phi_series ** 2)
    m4 = np.mean(phi_series ** 4)
    return float(1.0 - m4 / (3.0 * m2 ** 2)) if m2 > 0 else 0.0


def _run_cell(task):
    """Run one (mode, eta, seed, L) cell. Top-level so it pickles."""
    (im, iL, ie, isd, name, vmn, vmx, amn, amx,
     L, eta, seed, N) = task
    p = FlockParams(
        N=int(N), L=float(L),
        v_max=float(vmx), v_min=float(vmn),
        alpha_min=float(amn), alpha_max=float(amx),
        R_r=0.5, R_a=0.7, eta=float(eta),
        n_star=3.0, slope=2.0, seed=int(seed),
    )
    sim = FlockSimulator(p)
    for _ in range(N_WARM):
        sim.step()
    pser = np.empty(N_MEAS)
    sser = np.empty(N_MEAS)
    for k in range(N_MEAS):
        sim.step()
        pser[k] = sim.polarisation()
        sser[k] = sim.density_separation_index(n_bins=10)
    return (im, iL, ie, isd,
            float(N * pser.var()), float(pser.mean()),
            _u4(pser), float(sser.mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, required=True,
                    help="number density rho0 (= SIGMA)")
    ap.add_argument("--sizes", type=str, required=True,
                    help="comma-separated box sizes, e.g. 180,256")
    ap.add_argument("--out", type=str, required=True,
                    help="output .npz filename (under data/)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--flush", type=int, default=40,
                    help="save after this many freshly computed cells")
    args = ap.parse_args()

    sigma = float(args.rho)
    LS = [float(s) for s in args.sizes.split(",")]
    out = DATA / args.out
    n_mode, n_L, n_eta, n_seed = len(MODES), len(LS), len(ETAS), len(SEEDS)
    shape = (n_mode, n_L, n_eta, n_seed)
    chi = np.full(shape, np.nan)
    phi = np.full(shape, np.nan)
    u4 = np.full(shape, np.nan)
    ssep = np.full(shape, np.nan)
    Ns = np.array([int(round(sigma * L * L)) for L in LS])
    mode_names = [m[0] for m in MODES]

    # Cell-level resume: reload any finished cells.
    if out.exists():
        prev = np.load(out, allow_pickle=True)
        if prev["chi"].shape == shape:
            chi, phi, u4, ssep = (prev["chi"].copy(), prev["phi"].copy(),
                                  prev["u4"].copy(), prev["ssep"].copy())
            print(f"resuming: {int(np.isfinite(chi).sum())}/{chi.size} "
                  f"cells already done")

    def save():
        np.savez_compressed(
            out, Ls=np.array(LS), Ns=Ns,
            modes=np.array(mode_names), etas=ETAS,
            seeds=np.array(SEEDS), chi=chi, phi=phi, u4=u4, ssep=ssep,
            params=np.array([sigma, 0.5, 0.7, N_WARM, N_MEAS,
                             len(SEEDS)], dtype=float),
        )

    # Build the task list of not-yet-finished cells.
    tasks = []
    for iL, L in enumerate(LS):
        N = int(Ns[iL])
        for im, (name, vmn, vmx, amn, amx) in enumerate(MODES):
            for ie, eta in enumerate(ETAS):
                for isd, seed in enumerate(SEEDS):
                    if np.isfinite(chi[im, iL, ie, isd]):
                        continue
                    tasks.append((im, iL, ie, isd, name, vmn, vmx,
                                  amn, amx, L, float(eta), int(seed), N))
    total = len(tasks)
    print(f"rho0={sigma}  sizes={[int(x) for x in LS]}  "
          f"workers={args.workers}  cells to run: {total}/{chi.size}")
    if total == 0:
        print("nothing to do")
        return

    t0 = time.time()
    done = 0
    with Pool(processes=args.workers) as pool:
        for (im, iL, ie, isd, c, ph, u, s) in pool.imap_unordered(
                _run_cell, tasks, chunksize=1):
            chi[im, iL, ie, isd] = c
            phi[im, iL, ie, isd] = ph
            u4[im, iL, ie, isd] = u
            ssep[im, iL, ie, isd] = s
            done += 1
            if done % args.flush == 0:
                save()
                el = (time.time() - t0) / 60.0
                rate = done / max(el, 1e-9)
                eta_min = (total - done) / max(rate, 1e-9)
                print(f"  {done}/{total} cells  {el:.1f} min  "
                      f"ETA {eta_min:.0f} min", flush=True)
    save()
    print(f"\nruntime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")


if __name__ == "__main__":
    main()
