"""Matched-Gamma FSS control for the susceptibility synergy (referee B1).

The dense g(r) and C(tau) gaps reverse once the motility-only mode is
retuned so its alpha=1 kick decorrelates as fast as the full mode's
alpha=2 dense kick (Gamma = 1 - exp(-eta^alpha)). The three referees ask
the same of the susceptibility synergy Delta = a_full - a_motility: is the
faster chi_max growth of the full mode also just a lower-dense-noise
effect?

Unlike g(r)/C(tau) (measured at one fixed eta per mode), chi_max is a peak
over eta, so each mode is already compared at its own optimal noise. To
test the matched-Gamma question directly we re-run motility-only over a
Gamma-matched eta grid: for each grid value eta_g of the published scan we
set eta_mot = eta_g^2, so the motility (alpha=1) decorrelation
Gamma = 1 - exp(-eta_mot) equals the full-mode (alpha=2) dense
decorrelation 1 - exp(-eta_g^2) at eta_g. We then read chi_max(L) over this
matched grid and refit the slope, to compare against a_full and the
published a_motility.

Sizes L=15..128 (matching the referee request), ten seeds, same warm/measure
and fixed parameters as run_fss_parallel_nocone.py. Output keeps per-seed
chi so the slope CI bootstraps over seeds.

Usage:
  run_fss_matched_gamma_nocone.py [--workers 7]
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np

from _helpers import DATA, FlockParams, FlockSimulator

# Published scan eta grid; the matched motility grid is its square.
ETAS_REF = np.array([0.005, 0.010, 0.020, 0.035, 0.050,
                     0.075, 0.100, 0.150, 0.200, 0.300])
ETAS_MATCHED = ETAS_REF ** 2          # Gamma(alpha=1) matches Gamma(alpha=2) at eta_ref
SEEDS = list(range(11, 11 + 30, 3))   # 10 seeds
LS_DEFAULT = [15.0, 22.0, 30.0, 45.0, 64.0, 90.0, 128.0]
SIGMA = 2.22
N_WARM, N_MEAS = 1500, 1000
# motility-only (v3_limit): adaptive speed, fixed Cauchy alpha=1.
VMIN, VMAX, AMIN, AMAX = 0.005, 0.05, 1.0, 1.0


def _run_cell(task):
    iL, ie, isd, L, eta, seed, N = task
    p = FlockParams(N=int(N), L=float(L), v_max=VMAX, v_min=VMIN,
                    alpha_min=AMIN, alpha_max=AMAX, R_r=0.5, R_a=0.7,
                    eta=float(eta), n_star=3.0, slope=2.0, seed=int(seed))
    sim = FlockSimulator(p)
    for _ in range(N_WARM):
        sim.step()
    pser = np.empty(N_MEAS)
    for k in range(N_MEAS):
        sim.step()
        pser[k] = sim.polarisation()
    return (iL, ie, isd, float(N * pser.var()), float(pser.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--flush", type=int, default=40)
    ap.add_argument("--sizes", type=str, default=None,
                    help="comma-separated sizes; default is 15..128")
    ap.add_argument("--out", type=str,
                    default="double_fss_matched_gamma_nocone.npz")
    args = ap.parse_args()
    LS = ([float(s) for s in args.sizes.split(",")] if args.sizes
          else LS_DEFAULT)
    out = DATA / args.out
    nL, ne, ns = len(LS), len(ETAS_MATCHED), len(SEEDS)
    chi = np.full((nL, ne, ns), np.nan)
    phi = np.full((nL, ne, ns), np.nan)
    Ns = np.array([int(round(SIGMA * L * L)) for L in LS])

    if out.exists():
        prev = np.load(out, allow_pickle=True)
        if prev["chi"].shape == chi.shape:
            chi, phi = prev["chi"].copy(), prev["phi"].copy()
            print(f"resume: {int(np.isfinite(chi).sum())}/{chi.size} done")

    def save():
        np.savez_compressed(out, Ls=np.array(LS), Ns=Ns,
                            etas_ref=ETAS_REF, etas_matched=ETAS_MATCHED,
                            seeds=np.array(SEEDS), chi=chi, phi=phi,
                            params=np.array([SIGMA, 0.5, 0.7, N_WARM,
                                             N_MEAS, len(SEEDS)], float))

    tasks = []
    for iL, L in enumerate(LS):
        N = int(Ns[iL])
        for ie, eta in enumerate(ETAS_MATCHED):
            for isd, seed in enumerate(SEEDS):
                if np.isfinite(chi[iL, ie, isd]):
                    continue
                tasks.append((iL, ie, isd, L, float(eta), int(seed), N))
    total = len(tasks)
    print(f"matched-Gamma motility FSS: {total}/{chi.size} cells, "
          f"workers={args.workers}")
    if total == 0:
        print("nothing to do"); return

    t0, done = time.time(), 0
    with Pool(processes=args.workers) as pool:
        for (iL, ie, isd, c, ph) in pool.imap_unordered(_run_cell, tasks,
                                                        chunksize=1):
            chi[iL, ie, isd] = c
            phi[iL, ie, isd] = ph
            done += 1
            if done % args.flush == 0:
                save()
                el = (time.time() - t0) / 60.0
                print(f"  {done}/{total}  {el:.1f} min  "
                      f"ETA {(total-done)/max(done/max(el,1e-9),1e-9):.0f} min",
                      flush=True)
    save()
    print(f"runtime {(time.time()-t0)/60:.1f} min -> {out.name}")


if __name__ == "__main__":
    main()
