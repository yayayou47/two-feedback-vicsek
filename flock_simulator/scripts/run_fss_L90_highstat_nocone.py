"""L = 90 high-statistics rerun of the two motility-active modes.

The synergy diagnostic Delta_6 drops from +1.45 (n=5) to +0.46
(n=6) because the full-mode chi_max at L = 90 carries the largest
seed-level fluctuation of the seven-size series, fitted on only
three seeds. Here we re-run motility-only and double-adaptive at
L = 90 with ten seeds and store the per-seed chi_max so the L = 90
node -- and hence Delta_6 -- rests on the same statistics as the
microscopic diagnostics. The flat reference modes (Vicsek, Cauchy,
noise-shape) are left at their three-seed values; their slopes are
zero within CI and their seed dispersion is small.

Output: data/double_L90_highstat_nocone.npz with per-seed and
seed-averaged chi_max at L = 90 for the two modes.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator

ETAS = np.array([0.005, 0.010, 0.020, 0.035, 0.050,
                 0.075, 0.100, 0.150, 0.200, 0.300])
SEEDS = list(range(11, 11 + 30, 3))    # 10 seeds, same scheme as hs runs
N_WARM = 1500
N_MEAS = 1000
SIGMA = 2.22

MODES = [
    ("v3_limit", 0.005, 0.05, 1.0, 1.0),
    ("full",     0.005, 0.05, 1.0, 2.0),
]


def main() -> None:
    L = 90.0
    N = int(round(SIGMA * L * L))
    n_mode, n_eta, n_seeds = len(MODES), len(ETAS), len(SEEDS)
    # per-seed chi at every eta, so chi_max can be taken per seed
    chi_seed = np.zeros((n_mode, n_eta, n_seeds))
    phi_seed = np.zeros((n_mode, n_eta, n_seeds))

    pbar = tqdm(total=n_mode * n_eta * n_seeds, desc="L90_hs_nocone")
    t0 = time.time()
    for im, (name, vmn, vmx, amn, amx) in enumerate(MODES):
        for ie, eta in enumerate(ETAS):
            for isd, seed in enumerate(SEEDS):
                p = FlockParams(
                    N=N, L=float(L),
                    v_max=float(vmx), v_min=float(vmn),
                    alpha_min=float(amn), alpha_max=float(amx),
                    R_r=0.5, R_a=0.7, eta=float(eta),
                    n_star=3.0, slope=2.0, seed=int(seed),
                )
                sim = FlockSimulator(p)
                sim.state.theta[:] = 0.0
                for _ in range(N_WARM):
                    sim.step()
                acc = np.empty(N_MEAS)
                for k in range(N_MEAS):
                    sim.step()
                    acc[k] = sim.polarisation()
                phi_seed[im, ie, isd] = acc.mean()
                chi_seed[im, ie, isd] = N * acc.var()
                pbar.update(1)
    pbar.close()

    # chi_max per seed (max over eta), then seed mean/SE
    chi_max_seed = chi_seed.max(axis=1)            # (mode, seed)
    chi_max_mean = chi_max_seed.mean(axis=1)
    chi_max_se = chi_max_seed.std(axis=1, ddof=1) / np.sqrt(n_seeds)
    # seed-pooled chi (mean over seeds first, then max over eta) to
    # match the pooled estimator used in the 3-seed table
    chi_pooled = chi_seed.mean(axis=2)             # (mode, eta)
    chi_max_pooled = chi_pooled.max(axis=1)

    out = DATA / "double_L90_highstat_nocone.npz"
    np.savez_compressed(
        out,
        modes=np.array([m[0] for m in MODES]),
        L=L, etas=ETAS, seeds=np.array(SEEDS),
        chi_seed=chi_seed, phi_seed=phi_seed,
        chi_max_seed=chi_max_seed,
        chi_max_mean=chi_max_mean, chi_max_se=chi_max_se,
        chi_max_pooled=chi_max_pooled,
        params=np.array([SIGMA, N_WARM, N_MEAS, n_seeds]),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")
    print(f"\n{'mode':>10s}  {'chi_max(pooled)':>16s}  "
          f"{'chi_max(seed mean+-SE)':>24s}")
    for im, m in enumerate(MODES):
        print(f"  {m[0]:>10s}  {chi_max_pooled[im]:14.1f}    "
              f"{chi_max_mean[im]:10.1f}+-{chi_max_se[im]:.1f}")


if __name__ == "__main__":
    main()
