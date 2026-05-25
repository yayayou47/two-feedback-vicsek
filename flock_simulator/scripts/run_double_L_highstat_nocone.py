"""No-cone port of legacy run_double_L_highstat.py.

10-seed high-statistics gap test for s_sep on L in {50, 64, 80}
at the full and motility-only near-critical etas. Refines the
3-seed Lfine result with per-seed standard error.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, measure_phi_sep, warm


def main() -> None:
    sigma = 2.22
    Ls = np.array([50.0, 64.0, 80.0])
    etas = np.array([0.075, 0.100, 0.150, 0.200])
    seeds = list(range(11, 11 + 30, 3))   # 10 seeds
    n_warm, n_meas = 2000, 2000
    modes = [
        ("motility", 0.005, 0.05, 1.0, 1.0),
        ("full",     0.005, 0.05, 1.0, 2.0),
    ]
    n_mode, n_L, n_eta, n_seeds = (len(modes), len(Ls), len(etas), len(seeds))
    chi = np.zeros((n_mode, n_L, n_eta))
    s_sep = np.zeros((n_mode, n_L, n_eta))
    phi = np.zeros((n_mode, n_L, n_eta))
    s_sep_per_seed = np.zeros((n_mode, n_L, n_eta, n_seeds))
    phi_per_seed = np.zeros((n_mode, n_L, n_eta, n_seeds))

    pbar = tqdm(total=n_mode * n_L * n_eta * n_seeds, desc="L_highstat_nocone")
    t0 = time.time()
    for im, (label, vmn, vmx, amn, amx) in enumerate(modes):
        for iL, L in enumerate(Ls):
            N = int(round(sigma * L * L))
            for ie, eta in enumerate(etas):
                phi_acc, s_acc = [], []
                for isd, seed in enumerate(seeds):
                    p = FlockParams(
                        N=N, L=float(L),
                        v_max=float(vmx), v_min=float(vmn),
                        alpha_min=float(amn), alpha_max=float(amx),
                        R_r=0.5, R_a=0.7, eta=float(eta),
                        n_star=3.0, slope=2.0, seed=int(seed),
                    )
                    sim = FlockSimulator(p)
                    warm(sim, n_warm)
                    p_arr, s_arr = measure_phi_sep(sim, n_meas)
                    phi_acc.append(p_arr)
                    s_acc.append(s_arr)
                    phi_per_seed[im, iL, ie, isd] = p_arr.mean()
                    s_sep_per_seed[im, iL, ie, isd] = s_arr.mean()
                    pbar.update(1)
                phi_all = np.concatenate(phi_acc)
                phi[im, iL, ie] = phi_all.mean()
                chi[im, iL, ie] = N * phi_all.var()
                s_sep[im, iL, ie] = np.concatenate(s_acc).mean()
    pbar.close()
    out = DATA / "double_L_highstat_nocone.npz"
    np.savez_compressed(out,
                        modes=np.array([m[0] for m in modes]),
                        Ls=Ls, etas=etas, seeds=np.array(seeds),
                        phi=phi, chi=chi, s_sep=s_sep,
                        phi_per_seed=phi_per_seed,
                        s_sep_per_seed=s_sep_per_seed,
                        params=np.array([sigma, n_warm, n_meas, n_seeds]))
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")


if __name__ == "__main__":
    main()
