"""No-cone port of legacy run_double_orderpdf_largeL.py.

Order-parameter histograms for motility-only and full at L=64
and L=90, 5 seeds, 6000 measurement steps per seed.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, measure_phi, warm


def main() -> None:
    Ls = np.array([64.0, 90.0])
    sigma = 2.22
    n_warm, n_meas = 2000, 6000
    seeds = list(range(11, 11 + 15, 3))   # 5 seeds

    cases = [
        ("motility", 0.005, 0.05, 1.0, 1.0, 0.100),
        ("full",     0.005, 0.05, 1.0, 2.0, 0.150),
    ]
    n_cases, n_L, n_seeds = len(cases), len(Ls), len(seeds)
    phi_traj = np.zeros((n_cases, n_L, n_seeds * n_meas))
    eta_arr = np.zeros(n_cases)

    pbar = tqdm(total=n_cases * n_L * n_seeds, desc="orderpdf_largeL_nocone")
    t0 = time.time()
    for ic, (label, vmn, vmx, amn, amx, eta) in enumerate(cases):
        for iL, L in enumerate(Ls):
            N = int(round(sigma * L * L))
            traj_list = []
            for seed in seeds:
                p = FlockParams(
                    N=N, L=float(L),
                    v_max=float(vmx), v_min=float(vmn),
                    alpha_min=float(amn), alpha_max=float(amx),
                    R_r=0.5, R_a=0.7, eta=float(eta),
                    n_star=3.0, slope=2.0, seed=int(seed),
                )
                sim = FlockSimulator(p)
                warm(sim, n_warm)
                traj_list.append(measure_phi(sim, n_meas))
                pbar.update(1)
            phi_traj[ic, iL] = np.concatenate(traj_list)
            eta_arr[ic] = eta
    pbar.close()
    out = DATA / "double_orderpdf_largeL_nocone.npz"
    np.savez_compressed(out,
                        labels=np.array([c[0] for c in cases]),
                        Ls=Ls,
                        eta_per_case=eta_arr,
                        phi_traj=phi_traj,
                        seeds=np.array(seeds),
                        params=np.array([sigma, n_warm, n_meas, n_seeds],
                                         dtype=float))
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")


if __name__ == "__main__":
    main()
