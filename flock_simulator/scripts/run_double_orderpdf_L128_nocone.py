"""No-cone port of legacy run_double_orderpdf_L128.py.

Long-trajectory P(phi) at L=128: 4 modes, 5 seeds, 60000 step
measurement per seed (the 24-h run that resolved the L=90
shoulder in the legacy paper).

Output: data/double_orderpdf_L128_nocone.npz.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, measure_phi, warm


def main() -> None:
    L = 128.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm = 2000
    n_meas = 60000
    seeds = list(range(11, 11 + 15, 3))   # 5 seeds

    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0, 0.035),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0, 0.075),
        ("motility", 0.005, 0.05,  1.0, 1.0, 0.100),
        ("full",     0.005, 0.05,  1.0, 2.0, 0.150),
    ]
    n_cases, n_seeds = len(cases), len(seeds)
    phi_traj = np.zeros((n_cases, n_seeds * n_meas))
    eta_arr = np.zeros(n_cases)

    pbar = tqdm(total=n_cases * n_seeds, desc="orderpdf_L128_nocone")
    t0 = time.time()
    for ic, (label, vmn, vmx, amn, amx, eta) in enumerate(cases):
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
        phi_traj[ic] = np.concatenate(traj_list)
        eta_arr[ic] = eta
    pbar.close()

    out = DATA / "double_orderpdf_L128_nocone.npz"
    np.savez_compressed(
        out,
        labels=np.array([c[0] for c in cases]),
        L=L,
        eta_per_case=eta_arr,
        phi_traj=phi_traj,
        seeds=np.array(seeds),
        params=np.array([sigma, N, n_warm, n_meas, n_seeds], dtype=float),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")


if __name__ == "__main__":
    main()
