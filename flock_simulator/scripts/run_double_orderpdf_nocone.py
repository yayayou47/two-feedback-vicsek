"""No-cone port of legacy run_double_orderpdf.py.

Order-parameter histogram at L=30, near-critical eta, for v3-limit
and full modes. Output: double_orderpdf_nocone.npz.
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, measure_phi, warm


def main() -> None:
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm, n_meas = 2000, 6000
    seeds = [11, 23, 41]

    cases = [
        ("v3_limit", 0.005, 0.05, 1.0, 1.0, 0.100),
        ("full",     0.005, 0.05, 1.0, 2.0, 0.150),
    ]
    phi_traj = np.zeros((len(cases), len(seeds) * n_meas))
    labels = np.array([c[0] for c in cases])
    eta_arr = np.zeros(len(cases))

    pbar = tqdm(total=len(cases) * len(seeds), desc="orderpdf_nocone")
    for ic, (label, vmn, vmx, amn, amx, eta) in enumerate(cases):
        traj_list = []
        for seed in seeds:
            p = FlockParams(
                N=N, L=L,
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

    out = DATA / "double_orderpdf_nocone.npz"
    np.savez_compressed(
        out,
        labels=labels, phi_traj=phi_traj, eta=eta_arr,
        params=np.array([N, L, n_warm, n_meas, len(seeds)], dtype=float),
    )
    from scipy.stats import skew, kurtosis
    print()
    for ic, lbl in enumerate(labels):
        tr = phi_traj[ic]
        U4 = 1 - (tr ** 4).mean() / (3 * (tr ** 2).mean() ** 2)
        print(f"  {lbl}: eta={eta_arr[ic]:.3f}  mean={tr.mean():.3f}  "
              f"std={tr.std():.3f}  skew={skew(tr):+.2f}  "
              f"kurt={kurtosis(tr):+.2f}  U4={U4:.3f}")
    print(f"saved: {out.name}")


if __name__ == "__main__":
    main()
