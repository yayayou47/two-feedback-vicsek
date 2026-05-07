"""
Probability density of the polar order parameter for the FULL
two-feedback model and the v3 limit, at L = 30, near-critical
eta. Bimodal -> first-order; unimodal -> continuous. The open
question for v4 is whether the synergic regime produces the
two-peak structure that v3 alone could not.

Output: data/double_orderpdf.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from vicsek_double_adaptive import (DoubleAdaptiveParams,
                                     DoubleAdaptiveVicsek)


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def trace_phi(p, n_warm, n_meas):
    sim = DoubleAdaptiveVicsek(p)
    sim.theta[:] = 0.0
    for _ in range(n_warm):
        sim.step()
    out = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        out[k] = sim.polarisation()
    return out


def main():
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm = 2000
    n_meas = 6000
    seeds = [11, 23, 41]

    cases = [
        ("v3_limit", 0.005, 0.05, 1.0, 1.0, 0.100),
        ("full",     0.005, 0.05, 1.0, 2.0, 0.150),
    ]

    n_cases = len(cases)
    phi_traj = np.zeros((n_cases, len(seeds) * n_meas))
    labels = np.array([c[0] for c in cases])
    eta_arr = np.zeros(n_cases)

    pbar = tqdm(total=n_cases * len(seeds), desc="orderpdf")
    for ic, (label, vmn, vmx, amn, amx, eta) in enumerate(cases):
        traj_list = []
        for seed in seeds:
            p = DoubleAdaptiveParams(
                N=N, L=L,
                v_max=float(vmx), v_min=float(vmn),
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=0.5, R_a=0.7, beta=30.0,
                eta=float(eta),
                n_star=3.0, slope=2.0,
                seed=int(seed),
            )
            traj_list.append(trace_phi(p, n_warm, n_meas))
            pbar.update(1)
        phi_traj[ic] = np.concatenate(traj_list)
        eta_arr[ic] = eta
    pbar.close()

    np.savez_compressed(
        DATA / "double_orderpdf.npz",
        labels=labels, phi_traj=phi_traj, eta=eta_arr,
        params=np.array([N, L, n_warm, n_meas, len(seeds)],
                         dtype=float),
    )

    from scipy.stats import skew, kurtosis
    print()
    for ic, lbl in enumerate(labels):
        tr = phi_traj[ic]
        U4 = 1 - (tr ** 4).mean() / (3 * (tr ** 2).mean() ** 2)
        print(f"  {lbl}: eta={eta_arr[ic]:.3f}  mean={tr.mean():.3f}  "
              f"std={tr.std():.3f}  skew={skew(tr):+.2f}  "
              f"kurt={kurtosis(tr):+.2f}  U4={U4:.3f}")
    print("saved:", DATA / "double_orderpdf.npz")


if __name__ == "__main__":
    main()
