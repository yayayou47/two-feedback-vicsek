"""
Order-parameter histogram at $L = 64$ and $L = 90$, motility-only
ablation and double-adaptive model only, five seeds each. The
existing data at $L = 30$ shows unimodal $P(\\langle\\varphi
\\rangle)$ in both modes; this run tests whether a bimodal
structure emerges at larger $L$, which would diagnose a true
two-phase coexistence.

Output: data/double_orderpdf_largeL.npz
"""
from __future__ import annotations

from pathlib import Path
import time

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
    Ls = np.array([64.0, 90.0])
    sigma = 2.22
    n_warm = 2000
    n_meas = 6000
    seeds = list(range(11, 11 + 15, 3))   # 5 seeds

    cases = [
        ("motility", 0.005, 0.05, 1.0, 1.0, 0.100),
        ("full",     0.005, 0.05, 1.0, 2.0, 0.150),
    ]

    n_cases = len(cases)
    n_L = len(Ls)
    n_seeds = len(seeds)
    phi_traj = np.zeros((n_cases, n_L, n_seeds * n_meas))
    eta_arr = np.zeros(n_cases)

    pbar = tqdm(total=n_cases * n_L * n_seeds,
                desc="orderpdf_largeL")
    t0 = time.time()
    for ic, (label, vmn, vmx, amn, amx, eta) in enumerate(cases):
        for iL, L in enumerate(Ls):
            N = int(round(sigma * L * L))
            traj_list = []
            for seed in seeds:
                p = DoubleAdaptiveParams(
                    N=N, L=float(L),
                    v_max=float(vmx), v_min=float(vmn),
                    alpha_min=float(amn), alpha_max=float(amx),
                    R_r=0.5, R_a=0.7, beta=30.0,
                    eta=float(eta),
                    n_star=3.0, slope=2.0,
                    seed=int(seed),
                )
                traj_list.append(trace_phi(p, n_warm, n_meas))
                pbar.update(1)
            phi_traj[ic, iL] = np.concatenate(traj_list)
            eta_arr[ic] = eta
    pbar.close()

    np.savez_compressed(
        DATA / "double_orderpdf_largeL.npz",
        labels=np.array([c[0] for c in cases]),
        Ls=Ls,
        eta_per_case=eta_arr,
        phi_traj=phi_traj,
        seeds=np.array(seeds),
        params=np.array([sigma, n_warm, n_meas, n_seeds],
                         dtype=float),
    )

    from scipy.stats import skew, kurtosis
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
    for iL, L in enumerate(Ls):
        print(f"\n=== L = {int(L)} ===")
        for ic, (lbl, *_) in enumerate(cases):
            tr = phi_traj[ic, iL]
            U4 = (1.0 - (tr ** 4).mean()
                  / (3.0 * (tr ** 2).mean() ** 2))
            print(f"  {lbl:>10s}  mean={tr.mean():.3f}  "
                  f"std={tr.std():.3f}  "
                  f"skew={skew(tr):+.2f}  kurt={kurtosis(tr):+.2f}  "
                  f"U4={U4:.3f}")


if __name__ == "__main__":
    main()
