"""No-cone port of legacy run_double_finegrid.py.

Refines the small-eta region of the FSS at the four pilot sizes
{15, 22, 30, 45} plus L=64 (using L=64_nocone outputs externally).
This run targets only the four heavy-tailed modes at the four
pilot sizes: 4 modes x 4 sizes x 4 etas x 2 seeds = 128 jobs.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, measure_full, warm


def main() -> None:
    Ls = np.array([15.0, 22.0, 30.0, 45.0])
    sigma = 2.22
    etas = np.array([0.001, 0.002, 0.003, 0.0075])
    seeds = [11, 23]
    n_warm = 1500
    n_meas = 1000

    modes = [
        ("baseline", 0.05, 0.05, 1.0, 1.0),
        ("v2_limit", 0.05, 0.05, 1.0, 2.0),
        ("v3_limit", 0.005, 0.05, 1.0, 1.0),
        ("full",     0.005, 0.05, 1.0, 2.0),
    ]
    n_mode, n_L, n_eta = len(modes), len(Ls), len(etas)
    phi = np.zeros((n_mode, n_L, n_eta))
    chi = np.zeros_like(phi)
    U4 = np.zeros_like(phi)
    s_sep = np.zeros_like(phi)
    v_pop = np.zeros_like(phi)
    a_pop = np.zeros_like(phi)

    pbar = tqdm(total=n_mode * n_L * n_eta * len(seeds),
                desc="finegrid_nocone")
    t0 = time.time()
    for im, (label, vmn, vmx, amn, amx) in enumerate(modes):
        for iL, L in enumerate(Ls):
            N = int(round(sigma * L * L))
            for ie, eta in enumerate(etas):
                phi_acc, s_acc, v_acc, a_acc = [], [], [], []
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
                    p_arr, s_arr, v_arr, a_arr = measure_full(sim, n_meas)
                    phi_acc.append(p_arr)
                    s_acc.append(s_arr)
                    v_acc.append(v_arr)
                    a_acc.append(a_arr)
                    pbar.update(1)
                phi_all = np.concatenate(phi_acc)
                phi[im, iL, ie] = phi_all.mean()
                chi[im, iL, ie] = N * phi_all.var()
                U4[im, iL, ie] = (
                    1.0 - np.mean(phi_all ** 4)
                    / (3.0 * np.mean(phi_all ** 2) ** 2)
                )
                s_sep[im, iL, ie] = np.concatenate(s_acc).mean()
                v_pop[im, iL, ie] = np.concatenate(v_acc).mean()
                a_pop[im, iL, ie] = np.concatenate(a_acc).mean()
    pbar.close()

    out = DATA / "double_finegrid_nocone.npz"
    np.savez_compressed(
        out,
        modes=np.array([m[0] for m in modes]),
        Ls=Ls, etas=etas,
        phi=phi, chi=chi, U4=U4, s_sep=s_sep,
        v_pop=v_pop, a_pop=a_pop,
        params=np.array([sigma, n_warm, n_meas, len(seeds)]),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")


if __name__ == "__main__":
    main()
