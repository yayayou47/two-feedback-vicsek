"""No-cone port of legacy run_vicsek_gauss_ref.py.

Original Vicsek (Gaussian noise, fixed speed) reference at the
five pilot sizes. ``alpha_min = alpha_max = 2`` collapses the
noise-shape channel to a pure Gaussian; ``v_min = v_max = 0.05``
freezes the motility channel. Output: vicsek_gauss_ref_nocone.npz.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, measure_phi_sep, warm


def main() -> None:
    Ls = np.array([15.0, 22.0, 30.0, 45.0, 64.0])
    sigma = 2.22
    etas = np.array([0.005, 0.010, 0.020, 0.035, 0.050,
                     0.075, 0.100, 0.150, 0.200, 0.300])
    seeds = [11, 23, 41]
    n_warm, n_meas = 1500, 1000
    n_L, n_eta = len(Ls), len(etas)
    phi = np.zeros((n_L, n_eta))
    chi = np.zeros_like(phi)
    U4 = np.zeros_like(phi)
    s_sep = np.zeros_like(phi)

    pbar = tqdm(total=n_L * n_eta * len(seeds),
                desc="vicsek_gauss_nocone")
    t0 = time.time()
    for iL, L in enumerate(Ls):
        N = int(round(sigma * L * L))
        for ie, eta in enumerate(etas):
            phi_acc, s_acc = [], []
            for seed in seeds:
                p = FlockParams(
                    N=N, L=float(L),
                    v_max=0.05, v_min=0.05,
                    alpha_min=2.0, alpha_max=2.0,
                    R_r=0.5, R_a=0.7, eta=float(eta),
                    n_star=3.0, slope=2.0, seed=int(seed),
                )
                sim = FlockSimulator(p)
                warm(sim, n_warm)
                p_arr, s_arr = measure_phi_sep(sim, n_meas)
                phi_acc.append(p_arr)
                s_acc.append(s_arr)
                pbar.update(1)
            phi_all = np.concatenate(phi_acc)
            phi[iL, ie] = phi_all.mean()
            chi[iL, ie] = N * phi_all.var()
            U4[iL, ie] = (
                1.0 - np.mean(phi_all ** 4)
                / (3.0 * np.mean(phi_all ** 2) ** 2)
            )
            s_sep[iL, ie] = np.concatenate(s_acc).mean()
    pbar.close()
    out = DATA / "vicsek_gauss_ref_nocone.npz"
    np.savez_compressed(out, Ls=Ls, etas=etas,
                        phi=phi, chi=chi, U4=U4, s_sep=s_sep,
                        params=np.array([sigma, n_warm, n_meas, len(seeds)]))
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")
    chi_max_arr = np.array([chi[iL].max() for iL in range(n_L)])
    a, _ = np.polyfit(np.log(Ls), np.log(chi_max_arr), 1)
    print(f"  Vicsek-Gauss FSS slope (5 sizes): {a:+.3f}")


if __name__ == "__main__":
    main()
