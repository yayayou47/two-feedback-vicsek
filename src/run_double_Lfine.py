"""
Fine-L scan comparing the full double-adaptive model against the
motility-only ablation across the intermediate sizes
L in {38, 50, 60, 75, 105}, filling the gaps in the main FSS series.
Over the eta sub-grid {0.075, 0.100, 0.150, 0.200} with 5 seeds it
records phi, chi and the density-separation index s_sep for both
modes.

Output: data/double_Lfine.npz
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


def measure(sim, n_meas):
    phi = np.empty(n_meas)
    s_sep = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        phi[k] = sim.polarisation()
        s_sep[k] = sim.density_separation_index(n_bins=10)
    return phi, s_sep


def main():
    sigma = 2.22
    Ls = np.array([38.0, 50.0, 60.0, 75.0, 105.0])
    etas = np.array([0.075, 0.100, 0.150, 0.200])
    seeds = [11, 23, 41, 59, 73]
    n_warm = 1500
    n_meas = 1500

    modes = [
        ("motility", 0.005, 0.05, 1.0, 1.0),
        ("full",     0.005, 0.05, 1.0, 2.0),
    ]

    n_mode, n_L, n_eta = len(modes), len(Ls), len(etas)
    chi = np.zeros((n_mode, n_L, n_eta))
    s_sep = np.zeros((n_mode, n_L, n_eta))
    phi = np.zeros((n_mode, n_L, n_eta))

    pbar = tqdm(total=n_mode * n_L * n_eta * len(seeds),
                desc="Lfine")
    t0 = time.time()
    for im, (label, vmn, vmx, amn, amx) in enumerate(modes):
        for iL, L in enumerate(Ls):
            N = int(round(sigma * L * L))
            for ie, eta in enumerate(etas):
                phi_acc, s_acc = [], []
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
                    sim = DoubleAdaptiveVicsek(p)
                    sim.theta[:] = 0.0
                    for _ in range(n_warm):
                        sim.step()
                    p_arr, s_arr = measure(sim, n_meas)
                    phi_acc.append(p_arr)
                    s_acc.append(s_arr)
                    pbar.update(1)
                phi_all = np.concatenate(phi_acc)
                phi[im, iL, ie] = phi_all.mean()
                chi[im, iL, ie] = N * phi_all.var()
                s_sep[im, iL, ie] = np.concatenate(s_acc).mean()
    pbar.close()

    np.savez_compressed(
        DATA / "double_Lfine.npz",
        modes=np.array([m[0] for m in modes]),
        Ls=Ls, etas=etas,
        phi=phi, chi=chi, s_sep=s_sep,
        params=np.array([sigma, n_warm, n_meas, len(seeds)]),
    )

    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
    for iL, L in enumerate(Ls):
        s_full = s_sep[1, iL].max()
        s_mot = s_sep[0, iL].max()
        c_full = chi[1, iL].max()
        c_mot = chi[0, iL].max()
        gap = s_full - s_mot
        print(f"  L={int(L):3d}  s_sep: full={s_full:.3f}  "
              f"motility={s_mot:.3f}  gap={gap:+.3f}    "
              f"chi: full={c_full:6.2f}  motility={c_mot:6.2f}")


if __name__ == "__main__":
    main()
