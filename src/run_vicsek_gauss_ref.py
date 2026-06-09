"""
Canonical Vicsek reference run (Gaussian noise, fixed speed:
alpha=2, v_min=v_max=0.05) over the same eta grid and five sizes
$L = 15, 22, 30, 45, 64$ ($\\sigma = N/L^2 = 2.22$, seeds
11/23/41). Sweeps eta and records phi, the susceptibility chi,
the Binder cumulant U4, and the density-separation index s_sep
per size, then fits and prints the FSS slope of chi_max.

Output: data/vicsek_gauss_ref.npz (Ls, etas, phi, chi, U4, s_sep).
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


def measure(sim, n_meas):
    phi = np.empty(n_meas)
    s_sep = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        phi[k] = sim.polarisation()
        s_sep[k] = sim.density_separation_index(n_bins=10)
    return phi, s_sep


def main():
    Ls = np.array([15.0, 22.0, 30.0, 45.0, 64.0])
    sigma = 2.22
    etas = np.array([0.005, 0.010, 0.020, 0.035, 0.050,
                     0.075, 0.100, 0.150, 0.200, 0.300])
    seeds = [11, 23, 41]
    n_warm = 1500
    n_meas = 1000

    n_L, n_eta = len(Ls), len(etas)
    phi = np.zeros((n_L, n_eta))
    chi = np.zeros((n_L, n_eta))
    U4 = np.zeros((n_L, n_eta))
    s_sep = np.zeros((n_L, n_eta))

    pbar = tqdm(total=n_L * n_eta * len(seeds), desc="vicsek_gauss")
    t0 = time.time()
    for iL, L in enumerate(Ls):
        N = int(round(sigma * L * L))
        for ie, eta in enumerate(etas):
            phi_acc, s_acc = [], []
            for seed in seeds:
                p = DoubleAdaptiveParams(
                    N=N, L=float(L),
                    v_max=0.05, v_min=0.05,
                    alpha_min=2.0, alpha_max=2.0,    # Gaussian
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
            phi[iL, ie] = phi_all.mean()
            chi[iL, ie] = N * phi_all.var()
            U4[iL, ie] = (
                1.0 - np.mean(phi_all ** 4)
                / (3.0 * np.mean(phi_all ** 2) ** 2)
            )
            s_sep[iL, ie] = np.concatenate(s_acc).mean()
    pbar.close()

    np.savez_compressed(
        DATA / "vicsek_gauss_ref.npz",
        Ls=Ls, etas=etas,
        phi=phi, chi=chi, U4=U4, s_sep=s_sep,
        params=np.array([sigma, n_warm, n_meas, len(seeds)]),
    )

    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
    chi_max_arr = np.zeros(n_L)
    for iL, L in enumerate(Ls):
        ie_max = int(np.argmax(chi[iL]))
        chi_max_arr[iL] = chi[iL, ie_max]
        print(f"  L={L:>5.1f}  chi_max={chi[iL, ie_max]:7.2f}  "
              f"@eta={etas[ie_max]:.3f}  phi={phi[iL, ie_max]:.3f}  "
              f"U4_min={U4[iL].min():.3f}  s_sep_max={s_sep[iL].max():.2f}")
    a, _ = np.polyfit(np.log(Ls), np.log(chi_max_arr), 1)
    print(f"\n  Vicsek-Gauss FSS slope: {a:+.3f}")


if __name__ == "__main__":
    main()
