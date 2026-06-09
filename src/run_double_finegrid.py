"""
Small-eta extension of the four-mode controlled FSS, adding the
points eta in {0.001, 0.002, 0.003, 0.0075} below the existing grid
to resolve chi peaks that fall near eta_min. It sweeps the four modes
(baseline, v2_limit, v3_limit, full) at L in {15, 22, 30, 45, 64}
with 2 seeds, recording phi, chi, U4, s_sep and the mean adaptive
speed and exponent.

Output: data/double_finegrid.npz (same schema as the pilot,
restricted to the new eta values).
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
    v_mean = np.empty(n_meas)
    a_mean = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        phi[k] = sim.polarisation()
        s_sep[k] = sim.density_separation_index(n_bins=10)
        v_mean[k] = float(sim.v_i.mean())
        a_mean[k] = float(sim.alpha_i.mean())
    return phi, s_sep, v_mean, a_mean


def main():
    Ls = np.array([15.0, 22.0, 30.0, 45.0, 64.0])
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
    chi = np.zeros((n_mode, n_L, n_eta))
    U4 = np.zeros((n_mode, n_L, n_eta))
    s_sep = np.zeros((n_mode, n_L, n_eta))
    v_pop = np.zeros((n_mode, n_L, n_eta))
    a_pop = np.zeros((n_mode, n_L, n_eta))

    pbar = tqdm(total=n_mode * n_L * n_eta * len(seeds),
                desc="finegrid")
    t0 = time.time()
    for im, (label, vmn, vmx, amn, amx) in enumerate(modes):
        for iL, L in enumerate(Ls):
            N = int(round(sigma * L * L))
            for ie, eta in enumerate(etas):
                phi_acc, s_acc, v_acc, a_acc = [], [], [], []
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
                    p_arr, s_arr, v_arr, a_arr = measure(sim, n_meas)
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

    np.savez_compressed(
        DATA / "double_finegrid.npz",
        modes=np.array([m[0] for m in modes]),
        Ls=Ls, etas=etas,
        phi=phi, chi=chi, U4=U4, s_sep=s_sep,
        v_pop=v_pop, a_pop=a_pop,
        params=np.array([sigma, n_warm, n_meas, len(seeds)]),
    )

    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
    print(f"saved: {DATA / 'double_finegrid.npz'}")
    for im, (label, *_) in enumerate(modes):
        print(f"\n=== {label} ===")
        for iL, L in enumerate(Ls):
            ie_max = int(np.argmax(chi[im, iL]))
            print(f"  L={L:>5.1f}  chi_max(fine)={chi[im, iL, ie_max]:7.2f}  "
                  f"@eta={etas[ie_max]:.4f}  phi={phi[im, iL, ie_max]:.3f}")


if __name__ == "__main__":
    main()
