"""
Pilot of the double-adaptive Vicsek-Couzin model. Compares four
modes that share the same code path -- only the parameter ranges
differ:

  baseline   : fixed v=v_max, fixed alpha=1     (canonical heavy-tail metric Vicsek)
  v2_limit   : fixed v=v_max, alpha in [1, 2]   (noise-shape feedback alone)
  v3_limit   : v in [v_min, v_max], alpha=1     (motility feedback alone)
  full       : v in [v_min, v_max], alpha in [1, 2]  (both feedbacks tied)

Sweep eta on a grid for L in {15, 22, 30, 45} at sigma = N/L^2 ~ 2.22,
three seeds. Records per-mode/per-L/per-eta:
  phi, chi, U4, s_sep, mean v_i, mean alpha_i.

Output: data/double_pilot.npz
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


def measure(sim, n_meas, every=1):
    phi = np.empty(n_meas // every)
    s_sep = np.empty(n_meas // every)
    v_mean = np.empty(n_meas // every)
    a_mean = np.empty(n_meas // every)
    j = 0
    for k in range(n_meas):
        sim.step()
        if k % every == 0:
            phi[j] = sim.polarisation()
            s_sep[j] = sim.density_separation_index(n_bins=10)
            v_mean[j] = float(sim.v_i.mean())
            a_mean[j] = float(sim.alpha_i.mean())
            j += 1
    return phi[:j], s_sep[:j], v_mean[:j], a_mean[:j]


def main():
    Ls = np.array([15.0, 22.0, 30.0, 45.0])
    sigma = 2.22
    etas = np.array([0.005, 0.010, 0.020, 0.035, 0.050,
                     0.075, 0.100, 0.150, 0.200, 0.300])
    seeds = [11, 23, 41]
    n_warm = 1500
    n_meas = 1000

    # mode -> (v_min, v_max, alpha_min, alpha_max)
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
                desc="double_pilot")
    t0 = time.time()
    for im, (label, vmn, vmx, amn, amx) in enumerate(modes):
        for iL, L in enumerate(Ls):
            N = int(round(sigma * L * L))
            for ie, eta in enumerate(etas):
                phi_acc = []
                s_acc = []
                v_acc = []
                a_acc = []
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
        DATA / "double_pilot.npz",
        modes=np.array([m[0] for m in modes]),
        Ls=Ls, etas=etas,
        phi=phi, chi=chi, U4=U4, s_sep=s_sep,
        v_pop=v_pop, a_pop=a_pop,
        params=np.array([sigma, n_warm, n_meas, len(seeds)]),
    )

    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
    print(f"saved: {DATA / 'double_pilot.npz'}")
    for im, (label, *_) in enumerate(modes):
        print(f"\n=== {label} ===")
        for iL, L in enumerate(Ls):
            ie_max = int(np.argmax(chi[im, iL]))
            print(f"  L={L:>5.1f}  chi_max={chi[im, iL, ie_max]:6.2f}  "
                  f"@eta={etas[ie_max]:.3f}  phi={phi[im, iL, ie_max]:.3f}  "
                  f"U4={U4[im, iL, ie_max]:.3f}  "
                  f"s_sep={s_sep[im, iL, ie_max]:.2f}")


if __name__ == "__main__":
    main()
