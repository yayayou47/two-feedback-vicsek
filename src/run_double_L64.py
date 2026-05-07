"""
Controlled FSS extension of the four-mode pilot to L = 64. Uses
the same eta grid and the same seed budget as the pilot, so the
five-size log-log fit (15, 22, 30, 45, 64) refits each mode's
slope on a wider FSS lever and tests whether the pilot ordering
$a_{\rm baseline} < a_{v2} < a_{v3} < a_{\rm full}$ -- and the
super-additive synergy diagnostic $\Delta = +0.26$ -- survive.

Output: data/double_L64.npz with the same schema as
data/double_pilot.npz but at the single size L = 64. The figure
renderer will concatenate.
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
    L = 64.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    etas = np.array([0.005, 0.010, 0.020, 0.035, 0.050,
                     0.075, 0.100, 0.150, 0.200, 0.300])
    seeds = [11, 23, 41]
    n_warm = 1500
    n_meas = 1000

    modes = [
        ("baseline", 0.05, 0.05, 1.0, 1.0),
        ("v2_limit", 0.05, 0.05, 1.0, 2.0),
        ("v3_limit", 0.005, 0.05, 1.0, 1.0),
        ("full",     0.005, 0.05, 1.0, 2.0),
    ]

    n_mode, n_eta = len(modes), len(etas)
    phi = np.zeros((n_mode, n_eta))
    chi = np.zeros((n_mode, n_eta))
    U4 = np.zeros((n_mode, n_eta))
    s_sep = np.zeros((n_mode, n_eta))
    v_pop = np.zeros((n_mode, n_eta))
    a_pop = np.zeros((n_mode, n_eta))

    pbar = tqdm(total=n_mode * n_eta * len(seeds),
                desc=f"double_L{int(L)}")
    t0 = time.time()
    for im, (label, vmn, vmx, amn, amx) in enumerate(modes):
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
            phi[im, ie] = phi_all.mean()
            chi[im, ie] = N * phi_all.var()
            U4[im, ie] = (
                1.0 - np.mean(phi_all ** 4)
                / (3.0 * np.mean(phi_all ** 2) ** 2)
            )
            s_sep[im, ie] = np.concatenate(s_acc).mean()
            v_pop[im, ie] = np.concatenate(v_acc).mean()
            a_pop[im, ie] = np.concatenate(a_acc).mean()
    pbar.close()

    np.savez_compressed(
        DATA / "double_L64.npz",
        modes=np.array([m[0] for m in modes]),
        L=L, etas=etas,
        phi=phi, chi=chi, U4=U4, s_sep=s_sep,
        v_pop=v_pop, a_pop=a_pop,
        params=np.array([sigma, n_warm, n_meas, len(seeds)]),
    )

    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
    print(f"saved: {DATA / 'double_L64.npz'}")
    for im, (label, *_) in enumerate(modes):
        ie_max = int(np.argmax(chi[im]))
        print(f"  {label:10s}  chi_max={chi[im, ie_max]:7.2f}  "
              f"@eta={etas[ie_max]:.3f}  phi={phi[im, ie_max]:.3f}  "
              f"U4={U4[im, ie_max]:.3f}  s_sep={s_sep[im, ie_max]:.2f}")


if __name__ == "__main__":
    main()
