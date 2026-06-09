"""
Compute the time-averaged density profile along the polar-flow
axis for the baseline, v2-limit, v3-limit, and full two-feedback
modes at $L = 30$ ($\\sigma = 2.22$, seeds 11/23/41), each at its
own near-critical eta. Projects positions onto the instantaneous
mean-heading direction, histograms them into 60 bins (density and
v-weighted), and records phi and the band index max/mean per mode.

Output: data/double_profile.npz
   labels, centers, profiles, v_profiles, eta, phi, band_idx.
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


def project_and_hist(x, y, theta_avg, L, n_bins, weights=None):
    cx, sx = np.cos(theta_avg), np.sin(theta_avg)
    x_par = (x * cx + y * sx) % L
    counts, edges = np.histogram(
        x_par, bins=n_bins, range=(0.0, L), weights=weights
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def main():
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm = 2000
    n_meas = 4000
    n_skip = 25
    n_bins = 60
    seeds = [11, 23, 41]

    # mode tuned to its own near-critical eta
    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0, 0.035),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0, 0.075),
        ("v3_limit", 0.005, 0.05,  1.0, 1.0, 0.100),
        ("full",     0.005, 0.05,  1.0, 2.0, 0.150),
    ]

    profiles = np.zeros((len(cases), n_bins))
    v_profiles = np.zeros((len(cases), n_bins))
    eta_arr = np.zeros(len(cases))
    phi_arr = np.zeros(len(cases))
    band_idx = np.zeros(len(cases))
    labels = np.array([c[0] for c in cases])
    centers = None

    pbar = tqdm(total=len(cases) * len(seeds), desc="profile")
    for ic, (label, vmn, vmx, amn, amx, eta) in enumerate(cases):
        sum_prof = np.zeros(n_bins)
        sum_v = np.zeros(n_bins)
        sum_count_for_v = np.zeros(n_bins)
        n_acc = 0
        phi_acc = 0.0
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
            sim = DoubleAdaptiveVicsek(p)
            sim.theta[:] = 0.0
            for _ in range(n_warm):
                sim.step()
            for k in range(n_meas):
                sim.step()
                if k % n_skip == 0:
                    cx = np.cos(sim.theta).sum()
                    sxs = np.sin(sim.theta).sum()
                    theta_avg = float(np.arctan2(sxs, cx))
                    cs, cn = project_and_hist(
                        sim.x, sim.y, theta_avg, L, n_bins
                    )
                    _, cv = project_and_hist(
                        sim.x, sim.y, theta_avg, L, n_bins,
                        weights=sim.v_i,
                    )
                    if centers is None:
                        centers = cs
                    sum_prof += cn
                    sum_v += cv
                    sum_count_for_v += cn
                    n_acc += 1
                    phi_acc += sim.polarisation()
            pbar.update(1)
        prof = sum_prof / n_acc
        vprof = np.where(sum_count_for_v > 0,
                         sum_v / np.maximum(sum_count_for_v, 1),
                         0.0)
        profiles[ic] = prof
        v_profiles[ic] = vprof
        eta_arr[ic] = eta
        phi_arr[ic] = phi_acc / n_acc
        band_idx[ic] = prof.max() / prof.mean() if prof.mean() > 0 else 1.0
    pbar.close()

    np.savez_compressed(
        DATA / "double_profile.npz",
        labels=labels, centers=centers,
        profiles=profiles, v_profiles=v_profiles,
        eta=eta_arr, phi=phi_arr, band_idx=band_idx,
        params=np.array([N, L, n_warm, n_meas, n_skip,
                          len(seeds)], dtype=float),
    )

    print()
    for ic, lbl in enumerate(labels):
        print(f"  {lbl:10s}  eta={eta_arr[ic]:.3f}  "
              f"phi={phi_arr[ic]:.3f}  band_idx={band_idx[ic]:.3f}")
    print("saved:", DATA / "double_profile.npz")


if __name__ == "__main__":
    main()
