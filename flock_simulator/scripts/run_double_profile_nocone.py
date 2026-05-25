"""No-cone port of legacy run_double_profile.py.

Time-averaged density and speed profiles along the polar axis at
L=30, four modes at their near-critical eta. Output: band_idx
diagnoses traveling-band soliton (b > ~1.5) vs droplet (b ~ 1).
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, warm


def project_and_hist(x, y, theta_avg, L, n_bins, weights=None):
    cx, sx = np.cos(theta_avg), np.sin(theta_avg)
    x_par = (x * cx + y * sx) % L
    counts, edges = np.histogram(
        x_par, bins=n_bins, range=(0.0, L), weights=weights,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def main() -> None:
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm, n_meas, n_skip = 2000, 4000, 25
    n_bins = 60
    seeds = [11, 23, 41]

    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0, 0.035),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0, 0.075),
        ("v3_limit", 0.005, 0.05,  1.0, 1.0, 0.100),
        ("full",     0.005, 0.05,  1.0, 2.0, 0.150),
    ]
    profiles = np.zeros((len(cases), n_bins))
    v_profiles = np.zeros_like(profiles)
    eta_arr = np.zeros(len(cases))
    phi_arr = np.zeros(len(cases))
    band_idx = np.zeros(len(cases))
    labels = np.array([c[0] for c in cases])
    centers = None

    pbar = tqdm(total=len(cases) * len(seeds), desc="profile_nocone")
    for ic, (label, vmn, vmx, amn, amx, eta) in enumerate(cases):
        sum_prof = np.zeros(n_bins)
        sum_v = np.zeros(n_bins)
        sum_count_for_v = np.zeros(n_bins)
        n_acc, phi_acc = 0, 0.0
        for seed in seeds:
            p = FlockParams(
                N=N, L=L,
                v_max=float(vmx), v_min=float(vmn),
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=0.5, R_a=0.7, eta=float(eta),
                n_star=3.0, slope=2.0, seed=int(seed),
            )
            sim = FlockSimulator(p)
            warm(sim, n_warm)
            for k in range(n_meas):
                sim.step()
                if k % n_skip == 0:
                    cx = np.cos(sim.state.theta).sum()
                    sxs = np.sin(sim.state.theta).sum()
                    theta_avg = float(np.arctan2(sxs, cx))
                    cs, cn = project_and_hist(
                        sim.state.x, sim.state.y, theta_avg, L, n_bins,
                    )
                    _, cv = project_and_hist(
                        sim.state.x, sim.state.y, theta_avg, L, n_bins,
                        weights=sim.state.v_i,
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
                         sum_v / np.maximum(sum_count_for_v, 1), 0.0)
        profiles[ic] = prof
        v_profiles[ic] = vprof
        eta_arr[ic] = eta
        phi_arr[ic] = phi_acc / n_acc
        band_idx[ic] = prof.max() / prof.mean() if prof.mean() > 0 else 1.0
    pbar.close()

    out = DATA / "double_profile_nocone.npz"
    np.savez_compressed(
        out,
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
    print(f"saved: {out.name}")


if __name__ == "__main__":
    main()
