"""
Measures the density-stratified heading correlation g(r) at
L = 30, sigma = 2.22, eta = 0.150, over ten seeds, for the
decoupled-sigmoid Vicsek model in four configurations of the
speed and noise-shape thresholds:

  motility    : alpha frozen at 1 (reference)
  shared      : n_star_v = n_star_alpha = 3
  decoupled_A : n_star_v = 1, n_star_alpha = 5
  decoupled_B : n_star_v = 5, n_star_alpha = 1

Uses 2000 warm-up and six snapshots spaced by 200 steps, storing
per-seed dense and dilute g(r) curves per mode.

Output: data/double_decoupled.npz
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

from vicsek_decoupled import DecoupledParams, DecoupledVicsek


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def local_density(x, y, L, R=1.0):
    N = len(x)
    rho = np.zeros(N, dtype=int)
    halfL = 0.5 * L
    R2 = R * R
    for i in range(N):
        dx = x - x[i]; dy = y - y[i]
        dx = np.where(dx > halfL, dx - L, dx)
        dx = np.where(dx < -halfL, dx + L, dx)
        dy = np.where(dy > halfL, dy - L, dy)
        dy = np.where(dy < -halfL, dy + L, dy)
        d2 = dx * dx + dy * dy
        rho[i] = int(np.sum(d2 < R2)) - 1
    return rho


def gr_for_subset(x, y, theta, L, idx, r_edges):
    halfL = 0.5 * L
    n_bins = len(r_edges) - 1
    counts = np.zeros(n_bins, dtype=np.int64)
    sumc = np.zeros(n_bins)
    for i in idx:
        dx = x - x[i]; dy = y - y[i]
        dx = np.where(dx > halfL, dx - L, dx)
        dx = np.where(dx < -halfL, dx + L, dx)
        dy = np.where(dy > halfL, dy - L, dy)
        dy = np.where(dy < -halfL, dy + L, dy)
        d = np.sqrt(dx * dx + dy * dy)
        cos_dt = np.cos(theta - theta[i])
        mask = (d > 0.0) & (d < r_edges[-1])
        bin_idx = np.searchsorted(r_edges, d[mask]) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        np.add.at(counts, bin_idx, 1)
        np.add.at(sumc, bin_idx, cos_dt[mask])
    return counts, sumc


def main():
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm = 2000
    n_snap = 6
    n_skip = 200
    seeds = list(range(11, 11 + 30, 3))
    R_local = 1.0
    r_edges = np.linspace(0.5, 6.0, 24)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_bins = len(r_centers)
    eta = 0.150  # near-critical eta of the full mode

    # mode -> (n_star_v, slope_v, n_star_alpha, slope_alpha,
    #          alpha_min, alpha_max)
    modes = [
        ("motility",     3.0, 2.0, 3.0, 2.0, 1.0, 1.0),
        ("shared",       3.0, 2.0, 3.0, 2.0, 1.0, 2.0),
        ("decoupled_A",  1.0, 2.0, 5.0, 2.0, 1.0, 2.0),
        ("decoupled_B",  5.0, 2.0, 1.0, 2.0, 1.0, 2.0),
    ]

    n_modes = len(modes)
    n_seeds = len(seeds)
    gr_dense = np.zeros((n_modes, n_seeds, n_bins))
    gr_dilute = np.zeros((n_modes, n_seeds, n_bins))

    pbar = tqdm(total=n_modes * n_seeds * n_snap,
                 desc="decoupled")
    t0 = time.time()
    for im, (label, nsv, slv, nsa, sla, amn, amx) in enumerate(modes):
        for isd, seed in enumerate(seeds):
            p = DecoupledParams(
                N=N, L=L,
                v_max=0.05, v_min=0.005,
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=0.5, R_a=0.7, beta=30.0,
                eta=float(eta),
                n_star_v=float(nsv), slope_v=float(slv),
                n_star_alpha=float(nsa), slope_alpha=float(sla),
                seed=int(seed),
            )
            sim = DecoupledVicsek(p)
            sim.theta[:] = 0.0
            for _ in range(n_warm):
                sim.step()
            cD = np.zeros(n_bins, dtype=np.int64); sD = np.zeros(n_bins)
            cL = np.zeros(n_bins, dtype=np.int64); sL = np.zeros(n_bins)
            for snap in range(n_snap):
                for _ in range(n_skip):
                    sim.step()
                rho = local_density(sim.x, sim.y, L, R_local)
                hi = np.percentile(rho, 75)
                lo = np.percentile(rho, 25)
                idx_d = np.where(rho >= hi)[0]
                idx_l = np.where(rho <= lo)[0]
                a, b = gr_for_subset(sim.x, sim.y, sim.theta, L,
                                      idx_d, r_edges)
                cD += a; sD += b
                a, b = gr_for_subset(sim.x, sim.y, sim.theta, L,
                                      idx_l, r_edges)
                cL += a; sL += b
                pbar.update(1)
            gr_dense[im, isd] = np.where(cD > 0, sD / cD, np.nan)
            gr_dilute[im, isd] = np.where(cL > 0, sL / cL, np.nan)
    pbar.close()

    np.savez_compressed(
        DATA / "double_decoupled.npz",
        modes=np.array([m[0] for m in modes]),
        seeds=np.array(seeds),
        r_centers=r_centers,
        gr_dense_per_seed=gr_dense,
        gr_dilute_per_seed=gr_dilute,
        params=np.array([N, L, n_warm, n_snap, n_skip,
                          n_seeds, eta], dtype=float),
    )

    # Report.
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
    j = 0   # r ~ 0.62
    print(f"\ng(r=0.62) dense quartile, mean ± SE, n_seeds = {n_seeds}:")
    for im, (lbl, *_) in enumerate(modes):
        gd = gr_dense[im, :, j]
        gd_m = np.nanmean(gd)
        gd_e = (np.nanstd(gd, ddof=1)
                 / np.sqrt(np.sum(~np.isnan(gd))))
        print(f"  {lbl:>12s}: {gd_m:+.3f} ± {gd_e:.3f}")
    print()
    # Pairwise gap vs motility.
    print("Gap (mode - motility) at r=0.62, with z-score:")
    for im, (lbl, *_) in enumerate(modes):
        if lbl == "motility":
            continue
        diff = gr_dense[im, :, j] - gr_dense[0, :, j]
        ok = np.isfinite(diff)
        d_m = diff[ok].mean()
        d_e = diff[ok].std(ddof=1) / np.sqrt(ok.sum())
        z = d_m / max(d_e, 1e-9)
        print(f"  {lbl:>12s}: delta = {d_m:+.3f} ± {d_e:.3f}  "
              f"(z = {z:+.2f})")


if __name__ == "__main__":
    main()
