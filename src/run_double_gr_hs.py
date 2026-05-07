r"""
High-statistics heading correlation $g(r)$ at L = 30 with ten
seeds, density-stratified by top/bottom quartile of local
density. Per-seed g(r) is saved so seed-level standard errors
can be quoted on the gap full vs motility.
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


def local_density(x, y, L, R=1.0):
    N = len(x)
    rho = np.zeros(N, dtype=int)
    halfL = 0.5 * L
    R2 = R * R
    for i in range(N):
        dx = x - x[i]
        dy = y - y[i]
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
        dx = x - x[i]
        dy = y - y[i]
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
    seeds = list(range(11, 11 + 30, 3))   # 10 seeds
    R_local = 1.0
    r_edges = np.linspace(0.5, 6.0, 24)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_bins = len(r_centers)

    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0, 0.035),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0, 0.075),
        ("v3_limit", 0.005, 0.05,  1.0, 1.0, 0.100),
        ("full",     0.005, 0.05,  1.0, 2.0, 0.150),
    ]

    n_modes = len(cases)
    n_seeds = len(seeds)
    gr_dense_per_seed = np.zeros((n_modes, n_seeds, n_bins))
    gr_dilute_per_seed = np.zeros((n_modes, n_seeds, n_bins))

    pbar = tqdm(total=n_modes * n_seeds * n_snap, desc="g(r)_hs")
    t0 = time.time()
    for im, (label_name, vmn, vmx, amn, amx, eta) in enumerate(cases):
        for isd, seed in enumerate(seeds):
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
            cD = np.zeros(n_bins, dtype=np.int64)
            sD = np.zeros(n_bins)
            cL = np.zeros(n_bins, dtype=np.int64)
            sL = np.zeros(n_bins)
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
            gr_dense_per_seed[im, isd] = np.where(cD > 0, sD / cD,
                                                   np.nan)
            gr_dilute_per_seed[im, isd] = np.where(cL > 0, sL / cL,
                                                    np.nan)
    pbar.close()

    np.savez_compressed(
        DATA / "double_gr_hs.npz",
        labels=np.array([c[0] for c in cases]),
        seeds=np.array(seeds),
        r_centers=r_centers,
        gr_dense_per_seed=gr_dense_per_seed,
        gr_dilute_per_seed=gr_dilute_per_seed,
        params=np.array([N, L, n_warm, n_snap, n_skip,
                          n_seeds], dtype=float),
    )

    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
    # Pick the first inside-the-alignment-zone bin for the headline
    # comparison.
    j = 0
    print(f"\ng(r) at r = {r_centers[j]:.2f}, dense / dilute, "
          f"mean ± SE, n_seeds = {n_seeds}:")
    for im, c in enumerate(cases):
        gd = gr_dense_per_seed[im, :, j]
        gl = gr_dilute_per_seed[im, :, j]
        gd_m = np.nanmean(gd); gd_se = np.nanstd(gd, ddof=1) / np.sqrt(np.sum(~np.isnan(gd)))
        gl_m = np.nanmean(gl); gl_se = np.nanstd(gl, ddof=1) / np.sqrt(np.sum(~np.isnan(gl)))
        print(f"  {c[0]:>10s}  dense {gd_m:+.3f}±{gd_se:.3f}    "
              f"dilute {gl_m:+.3f}±{gl_se:.3f}")

    # Pairwise z-score: full dense - motility dense
    print("\nPairwise full vs motility on dense-quartile g(r):")
    for j_test in (0, 2, 5):
        full = gr_dense_per_seed[3, :, j_test]
        mot = gr_dense_per_seed[2, :, j_test]
        diff = full - mot
        ok = np.isfinite(diff)
        d_mean = diff[ok].mean()
        d_se = diff[ok].std(ddof=1) / np.sqrt(ok.sum())
        z = d_mean / max(d_se, 1e-9)
        print(f"  r = {r_centers[j_test]:.2f}: "
              f"delta = {d_mean:+.3f} ± {d_se:.3f}  (z = {z:+.2f})")


if __name__ == "__main__":
    main()
