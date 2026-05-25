"""No-cone port of legacy run_double_gr_hs.py.

High-statistics density-stratified heading correlation g(r) at
L=30, four modes, 10 seeds, 6 snapshots each. Output:
double_gr_hs_nocone.npz.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, warm


def local_density(x, y, L, R):
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


def main() -> None:
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm, n_snap, n_skip = 2000, 6, 200
    seeds = list(range(11, 11 + 30, 3))
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
    n_modes, n_seeds = len(cases), len(seeds)
    gr_dense_per_seed = np.zeros((n_modes, n_seeds, n_bins))
    gr_dilute_per_seed = np.zeros((n_modes, n_seeds, n_bins))

    pbar = tqdm(total=n_modes * n_seeds * n_snap, desc="gr_hs_nocone")
    t0 = time.time()
    for im, (label_name, vmn, vmx, amn, amx, eta) in enumerate(cases):
        for isd, seed in enumerate(seeds):
            p = FlockParams(
                N=N, L=L,
                v_max=float(vmx), v_min=float(vmn),
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=0.5, R_a=0.7, eta=float(eta),
                n_star=3.0, slope=2.0, seed=int(seed),
            )
            sim = FlockSimulator(p)
            warm(sim, n_warm)
            cD = np.zeros(n_bins, dtype=np.int64)
            sD = np.zeros(n_bins)
            cL = np.zeros(n_bins, dtype=np.int64)
            sL = np.zeros(n_bins)
            for snap in range(n_snap):
                for _ in range(n_skip):
                    sim.step()
                rho = local_density(sim.state.x, sim.state.y, L, R_local)
                hi = np.percentile(rho, 75)
                lo = np.percentile(rho, 25)
                idx_d = np.where(rho >= hi)[0]
                idx_l = np.where(rho <= lo)[0]
                a, b = gr_for_subset(sim.state.x, sim.state.y,
                                      sim.state.theta, L, idx_d, r_edges)
                cD += a; sD += b
                a, b = gr_for_subset(sim.state.x, sim.state.y,
                                      sim.state.theta, L, idx_l, r_edges)
                cL += a; sL += b
                pbar.update(1)
            gr_dense_per_seed[im, isd] = np.where(cD > 0, sD / cD, np.nan)
            gr_dilute_per_seed[im, isd] = np.where(cL > 0, sL / cL, np.nan)
    pbar.close()

    out = DATA / "double_gr_hs_nocone.npz"
    np.savez_compressed(
        out,
        labels=np.array([c[0] for c in cases]),
        seeds=np.array(seeds),
        r_centers=r_centers,
        gr_dense_per_seed=gr_dense_per_seed,
        gr_dilute_per_seed=gr_dilute_per_seed,
        params=np.array([N, L, n_warm, n_snap, n_skip, n_seeds],
                         dtype=float),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")


if __name__ == "__main__":
    main()
