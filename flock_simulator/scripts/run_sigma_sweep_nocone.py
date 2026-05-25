"""
Density (sigma) sweep that tests the prediction of the 2D
decoupled-sigmoid map: the critical motility threshold at which
the dense-quartile gap Delta_g changes sign should scale
linearly with the global density sigma.

For each sigma in {1.0, 1.5, 3.0} (sigma = 2.22 reuses the 2D
map data) we hold L = 30, n_star_alpha = 3, eta = 0.15, and
sweep n_star_v over {1, 2, 3, 4, 5, 6, 8}. For each cell we
measure the dense-quartile g(r ~ 0.6) and subtract, seed by
seed, the motility-only ablation at the same sigma and seeds.

Output: data/double_sigma_sweep_nocone.npz with per-seed g(r),
the motility references, and the paired-diff gap at r ~ 0.62
per (sigma, n_star_v).

Run from version4/ as:
  ../.venv/bin/python flock_simulator/scripts/run_sigma_sweep_nocone.py
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, warm


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
        rho[i] = int(np.sum(dx * dx + dy * dy < R2)) - 1
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


def measure_cell(N, L, eta, nsv, nsa, alpha_min, alpha_max,
                  seeds, n_warm, n_snap, n_skip, r_edges,
                  n_bins, R_local):
    gr_dense = np.zeros((len(seeds), n_bins))
    gr_dilute = np.zeros((len(seeds), n_bins))
    for isd, seed in enumerate(seeds):
        p = FlockParams(
            N=N, L=L,
            v_max=0.05, v_min=0.005,
            alpha_min=float(alpha_min), alpha_max=float(alpha_max),
            R_r=0.5, R_a=0.7, eta=float(eta),
            n_star=3.0, slope=2.0,
            n_star_v=float(nsv), slope_v=2.0,
            n_star_alpha=float(nsa), slope_alpha=2.0,
            seed=int(seed),
        )
        sim = FlockSimulator(p)
        warm(sim, n_warm)
        cD = np.zeros(n_bins, dtype=np.int64); sD = np.zeros(n_bins)
        cL = np.zeros(n_bins, dtype=np.int64); sL = np.zeros(n_bins)
        for _ in range(n_snap):
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
        gr_dense[isd] = np.where(cD > 0, sD / cD, np.nan)
        gr_dilute[isd] = np.where(cL > 0, sL / cL, np.nan)
    return gr_dense, gr_dilute


def main() -> None:
    L = 30.0
    n_warm, n_snap, n_skip = 2000, 6, 200
    seeds = list(range(11, 11 + 30, 3))
    R_local = 1.0
    r_edges = np.linspace(0.5, 6.0, 24)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_bins = len(r_centers)
    eta = 0.150

    sigmas = np.array([1.0, 1.5, 3.0])
    n_v_grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
    n_a = 3.0

    n_s = len(sigmas)
    n_nv = len(n_v_grid)
    gr_dense_full = np.zeros((n_s, n_nv, len(seeds), n_bins))
    gr_dense_mot = np.zeros((n_s, len(seeds), n_bins))
    N_per_sigma = np.zeros(n_s, dtype=int)

    pbar = tqdm(total=n_s * (n_nv + 1), desc="sigma_sweep")
    t0 = time.time()
    for isg, sigma in enumerate(sigmas):
        N = int(round(sigma * L * L))
        N_per_sigma[isg] = N

        # Motility-only reference (alpha frozen at 1).
        gd, _ = measure_cell(N, L, eta, 3.0, 3.0, 1.0, 1.0,
                              seeds, n_warm, n_snap, n_skip,
                              r_edges, n_bins, R_local)
        gr_dense_mot[isg] = gd
        pbar.update(1)

        for iv, nsv in enumerate(n_v_grid):
            gd_full, _ = measure_cell(
                N, L, eta, nsv, n_a, 1.0, 2.0,
                seeds, n_warm, n_snap, n_skip,
                r_edges, n_bins, R_local,
            )
            gr_dense_full[isg, iv] = gd_full
            pbar.update(1)
    pbar.close()

    out = DATA / "double_sigma_sweep_nocone.npz"
    np.savez_compressed(
        out,
        sigmas=sigmas, n_v_grid=n_v_grid, n_a=n_a,
        N_per_sigma=N_per_sigma,
        seeds=np.array(seeds),
        r_centers=r_centers,
        gr_dense_full=gr_dense_full,
        gr_dense_motility=gr_dense_mot,
        params=np.array([L, eta, n_warm, n_snap, n_skip,
                          len(seeds)], dtype=float),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min   "
          f"saved: {out.name}")
    j = int(np.argmin(np.abs(r_centers - 0.62)))
    print(f"\nDelta_g(r ~ 0.62), paired vs motility, "
          f"n_star_alpha = {n_a:.0f}:")
    print(f"  {'sigma':>5s} " + " ".join(
        f"{'n*v=' + str(int(nv)):>10s}" for nv in n_v_grid))
    for isg, sigma in enumerate(sigmas):
        row = []
        for iv in range(len(n_v_grid)):
            diff = gr_dense_full[isg, iv, :, j] - gr_dense_mot[isg, :, j]
            ok = np.isfinite(diff)
            dm = diff[ok].mean() if ok.any() else float("nan")
            row.append(f"{dm:+10.3f}")
        print(f"  {sigma:>5.2f} " + " ".join(row))


if __name__ == "__main__":
    main()
