"""
Two-dimensional sweep of the decoupled-sigmoid timing test.

The legacy decoupled run sampled three configurations only:
shared ((3, 3)), motility-first ((1, 5)) and alpha-first
((5, 1)). The 2D map here resolves the full
(n_star_v, n_star_alpha) quadrant on a 5x5 grid in
{1, 2, 3, 5, 8}, with the motility-only ablation as the
paired reference, ten seeds per cell, at L = 30.

For every cell we measure the dense-quartile heading
correlation g(r ~= 0.6) and subtract, seed-by-seed, the same
quantity for the motility-only ablation; this gives a paired
gap Delta_g(n_star_v, n_star_alpha) whose sign discriminates
amplifying vs damaging timing.

Output: data/double_decoupled_2d_nocone.npz with the per-seed
g(r) curves, the paired-diff gap and SE at r~0.62, and the
fit Delta_g vs (n_star_v - n_star_alpha) that the legacy
three-point measurement could not resolve.
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


def run_one(nsv, nsa, alpha_min, alpha_max, seeds, n_warm,
             n_snap, n_skip, L, N, eta, r_edges, n_bins,
             R_local):
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
        for _snap in range(n_snap):
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
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm, n_snap, n_skip = 2000, 6, 200
    seeds = list(range(11, 11 + 30, 3))
    R_local = 1.0
    r_edges = np.linspace(0.5, 6.0, 24)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_bins = len(r_centers)
    eta = 0.150

    n_star_grid = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    n_v_grid = n_star_grid
    n_a_grid = n_star_grid

    # Motility-only reference at the same seeds and warmup.
    print("Running motility-only reference (10 seeds)...")
    t0 = time.time()
    gr_dense_mot, gr_dilute_mot = run_one(
        3.0, 3.0, 1.0, 1.0, seeds, n_warm, n_snap, n_skip,
        L, N, eta, r_edges, n_bins, R_local,
    )
    print(f"  reference done in {(time.time() - t0) / 60:.1f} min")

    n_cells = len(n_v_grid) * len(n_a_grid)
    gr_dense = np.zeros((len(n_v_grid), len(n_a_grid),
                         len(seeds), n_bins))
    gr_dilute = np.zeros_like(gr_dense)
    pbar = tqdm(total=n_cells, desc="decoupled_2d")
    t0 = time.time()
    for iv, nsv in enumerate(n_v_grid):
        for ia, nsa in enumerate(n_a_grid):
            gd, gl = run_one(
                nsv, nsa, 1.0, 2.0, seeds, n_warm, n_snap,
                n_skip, L, N, eta, r_edges, n_bins, R_local,
            )
            gr_dense[iv, ia] = gd
            gr_dilute[iv, ia] = gl
            pbar.update(1)
    pbar.close()

    out = DATA / "double_decoupled_2d_nocone.npz"
    np.savez_compressed(
        out,
        n_v_grid=n_v_grid, n_a_grid=n_a_grid,
        seeds=np.array(seeds),
        r_centers=r_centers,
        gr_dense=gr_dense, gr_dilute=gr_dilute,
        gr_dense_motility=gr_dense_mot,
        gr_dilute_motility=gr_dilute_mot,
        params=np.array([N, L, n_warm, n_snap, n_skip,
                          len(seeds), eta], dtype=float),
    )
    print()
    print(f"runtime (grid): {(time.time() - t0) / 60:.1f} min")
    print(f"saved: {out.name}")

    # Headline: paired Delta_g at r ~ 0.62.
    j = int(np.argmin(np.abs(r_centers - 0.62)))
    print(f"\nPaired Delta_g (full[seed] - motility[seed]) at "
          f"r = {r_centers[j]:.3f}, 10 seeds:")
    print(f"{'n_v\\n_a':>10s} " + " ".join(
        f"{na:>8.0f}" for na in n_a_grid))
    for iv, nsv in enumerate(n_v_grid):
        row = []
        for ia in range(len(n_a_grid)):
            diff = gr_dense[iv, ia, :, j] - gr_dense_mot[:, j]
            ok = np.isfinite(diff)
            dm = diff[ok].mean() if ok.any() else float("nan")
            row.append(f"{dm:+8.3f}")
        print(f"{nsv:>10.0f} " + " ".join(row))


if __name__ == "__main__":
    main()
