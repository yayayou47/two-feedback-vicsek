"""No-cone port of legacy run_double_micro_hs_L128.py.

10-seed dense/dilute g(r) + cluster sweep at L=128 for
motility-only and full modes. Closes the persistence test.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, warm


def cluster_sizes(x, y, L, n_bin=10, factor=1.5):
    H, _, _ = np.histogram2d(
        x, y, bins=[n_bin, n_bin], range=[[0, L], [0, L]])
    nz = H[H > 0]
    if len(nz) == 0:
        return np.array([], dtype=int)
    threshold = factor * np.median(nz)
    mask = H > threshold
    if not mask.any():
        return np.array([], dtype=int)
    tiled = np.tile(mask, (3, 3))
    labelled, _ = label(tiled)
    central = labelled[n_bin:2 * n_bin, n_bin:2 * n_bin]
    H_tiled = np.tile(H, (3, 3))
    sizes_dict: dict[int, int] = {}
    for cid in np.unique(central[central > 0]):
        sizes_dict[int(cid)] = int(H_tiled[labelled == cid].sum() / 9)
    return np.array(list(sizes_dict.values()), dtype=int)


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


def main() -> None:
    sigma = 2.22
    L = 128.0
    N = int(round(sigma * L * L))
    n_warm, n_meas = 1500, 4000
    n_skip_cl, n_snap_gr, n_skip_gr = 50, 6, 200
    seeds = list(range(11, 11 + 30, 3))
    n_bin = 10
    R_local = 1.0
    r_edges = np.linspace(0.5, 6.0, 24)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_bins = len(r_centers)

    modes = [
        ("motility", 0.005, 0.05, 1.0, 1.0, 0.100),
        ("full",     0.005, 0.05, 1.0, 2.0, 0.150),
    ]
    n_modes, n_seeds = len(modes), len(seeds)
    n_per_seed = np.zeros((n_modes, n_seeds), dtype=int)
    max_per_seed = np.zeros((n_modes, n_seeds), dtype=int)
    gr_dense = np.zeros((n_modes, n_seeds, n_bins))
    gr_dilute = np.zeros((n_modes, n_seeds, n_bins))

    pbar = tqdm(total=n_modes * n_seeds, desc=f"micro_hs_L128_nocone")
    t0 = time.time()
    for im, (label_name, vmn, vmx, amn, amx, eta) in enumerate(modes):
        for isd, seed in enumerate(seeds):
            p = FlockParams(
                N=N, L=float(L),
                v_max=float(vmx), v_min=float(vmn),
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=0.5, R_a=0.7, eta=float(eta),
                n_star=3.0, slope=2.0, seed=int(seed),
            )
            sim = FlockSimulator(p)
            warm(sim, n_warm)
            sizes_seed: list[int] = []
            gr_done = 0
            next_gr_step = n_skip_gr - 1
            cD = np.zeros(n_bins, dtype=np.int64); sD = np.zeros(n_bins)
            cL = np.zeros(n_bins, dtype=np.int64); sL = np.zeros(n_bins)
            for k in range(n_meas):
                sim.step()
                if k % n_skip_cl == 0:
                    s = cluster_sizes(sim.state.x, sim.state.y,
                                       float(L), n_bin)
                    sizes_seed.extend(s.tolist())
                if gr_done < n_snap_gr and k == next_gr_step:
                    rho = local_density(sim.state.x, sim.state.y,
                                         float(L), R_local)
                    hi = np.percentile(rho, 75)
                    lo = np.percentile(rho, 25)
                    idx_d = np.where(rho >= hi)[0]
                    idx_l = np.where(rho <= lo)[0]
                    a, b = gr_for_subset(sim.state.x, sim.state.y,
                                          sim.state.theta, float(L),
                                          idx_d, r_edges)
                    cD += a; sD += b
                    a, b = gr_for_subset(sim.state.x, sim.state.y,
                                          sim.state.theta, float(L),
                                          idx_l, r_edges)
                    cL += a; sL += b
                    gr_done += 1
                    next_gr_step += n_skip_gr
            arr = np.array(sizes_seed, dtype=int)
            n_per_seed[im, isd] = len(arr)
            max_per_seed[im, isd] = int(arr.max()) if len(arr) else 0
            gr_dense[im, isd] = np.where(cD > 0, sD / cD, np.nan)
            gr_dilute[im, isd] = np.where(cL > 0, sL / cL, np.nan)
            pbar.update(1)
    pbar.close()

    out = DATA / "double_micro_hs_L128_nocone.npz"
    np.savez_compressed(
        out,
        modes=np.array([m[0] for m in modes]),
        L=L, seeds=np.array(seeds), r_centers=r_centers,
        n_per_seed=n_per_seed,
        max_per_seed=max_per_seed,
        gr_dense_per_seed=gr_dense,
        gr_dilute_per_seed=gr_dilute,
        params=np.array([sigma, n_warm, n_meas, n_skip_cl,
                          n_snap_gr, n_skip_gr, n_seeds, n_bin],
                          dtype=float),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")


if __name__ == "__main__":
    main()
