"""
High-statistics cluster-size measurement at L = 30 with ten
seeds. We aggregate the cluster ensemble within each seed and
report per-seed (n_clusters, max_size, mean_size) so that
seed-level standard errors can be quoted on the differences.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from tqdm import tqdm
from scipy.ndimage import label

from vicsek_double_adaptive import (DoubleAdaptiveParams,
                                     DoubleAdaptiveVicsek)


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


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


def main():
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm = 2000
    n_meas = 4000
    n_skip = 50
    seeds = list(range(11, 11 + 30, 3))   # 10 seeds
    n_bin = 10

    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0, 0.035),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0, 0.075),
        ("v3_limit", 0.005, 0.05,  1.0, 1.0, 0.100),
        ("full",     0.005, 0.05,  1.0, 2.0, 0.150),
    ]

    n_modes = len(cases)
    n_seeds = len(seeds)
    n_per_seed = np.zeros((n_modes, n_seeds), dtype=int)
    max_per_seed = np.zeros((n_modes, n_seeds), dtype=int)
    mean_per_seed = np.zeros((n_modes, n_seeds))

    pbar = tqdm(total=n_modes * n_seeds, desc="clusters_hs")
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
            sizes_seed: list[int] = []
            for k in range(n_meas):
                sim.step()
                if k % n_skip == 0:
                    s = cluster_sizes(sim.x, sim.y, L, n_bin)
                    sizes_seed.extend(s.tolist())
            arr = np.array(sizes_seed, dtype=int)
            n_per_seed[im, isd] = len(arr)
            max_per_seed[im, isd] = int(arr.max()) if len(arr) else 0
            mean_per_seed[im, isd] = float(arr.mean()) if len(arr) else 0.0
            pbar.update(1)
    pbar.close()

    np.savez_compressed(
        DATA / "double_clusters_hs.npz",
        labels=np.array([c[0] for c in cases]),
        seeds=np.array(seeds),
        n_per_seed=n_per_seed,
        max_per_seed=max_per_seed,
        mean_per_seed=mean_per_seed,
        params=np.array([N, L, n_warm, n_meas, n_skip,
                          n_seeds, n_bin], dtype=float),
    )

    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
    print(f"\n{'mode':>10s}  {'<n_cl>':>10s}  {'<max>':>10s}  "
          f"{'<mean>':>10s}")
    for im, c in enumerate(cases):
        n_mean = n_per_seed[im].mean()
        n_se = n_per_seed[im].std(ddof=1) / np.sqrt(n_seeds)
        max_mean = max_per_seed[im].mean()
        max_se = max_per_seed[im].std(ddof=1) / np.sqrt(n_seeds)
        mean_mean = mean_per_seed[im].mean()
        mean_se = mean_per_seed[im].std(ddof=1) / np.sqrt(n_seeds)
        print(f"  {c[0]:>10s}  "
              f"{n_mean:6.1f}±{n_se:.1f}  "
              f"{max_mean:5.1f}±{max_se:.1f}  "
              f"{mean_mean:5.2f}±{mean_se:.2f}")

    # Pairwise z-scores: full vs motility (v3_limit)
    print("\nPairwise full vs motility on max cluster size:")
    diff = (max_per_seed[3] - max_per_seed[2])  # full - motility
    z = diff.mean() / (diff.std(ddof=1) / np.sqrt(n_seeds))
    print(f"  delta(max) = {diff.mean():+.2f} +/- "
          f"{diff.std(ddof=1)/np.sqrt(n_seeds):.2f}  (z = {z:+.2f})")
    diff = (n_per_seed[3] - n_per_seed[2])  # full - motility
    z = diff.mean() / max(diff.std(ddof=1) / np.sqrt(n_seeds), 1e-9)
    print(f"  delta(n_clusters) = {diff.mean():+.1f} +/- "
          f"{diff.std(ddof=1)/np.sqrt(n_seeds):.1f}  (z = {z:+.2f})")


if __name__ == "__main__":
    main()
