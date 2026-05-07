"""
Cluster-size distribution in the dense phase, for the four
heavy-tailed-noise modes at the near-critical eta. We bin the
particles on a coarse $n_{\rm bin} \times n_{\rm bin}$ spatial
grid, threshold at the median bin occupancy, and run a
connected-component labelling on the binary mask. Each connected
component is one cluster; we collect its size in particles. The
cluster-size distribution is the natural microscopic signature
of motility-induced phase separation: a spinodal regime gives a
power-law tail with exponent close to $-2$ in two dimensions,
while a binodal regime gives one dominant giant cluster plus a
gas of small ones.

Output: data/double_clusters.npz
   labels[mode], sizes[mode][trial] -- ragged sizes-of-clusters
   per mode (concatenated across multiple snapshots and seeds).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm
from scipy.ndimage import label

from vicsek_double_adaptive import (DoubleAdaptiveParams,
                                     DoubleAdaptiveVicsek)


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def cluster_sizes(x, y, L, n_bin=10, factor=1.5):
    """Bin on a coarse grid, threshold at factor * median, return
    sizes (in particles) of periodic-aware connected components.
    A bin width of L/10 = 3.0 at L = 30 puts on average ~20
    particles per bin at sigma = 2.22, and a contiguous patch of
    even a few bins above the threshold contains O(100)
    particles, the right scale for a MIPS dense phase."""
    H, _, _ = np.histogram2d(
        x, y, bins=[n_bin, n_bin],
        range=[[0, L], [0, L]],
    )
    nz = H[H > 0]
    if len(nz) == 0:
        return np.array([], dtype=int)
    threshold = factor * np.median(nz)
    mask = H > threshold
    if not mask.any():
        return np.array([], dtype=int)
    # Periodic-aware: tile the mask 3x3 and label, then map back
    # cluster ids that touch the central tile and merge wrapped
    # components.
    tiled = np.tile(mask, (3, 3))
    labelled, _ = label(tiled)
    central = labelled[n_bin:2 * n_bin, n_bin:2 * n_bin]
    H_tiled = np.tile(H, (3, 3))
    sizes_dict: dict[int, int] = {}
    cluster_ids = np.unique(central[central > 0])
    for cid in cluster_ids:
        sizes_dict[int(cid)] = int(H_tiled[labelled == cid].sum() / 9)
    return np.array(list(sizes_dict.values()), dtype=int)


def main():
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm = 2000
    n_meas = 4000
    n_skip = 50
    seeds = [11, 23, 41]
    n_bin = 10

    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0, 0.035),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0, 0.075),
        ("v3_limit", 0.005, 0.05,  1.0, 1.0, 0.100),
        ("full",     0.005, 0.05,  1.0, 2.0, 0.150),
    ]

    sizes_per_mode: dict[str, list[int]] = {c[0]: [] for c in cases}
    pbar = tqdm(total=len(cases) * len(seeds), desc="clusters")
    for label_name, vmn, vmx, amn, amx, eta in cases:
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
                    s = cluster_sizes(sim.x, sim.y, L, n_bin)
                    sizes_per_mode[label_name].extend(s.tolist())
            pbar.update(1)
    pbar.close()

    # Save as a dict-of-arrays via npz with per-mode keys.
    save_kwargs = {}
    save_kwargs["labels"] = np.array([c[0] for c in cases])
    save_kwargs["eta_per_mode"] = np.array([c[5] for c in cases])
    for c in cases:
        save_kwargs[f"sizes_{c[0]}"] = np.array(
            sizes_per_mode[c[0]], dtype=int)
    np.savez_compressed(DATA / "double_clusters.npz", **save_kwargs)

    print()
    for c in cases:
        sizes = np.array(sizes_per_mode[c[0]])
        if len(sizes) == 0:
            print(f"  {c[0]}: NO clusters above threshold")
            continue
        print(f"  {c[0]:10s}  n_clusters={len(sizes):5d}  "
              f"max={sizes.max():5d}  median={int(np.median(sizes)):4d}  "
              f"mean={sizes.mean():.1f}  giant_frac={sizes.max()/N:.2f}")
    print(f"saved: {DATA / 'double_clusters.npz'}")


if __name__ == "__main__":
    main()
