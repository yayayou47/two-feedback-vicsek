"""
Patch-lifetime tracking: do dense clusters persist longer in the
double-adaptive (full) model than in the motility-only ablation?

For each mode at its near-critical eta (motility 0.10, full 0.15)
and each seed, run a long trajectory at L = 30. Every n_skip
steps:

  1. histogram particles onto a 10x10 grid;
  2. threshold cells at 1.5 x median non-empty occupancy and
     extract periodic connected components (the same procedure
     used for the cluster-size measurement);
  3. compute each connected component's centroid (periodic mean)
     and total particle count.

Across snapshots, match every component to a component of the
previous snapshot via two criteria: (a) periodic centroid
distance below ``d_match`` and (b) particle-count ratio in
``[1/r, r]``. Unmatched components start new patches; matched
components continue them. The patch lifetime is the number of
consecutive snapshots over which it was tracked.

The hypothesis tested is that the dense-phase rectification
stabilises patches as persistent geometric objects, so
<tau_patch>_full > <tau_patch>_motility.

Output: data/double_patch_lifetime.npz with per-seed lifetime
arrays, the per-mode aggregate distribution, and basic
descriptive statistics.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, warm


def cluster_components(x, y, L, n_bin=10, factor=1.5):
    """Return (cell_mask, particle_counts, centroids) for the
    threshold-and-label cluster geometry on a periodic 10x10 grid."""
    H, _, _ = np.histogram2d(
        x, y, bins=[n_bin, n_bin], range=[[0, L], [0, L]])
    nz = H[H > 0]
    if len(nz) == 0:
        return [], np.array([]), np.zeros((0, 2))
    threshold = factor * np.median(nz)
    mask = H > threshold
    if not mask.any():
        return [], np.array([]), np.zeros((0, 2))
    tiled = np.tile(mask, (3, 3))
    labelled, _ = label(tiled)
    central = labelled[n_bin:2 * n_bin, n_bin:2 * n_bin]
    H_tiled = np.tile(H, (3, 3))
    cell_size = L / n_bin

    sizes = []
    centroids = []
    for cid in np.unique(central[central > 0]):
        # Cells of this component in the tiled grid; pick the one
        # that overlaps the central tile to anchor the periodic
        # centroid calculation.
        bin_mask = labelled == cid
        # Particle count: same as the per-cell summed occupancy / 9.
        size = int(H_tiled[bin_mask].sum() / 9)
        # Cells in the central tile that belong to this component.
        ii, jj = np.where((central == cid))
        if len(ii) == 0:
            continue
        # Periodic centroid via circular-mean across the cells in
        # the central tile (using cell-centre coordinates).
        cx = (ii + 0.5) * cell_size
        cy = (jj + 0.5) * cell_size
        # circular mean over the periodic box of side L
        ang_x = 2 * np.pi * cx / L
        ang_y = 2 * np.pi * cy / L
        cx_bar = np.arctan2(np.sin(ang_x).mean(),
                             np.cos(ang_x).mean()) * L / (2 * np.pi) % L
        cy_bar = np.arctan2(np.sin(ang_y).mean(),
                             np.cos(ang_y).mean()) * L / (2 * np.pi) % L
        sizes.append(size)
        centroids.append((cx_bar, cy_bar))
    return list(range(len(sizes))), np.array(sizes), np.array(centroids)


def periodic_dist(c1, c2, L):
    dx = c1[0] - c2[0]; dy = c1[1] - c2[1]
    halfL = 0.5 * L
    if dx > halfL: dx -= L
    elif dx < -halfL: dx += L
    if dy > halfL: dy -= L
    elif dy < -halfL: dy += L
    return float(np.sqrt(dx * dx + dy * dy))


def match_and_track(snapshots, L, d_match=4.0, size_ratio=2.0):
    """Match clusters between consecutive snapshots and return a
    list of patch lifetimes (in snapshot units).

    snapshots: list of (sizes, centroids) per snapshot.
    Matching rule: greedy nearest neighbour with two cutoffs:
      - periodic centroid distance < d_match
      - particle-count ratio max(s1, s2) / min(s1, s2) <= size_ratio
    """
    n_snaps = len(snapshots)
    if n_snaps < 2:
        return np.array([], dtype=int)
    # patch_id[s] = list of patch IDs (one per cluster in snapshot s)
    patch_id_per_snap = [None] * n_snaps
    next_patch_id = 0
    # snapshot 0: every cluster opens a new patch
    sizes0, cents0 = snapshots[0]
    patch_id_per_snap[0] = list(range(len(sizes0)))
    next_patch_id = len(sizes0)
    # birth/death tracking
    patch_birth = {i: 0 for i in range(len(sizes0))}
    patch_death = {i: 0 for i in range(len(sizes0))}

    for s in range(1, n_snaps):
        sizes_prev, cents_prev = snapshots[s - 1]
        sizes_cur, cents_cur = snapshots[s]
        ids_prev = patch_id_per_snap[s - 1]
        ids_cur = [-1] * len(sizes_cur)
        # Greedy matching: for each cur cluster, pick the closest prev
        # that meets criteria and is not already taken.
        taken_prev = set()
        # Build candidate pairs sorted by distance.
        candidates = []
        for ic in range(len(sizes_cur)):
            for ip in range(len(sizes_prev)):
                if ip in taken_prev:
                    continue
                d = periodic_dist(cents_cur[ic], cents_prev[ip], L)
                if d > d_match:
                    continue
                ratio = max(sizes_cur[ic], sizes_prev[ip]) \
                    / max(min(sizes_cur[ic], sizes_prev[ip]), 1)
                if ratio > size_ratio:
                    continue
                candidates.append((d, ic, ip))
        candidates.sort()
        for d, ic, ip in candidates:
            if ip in taken_prev or ids_cur[ic] != -1:
                continue
            ids_cur[ic] = ids_prev[ip]
            patch_death[ids_prev[ip]] = s
            taken_prev.add(ip)
        for ic in range(len(sizes_cur)):
            if ids_cur[ic] == -1:
                ids_cur[ic] = next_patch_id
                patch_birth[next_patch_id] = s
                patch_death[next_patch_id] = s
                next_patch_id += 1
        patch_id_per_snap[s] = ids_cur

    lifetimes = np.array(
        [patch_death[i] - patch_birth[i] + 1
         for i in range(next_patch_id)],
        dtype=int,
    )
    return lifetimes


def main() -> None:
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm = 2000
    n_snap = 200
    n_skip = 50
    seeds = list(range(11, 11 + 30, 3))
    d_match = 4.0     # ~ L/8
    size_ratio = 2.0

    modes = [
        ("motility", 0.005, 0.05, 1.0, 1.0, 0.100),
        ("full",     0.005, 0.05, 1.0, 2.0, 0.150),
    ]
    lifetimes_per_mode = {m[0]: [] for m in modes}
    n_clusters_per_snap = {m[0]: [] for m in modes}

    pbar = tqdm(total=len(modes) * len(seeds),
                desc="patch_lifetime")
    t0 = time.time()
    for label_name, vmn, vmx, amn, amx, eta in modes:
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
            snapshots = []
            n_cl_seq = []
            for k in range(n_snap):
                for _ in range(n_skip):
                    sim.step()
                _, sizes, cents = cluster_components(
                    sim.state.x, sim.state.y, L)
                snapshots.append((sizes, cents))
                n_cl_seq.append(len(sizes))
            lifetimes = match_and_track(
                snapshots, L, d_match=d_match,
                size_ratio=size_ratio,
            )
            lifetimes_per_mode[label_name].append(lifetimes)
            n_clusters_per_snap[label_name].append(np.array(n_cl_seq))
            pbar.update(1)
    pbar.close()

    out = DATA / "double_patch_lifetime.npz"
    np.savez_compressed(
        out,
        seeds=np.array(seeds),
        lifetimes_motility=np.concatenate(lifetimes_per_mode["motility"]),
        lifetimes_full=np.concatenate(lifetimes_per_mode["full"]),
        n_clusters_motility=np.array(n_clusters_per_snap["motility"]),
        n_clusters_full=np.array(n_clusters_per_snap["full"]),
        params=np.array([N, L, n_warm, n_snap, n_skip, len(seeds),
                          d_match, size_ratio], dtype=float),
    )

    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min   "
          f"saved: {out.name}")
    print()
    for m in ("motility", "full"):
        lts = np.concatenate(lifetimes_per_mode[m])
        # exclude single-snapshot patches (incipient) when reporting
        # mean lifetime, but keep them in the raw distribution.
        lt_persistent = lts[lts >= 2]
        print(f"  {m:>10s}: n_patches = {len(lts):>5d}  "
              f"<tau> all = {lts.mean():>5.2f} +/- {lts.std(ddof=1)/np.sqrt(len(lts)):.2f}   "
              f"<tau> persistent = "
              f"{lt_persistent.mean() if len(lt_persistent) else float('nan'):>5.2f}  "
              f"max = {lts.max()}")

    # Paired diff between full and motility per seed (mean lifetime).
    seed_means_full = np.array(
        [lt.mean() for lt in lifetimes_per_mode["full"]])
    seed_means_mot = np.array(
        [lt.mean() for lt in lifetimes_per_mode["motility"]])
    diff = seed_means_full - seed_means_mot
    print(f"\nPaired diff <tau>_full - <tau>_motility per seed:")
    print(f"  mean = {diff.mean():+.3f} +/- {diff.std(ddof=1)/np.sqrt(len(diff)):.3f}  "
          f"z = {diff.mean()/(diff.std(ddof=1)/np.sqrt(len(diff))):+.2f}")


if __name__ == "__main__":
    main()
