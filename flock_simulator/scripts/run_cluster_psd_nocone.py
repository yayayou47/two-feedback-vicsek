"""Cluster-size distribution P(s) of the dense phase.

The summary-statistic cluster run (run_double_clusters_hs_nocone.py)
stores only the per-seed count / max / mean. To tell a spinodal
regime (power-law tail, 2D exponent near -2) from a binodal one
(one giant cluster coexisting with small ones) we need the full
distribution P(s). At sigma = 2.22 the bare contact graph (edge
for d_ij < R_a) percolates -- s_max/N ~ 0.9 in every mode -- so it
cannot discriminate. We therefore measure the size distribution of
the *dense-phase* clusters, defined exactly as in Fig. 11: a 10x10
grid, each cell thresholded at 1.5x the median occupancy,
periodic-aware connected-component labelling of the dense mask, the
size being the particle count summed over the connected dense bins.

Four modes at their near-critical eta, L = 30, 10 seeds, many
snapshots. Output: data/double_cluster_psd_nocone.npz with the
geomspace-binned histogram per mode, the giant-cluster fraction
s_max/N per snapshot, and a power-law tail-exponent fit.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, warm


def cluster_sizes_dense(x, y, L, n_bin=10, factor=1.5):
    """Particle-count sizes of connected dense-bin clusters.

    Identical dense-phase definition to Fig. 11 / the cluster
    summary run: histogram on an n_bin x n_bin grid, threshold at
    ``factor`` times the median non-empty occupancy, label the
    periodic-tiled dense mask, and sum the occupancy over each
    connected component (dividing the 3x3 tiling overcount by 9)."""
    H, _, _ = np.histogram2d(
        x % L, y % L, bins=[n_bin, n_bin], range=[[0, L], [0, L]])
    nz = H[H > 0]
    if len(nz) == 0:
        return np.array([], dtype=int)
    threshold = factor * np.median(nz)
    mask = H > threshold
    if not mask.any():
        return np.array([], dtype=int)
    tiled = np.tile(mask, (3, 3))
    labelled, _ = label(tiled)
    H_tiled = np.tile(H, (3, 3))
    central = labelled[n_bin:2 * n_bin, n_bin:2 * n_bin]
    sizes = []
    for cid in np.unique(central[central > 0]):
        sizes.append(int(round(H_tiled[labelled == cid].sum() / 9)))
    return np.array(sizes, dtype=int)


def main() -> None:
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    R_a = 0.7
    n_warm, n_meas, n_skip = 2000, 4000, 50
    seeds = list(range(11, 11 + 30, 3))

    # geomspace bins from 1 to N, shared across modes
    bins = np.geomspace(1.0, N + 1.0, 31)
    centers = np.sqrt(bins[:-1] * bins[1:])

    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0, 0.035),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0, 0.075),
        ("v3_limit", 0.005, 0.05,  1.0, 1.0, 0.100),
        ("full",     0.005, 0.05,  1.0, 2.0, 0.150),
    ]
    n_modes, n_seeds = len(cases), len(seeds)
    hist = np.zeros((n_modes, len(centers)))
    smax_frac = np.zeros((n_modes, n_seeds))

    pbar = tqdm(total=n_modes * n_seeds, desc="cluster_psd_nocone")
    t0 = time.time()
    for im, (name, vmn, vmx, amn, amx, eta) in enumerate(cases):
        for isd, seed in enumerate(seeds):
            p = FlockParams(
                N=N, L=L,
                v_max=float(vmx), v_min=float(vmn),
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=0.5, R_a=R_a, eta=float(eta),
                n_star=3.0, slope=2.0, seed=int(seed),
            )
            sim = FlockSimulator(p)
            warm(sim, n_warm)
            smax_seed = []
            for k in range(n_meas):
                sim.step()
                if k % n_skip == 0:
                    s = cluster_sizes_dense(sim.state.x, sim.state.y, L)
                    if len(s) == 0:
                        continue
                    h, _ = np.histogram(s, bins=bins)
                    hist[im] += h
                    smax_seed.append(s.max() / N)
            smax_frac[im, isd] = (float(np.mean(smax_seed))
                                  if smax_seed else 0.0)
            pbar.update(1)
    pbar.close()

    # Normalise each mode's histogram to a density over log-bins and
    # fit a power-law tail P(s) ~ s^{-tau} over the decade s in [5, 200].
    tau = np.zeros(n_modes)
    widths = np.diff(bins)
    pdf = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1) / widths
    fit_mask = (centers >= 5.0) & (centers <= 200.0)
    for im in range(n_modes):
        y = pdf[im, fit_mask]
        xx = centers[fit_mask]
        ok = y > 0
        if ok.sum() >= 3:
            slope, _ = np.polyfit(np.log(xx[ok]), np.log(y[ok]), 1)
            tau[im] = -slope
        else:
            tau[im] = np.nan

    out = DATA / "double_cluster_psd_nocone.npz"
    np.savez_compressed(
        out,
        labels=np.array([c[0] for c in cases]),
        etas=np.array([c[5] for c in cases]),
        seeds=np.array(seeds),
        bins=bins, centers=centers,
        hist=hist, pdf=pdf, smax_frac=smax_frac, tau=tau,
        params=np.array([N, L, n_warm, n_meas, n_skip, n_seeds],
                         dtype=float),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")
    print(f"\n{'mode':>10s}  {'tail tau':>10s}  {'<s_max/N>':>12s}")
    for im, c in enumerate(cases):
        sf = smax_frac[im].mean()
        sf_se = smax_frac[im].std(ddof=1) / np.sqrt(n_seeds)
        print(f"  {c[0]:>10s}  {tau[im]:8.2f}    {sf:6.3f}+-{sf_se:.3f}")


if __name__ == "__main__":
    main()
