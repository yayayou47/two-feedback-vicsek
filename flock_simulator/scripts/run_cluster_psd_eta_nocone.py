"""Cluster-size distribution P(s) of the dense phase across noise.

Companion to run_cluster_psd_nocone.py, which measures P(s) at one
near-critical eta per mode. The two fixed-speed references (Cauchy
baseline, noise-shape only) carry no motility channel and therefore
no MIPS, so at their near-critical eta they have essentially no
dense phase and the tail exponent tau is undefined. To complete the
cluster analysis of *every* mode we sweep eta across the
order--disorder window: a fixed-speed flock structures its density
through the Vicsek band instability, strongest at low eta, so any
dense-phase tail these modes can form appears here.

Same dense-phase definition as Fig. 11 / run_cluster_psd_nocone.py:
a 10x10 grid, each cell thresholded at 1.5x the median occupancy,
periodic-aware connected-component labelling of the dense mask, the
size being the particle count summed over the connected dense bins.

Four modes x four eta in {0.02, 0.05, 0.10, 0.20}, L = 30, 10 seeds,
many snapshots. Output: data/double_cluster_psd_eta_nocone.npz with
the geomspace-binned histogram per (mode, eta), the giant-cluster
fraction s_max/N per seed, and a power-law tail-exponent fit.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, warm
from run_cluster_psd_nocone import cluster_sizes_dense


def main() -> None:
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    R_a = 0.7
    n_warm, n_meas, n_skip = 2000, 4000, 50
    seeds = list(range(11, 11 + 30, 3))
    etas = [0.02, 0.05, 0.10, 0.20]

    # geomspace bins from 1 to N, shared across modes and eta
    bins = np.geomspace(1.0, N + 1.0, 31)
    centers = np.sqrt(bins[:-1] * bins[1:])

    # (name, v_min, v_max, alpha_min, alpha_max) -- eta is swept
    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0),
        ("v3_limit", 0.005, 0.05,  1.0, 1.0),
        ("full",     0.005, 0.05,  1.0, 2.0),
    ]
    n_modes, n_eta, n_seeds = len(cases), len(etas), len(seeds)
    hist = np.zeros((n_modes, n_eta, len(centers)))
    smax_frac = np.zeros((n_modes, n_eta, n_seeds))

    pbar = tqdm(total=n_modes * n_eta * n_seeds,
                desc="cluster_psd_eta_nocone")
    t0 = time.time()
    for im, (name, vmn, vmx, amn, amx) in enumerate(cases):
        for ie, eta in enumerate(etas):
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
                        s = cluster_sizes_dense(sim.state.x,
                                                sim.state.y, L)
                        if len(s) == 0:
                            continue
                        h, _ = np.histogram(s, bins=bins)
                        hist[im, ie] += h
                        smax_seed.append(s.max() / N)
                smax_frac[im, ie, isd] = (float(np.mean(smax_seed))
                                          if smax_seed else 0.0)
                pbar.update(1)
    pbar.close()

    # Normalise each (mode, eta) histogram to a density over log-bins
    # and fit a power-law tail P(s) ~ s^{-tau} over s in [5, 200].
    tau = np.full((n_modes, n_eta), np.nan)
    widths = np.diff(bins)
    totals = np.maximum(hist.sum(axis=2, keepdims=True), 1)
    pdf = hist / totals / widths
    fit_mask = (centers >= 5.0) & (centers <= 200.0)
    for im in range(n_modes):
        for ie in range(n_eta):
            y = pdf[im, ie, fit_mask]
            xx = centers[fit_mask]
            ok = y > 0
            if ok.sum() >= 3:
                slope, _ = np.polyfit(np.log(xx[ok]), np.log(y[ok]), 1)
                tau[im, ie] = -slope

    out = DATA / "double_cluster_psd_eta_nocone.npz"
    np.savez_compressed(
        out,
        labels=np.array([c[0] for c in cases]),
        etas=np.array(etas),
        seeds=np.array(seeds),
        bins=bins, centers=centers,
        hist=hist, pdf=pdf, smax_frac=smax_frac, tau=tau,
        params=np.array([N, L, n_warm, n_meas, n_skip, n_seeds],
                        dtype=float),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")
    header = "  ".join(f"eta={e:g}" for e in etas)
    print(f"\n{'mode':>10s}  tail tau over eta:  {header}")
    for im, c in enumerate(cases):
        row = "  ".join(f"{tau[im, ie]:7.2f}" for ie in range(n_eta))
        print(f"  {c[0]:>10s}  {row}")
    print(f"\n{'mode':>10s}  <s_max/N> over eta")
    for im, c in enumerate(cases):
        row = "  ".join(f"{smax_frac[im, ie].mean():6.3f}"
                        for ie in range(n_eta))
        print(f"  {c[0]:>10s}  {row}")


if __name__ == "__main__":
    main()
