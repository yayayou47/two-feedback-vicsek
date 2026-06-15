r"""B1 referee test: dense-phase $g(r)$ at MATCHED effective decorrelation.

The published dense heading-correlation gap compares the full mode at its
near-critical $\eta = 0.15$ (dense $\alpha \to 2$) against motility-only at
$\eta = 0.10$ (dense $\alpha = 1$). Those two dense phases do not decorrelate
at the same rate: with $\Gamma(\eta,\alpha) = 1 - e^{-\eta^\alpha}$ (App. A),
the full dense kick has $\Gamma = 1 - e^{-0.15^2} = 0.022$ while the
motility-only dense kick has $\Gamma = 1 - e^{-0.10} = 0.095$, a factor
~4 apart. A referee (Reviewer 2) asks whether the gap survives once the two
dense phases are compared at matched $\Gamma$.

We retune motility-only to $\eta_{\rm mot}$ such that its $\alpha = 1$ dense
kick decorrelates as fast as the full mode's $\alpha = 2$ kick at
$\eta = 0.15$: $1 - e^{-\eta_{\rm mot}} = 0.022 \Rightarrow
\eta_{\rm mot} = 0.0225$. We then re-measure the dense $g(r)$ for
  (i)  full at $\eta = 0.15$ (alpha 1->2),
  (ii) motility-only at its own $\eta = 0.10$ (reproduces the published gap),
  (iii) motility-only at the matched $\eta = 0.0225$,
on the same L=30, rho0=2.22, ten-seed protocol as run_double_gr_hs.py.

Output: data/double_gr_gamma_matched_nocone.npz
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
        rho[i] = int(np.count_nonzero(dx * dx + dy * dy < R2)) - 1
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
        bin_idx = np.clip(np.searchsorted(r_edges, d[mask]) - 1, 0, n_bins - 1)
        np.add.at(counts, bin_idx, 1)
        np.add.at(sumc, bin_idx, cos_dt[mask])
    return counts, sumc


def main():
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm, n_snap, n_skip = 2000, 6, 200
    seeds = list(range(11, 11 + 30, 3))           # 10 seeds
    R_local = 1.0
    r_edges = np.linspace(0.5, 6.0, 24)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_bins = len(r_centers)

    # (label, v_min, v_max, alpha_min, alpha_max, eta)
    cases = [
        ("full",        0.005, 0.05, 1.0, 2.0, 0.150),
        ("v3_own",      0.005, 0.05, 1.0, 1.0, 0.100),
        ("v3_matched",  0.005, 0.05, 1.0, 1.0, 0.0225),
    ]
    n_modes, n_seeds = len(cases), len(seeds)
    grD = np.zeros((n_modes, n_seeds, n_bins))
    grL = np.zeros((n_modes, n_seeds, n_bins))

    t0 = time.time()
    pbar = tqdm(total=n_modes * n_seeds * n_snap, desc="gr_gamma")
    for im, (name, vmn, vmx, amn, amx, eta) in enumerate(cases):
        for isd, seed in enumerate(seeds):
            p = DoubleAdaptiveParams(
                N=N, L=L, v_max=float(vmx), v_min=float(vmn),
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=0.5, R_a=0.7, beta=30.0, eta=float(eta),
                n_star=3.0, slope=2.0, seed=int(seed))
            sim = DoubleAdaptiveVicsek(p)
            sim.theta[:] = 0.0
            for _ in range(n_warm):
                sim.step()
            cD = np.zeros(n_bins, np.int64); sD = np.zeros(n_bins)
            cL = np.zeros(n_bins, np.int64); sL = np.zeros(n_bins)
            for _ in range(n_snap):
                for _ in range(n_skip):
                    sim.step()
                rho = local_density(sim.x, sim.y, L, R_local)
                idx_d = np.where(rho >= np.percentile(rho, 75))[0]
                idx_l = np.where(rho <= np.percentile(rho, 25))[0]
                a, b = gr_for_subset(sim.x, sim.y, sim.theta, L, idx_d, r_edges)
                cD += a; sD += b
                a, b = gr_for_subset(sim.x, sim.y, sim.theta, L, idx_l, r_edges)
                cL += a; sL += b
                pbar.update(1)
            grD[im, isd] = np.where(cD > 0, sD / cD, np.nan)
            grL[im, isd] = np.where(cL > 0, sL / cL, np.nan)
    pbar.close()

    np.savez_compressed(
        DATA / "double_gr_gamma_matched_nocone.npz",
        labels=np.array([c[0] for c in cases]),
        etas=np.array([c[5] for c in cases]),
        seeds=np.array(seeds), r_centers=r_centers,
        gr_dense_per_seed=grD, gr_dilute_per_seed=grL,
        params=np.array([N, L, n_warm, n_snap, n_skip, n_seeds], float))
    print(f"runtime {(time.time()-t0)/60:.1f} min -> "
          "double_gr_gamma_matched_nocone.npz")


if __name__ == "__main__":
    main()
