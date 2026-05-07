"""
Heading-correlation function $g(r) = \langle \cos[\theta_i -
\theta_j] \rangle_{|r_{ij}| = r}$ for the four heavy-tailed-noise
modes at the near-critical eta, stratified by local density. The
expectation from the rectification mechanism: the noise-shape
adaptation should narrow $g(r)$ inside the dense phase compared
with the motility-only ablation, while leaving the dilute-phase
correlation essentially unchanged.

Output: data/double_gr.npz
   labels, r_centers, gr_dense[mode, r], gr_dilute[mode, r]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from vicsek_double_adaptive import (DoubleAdaptiveParams,
                                     DoubleAdaptiveVicsek)


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def local_density(x, y, L, R=1.0):
    """Count neighbours within R for each particle (no periodic
    correction beyond the half-box wrap; OK for R << L/2)."""
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
        rho[i] = int(np.sum(d2 < R2)) - 1   # exclude self
    return rho


def gr_for_subset(x, y, theta, L, idx, r_edges):
    """Compute g(r) restricted to particles in idx (any j allowed
    as the partner). Returns (counts, sum_cos)."""
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
        # Bin by distance; exclude self (d=0).
        mask = (d > 0.0) & (d < r_edges[-1])
        bin_idx = np.searchsorted(r_edges, d[mask]) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        np.add.at(counts, bin_idx, 1)
        np.add.at(sumc, bin_idx, cos_dt[mask])
    return counts, sumc


def main():
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm = 2000
    n_snap = 6
    n_skip = 200
    seeds = [11, 23, 41]
    R_local = 1.0
    # Start from r = R_r so we exclude the trivial repulsion zone;
    # finer bin spacing in the alignment range so dense-phase decay
    # is resolved.
    r_edges = np.linspace(0.5, 6.0, 24)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    n_bins = len(r_centers)

    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0, 0.035),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0, 0.075),
        ("v3_limit", 0.005, 0.05,  1.0, 1.0, 0.100),
        ("full",     0.005, 0.05,  1.0, 2.0, 0.150),
    ]

    n_modes = len(cases)
    counts_d = np.zeros((n_modes, n_bins), dtype=np.int64)
    sumc_d = np.zeros((n_modes, n_bins))
    counts_l = np.zeros((n_modes, n_bins), dtype=np.int64)
    sumc_l = np.zeros((n_modes, n_bins))

    pbar = tqdm(total=n_modes * len(seeds) * n_snap, desc="g(r)")
    for im, (label_name, vmn, vmx, amn, amx, eta) in enumerate(cases):
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
            for snap in range(n_snap):
                for _ in range(n_skip):
                    sim.step()
                rho = local_density(sim.x, sim.y, L, R_local)
                hi = np.percentile(rho, 75)
                lo = np.percentile(rho, 25)
                idx_dense = np.where(rho >= hi)[0]
                idx_dilute = np.where(rho <= lo)[0]
                cD, sD = gr_for_subset(sim.x, sim.y, sim.theta, L,
                                        idx_dense, r_edges)
                cL, sL = gr_for_subset(sim.x, sim.y, sim.theta, L,
                                        idx_dilute, r_edges)
                counts_d[im] += cD
                sumc_d[im] += sD
                counts_l[im] += cL
                sumc_l[im] += sL
                pbar.update(1)
    pbar.close()

    gr_dense = np.where(counts_d > 0, sumc_d / counts_d, np.nan)
    gr_dilute = np.where(counts_l > 0, sumc_l / counts_l, np.nan)

    np.savez_compressed(
        DATA / "double_gr.npz",
        labels=np.array([c[0] for c in cases]),
        eta_per_mode=np.array([c[5] for c in cases]),
        r_centers=r_centers,
        gr_dense=gr_dense, gr_dilute=gr_dilute,
        counts_dense=counts_d, counts_dilute=counts_l,
    )

    print()
    print("g(r) head: dense / dilute, first 5 bins")
    for im, (lbl, *_) in enumerate(cases):
        gd = gr_dense[im, :5]
        gl = gr_dilute[im, :5]
        print(f"  {lbl:10s}  dense  {' '.join(f'{v:+.3f}' for v in gd)}")
        print(f"  {lbl:10s}  dilute {' '.join(f'{v:+.3f}' for v in gl)}")
    print(f"saved: {DATA / 'double_gr.npz'}")


if __name__ == "__main__":
    main()
