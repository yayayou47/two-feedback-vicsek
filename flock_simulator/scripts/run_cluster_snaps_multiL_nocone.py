"""Near-critical real-space snapshots at several sizes for the
cluster-identification figure (Fig 12).

Produces motility-only and double-adaptive configurations at the
near-critical eta for L in {30, 90, 128} (single seed, long
warm-up), so the cluster map can show how the dense-droplet
morphology persists across system size. Output:
data/double_cluster_snaps_multiL_nocone.npz with per (mode, L)
particle positions / headings, same fields the cluster-map
renderer expects (mode_labels, L_list, x, y, theta, eta, phi).
"""
from __future__ import annotations

import time
import numpy as np

from _helpers import DATA, FlockParams, FlockSimulator, warm


def main() -> None:
    sigma = 2.22
    eta = 0.10            # near-critical
    seed = 11
    n_warm = 3000
    Ls = [30.0, 90.0, 128.0]
    modes = [
        ("v3_limit", 0.005, 0.05, 1.0, 1.0),
        ("full",     0.005, 0.05, 1.0, 2.0),
    ]
    n_modes, n_L = len(modes), len(Ls)
    Ns = [int(round(sigma * L * L)) for L in Ls]
    Nmax = max(Ns)

    # Ragged across L -> pad to Nmax with NaN, store true counts.
    x = np.full((n_modes, n_L, Nmax), np.nan)
    y = np.full((n_modes, n_L, Nmax), np.nan)
    theta = np.full((n_modes, n_L, Nmax), np.nan)
    v = np.full((n_modes, n_L, Nmax), np.nan)
    phi = np.zeros((n_modes, n_L))
    counts = np.zeros((n_modes, n_L), dtype=int)

    t0 = time.time()
    for im, (mlabel, vmn, vmx, amn, amx) in enumerate(modes):
        for iL, L in enumerate(Ls):
            N = Ns[iL]
            p = FlockParams(
                N=N, L=float(L),
                v_max=float(vmx), v_min=float(vmn),
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=0.5, R_a=0.7, eta=float(eta),
                n_star=3.0, slope=2.0, seed=seed,
            )
            sim = FlockSimulator(p)
            warm(sim, n_warm)
            x[im, iL, :N] = sim.state.x
            y[im, iL, :N] = sim.state.y
            theta[im, iL, :N] = sim.state.theta
            v[im, iL, :N] = sim.state.v_i
            phi[im, iL] = sim.polarisation()
            counts[im, iL] = N
            print(f"  {mlabel}/L={L:g}: N={N} phi={phi[im, iL]:.3f}")

    out = DATA / "double_cluster_snaps_multiL_nocone.npz"
    np.savez_compressed(
        out,
        mode_labels=np.array([m[0] for m in modes]),
        L_list=np.array(Ls),
        counts=counts,
        x=x, y=y, theta=theta, v=v, eta=float(eta), phi=phi,
        params=np.array([sigma, 0.5, 0.7, n_warm, seed], dtype=float),
    )
    print(f"\nruntime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")


if __name__ == "__main__":
    main()
