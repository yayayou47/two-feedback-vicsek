"""Real-space snapshots across sizes for the extended snapshot grid.

Produces motility-only (v3_limit) and double-adaptive (full)
configurations at three noise levels (ordered / near-critical /
disordered) for L in {30, 90, 128}, single seed, long warm-up.
Output: data/double_snapshot_multiL_nocone.npz with arrays shaped
(n_L, n_modes, n_cases, Nmax) padded with NaN, plus per-(L,mode,
case) counts and polar order.
"""
from __future__ import annotations

import time
import numpy as np

from _helpers import DATA, FlockParams, FlockSimulator, warm


def main() -> None:
    sigma = 2.22
    seed = 11
    n_warm = 3000
    Ls = [30.0, 90.0, 128.0]
    cases = [
        ("ordered",       0.020),
        ("near_critical", 0.100),
        ("disordered",    0.300),
    ]
    modes = [
        ("v3_limit", 0.005, 0.05, 1.0, 1.0),
        ("full",     0.005, 0.05, 1.0, 2.0),
    ]
    n_L, n_modes, n_cases = len(Ls), len(modes), len(cases)
    Ns = [int(round(sigma * L * L)) for L in Ls]
    Nmax = max(Ns)

    x = np.full((n_L, n_modes, n_cases, Nmax), np.nan)
    y = np.full((n_L, n_modes, n_cases, Nmax), np.nan)
    theta = np.full((n_L, n_modes, n_cases, Nmax), np.nan)
    v = np.full((n_L, n_modes, n_cases, Nmax), np.nan)
    phi = np.zeros((n_L, n_modes, n_cases))
    counts = np.zeros((n_L, n_modes, n_cases), dtype=int)
    eta_arr = np.array([c[1] for c in cases])

    t0 = time.time()
    for iL, L in enumerate(Ls):
        N = Ns[iL]
        for im, (mlabel, vmn, vmx, amn, amx) in enumerate(modes):
            for ic, (clabel, eta) in enumerate(cases):
                p = FlockParams(
                    N=N, L=float(L),
                    v_max=float(vmx), v_min=float(vmn),
                    alpha_min=float(amn), alpha_max=float(amx),
                    R_r=0.5, R_a=0.7, eta=float(eta),
                    n_star=3.0, slope=2.0, seed=seed,
                )
                sim = FlockSimulator(p)
                warm(sim, n_warm)
                x[iL, im, ic, :N] = sim.state.x
                y[iL, im, ic, :N] = sim.state.y
                theta[iL, im, ic, :N] = sim.state.theta
                v[iL, im, ic, :N] = sim.state.v_i
                phi[iL, im, ic] = sim.polarisation()
                counts[iL, im, ic] = N
                print(f"  L={L:g}/{mlabel}/{clabel}: N={N} "
                      f"phi={phi[iL, im, ic]:.3f}")

    out = DATA / "double_snapshot_multiL_nocone.npz"
    np.savez_compressed(
        out,
        L_list=np.array(Ls),
        mode_labels=np.array([m[0] for m in modes]),
        case_labels=np.array([c[0] for c in cases]),
        eta=eta_arr, counts=counts,
        x=x, y=y, theta=theta, v=v, phi=phi,
        params=np.array([sigma, 0.5, 0.7, n_warm, seed], dtype=float),
    )
    print(f"\nruntime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")


if __name__ == "__main__":
    main()
