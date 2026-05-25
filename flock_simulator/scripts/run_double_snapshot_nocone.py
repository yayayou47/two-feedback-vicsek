"""No-cone port of legacy run_double_snapshot.py.

Real-space snapshots of v3-limit and full at three eta values
(L=30, single seed), for Fig 3 of the manuscript.
"""
from __future__ import annotations

import numpy as np

from _helpers import DATA, FlockParams, FlockSimulator, warm


def main() -> None:
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm = 3000
    seed = 11

    cases = [
        ("ordered",       0.020),
        ("near_critical", 0.100),
        ("disordered",    0.300),
    ]
    modes = [
        ("v3_limit", 0.005, 0.05, 1.0, 1.0),
        ("full",     0.005, 0.05, 1.0, 2.0),
    ]
    n_modes, n_cases = len(modes), len(cases)
    x_arr = np.zeros((n_modes, n_cases, N))
    y_arr = np.zeros((n_modes, n_cases, N))
    theta_arr = np.zeros((n_modes, n_cases, N))
    v_arr = np.zeros((n_modes, n_cases, N))
    alpha_arr = np.zeros((n_modes, n_cases, N))
    eta_arr = np.zeros(n_cases)
    phi_arr = np.zeros((n_modes, n_cases))

    for im, (mlabel, vmn, vmx, amn, amx) in enumerate(modes):
        for ic, (clabel, eta) in enumerate(cases):
            p = FlockParams(
                N=N, L=L,
                v_max=float(vmx), v_min=float(vmn),
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=0.5, R_a=0.7, eta=float(eta),
                n_star=3.0, slope=2.0, seed=seed,
            )
            sim = FlockSimulator(p)
            warm(sim, n_warm)
            x_arr[im, ic] = sim.state.x
            y_arr[im, ic] = sim.state.y
            theta_arr[im, ic] = sim.state.theta
            v_arr[im, ic] = sim.state.v_i
            alpha_arr[im, ic] = sim.state.alpha_i
            eta_arr[ic] = eta
            phi_arr[im, ic] = sim.polarisation()
            print(f"  {mlabel}/{clabel}: eta={eta:.3f}  "
                  f"phi={phi_arr[im, ic]:.3f}  "
                  f"<v>={v_arr[im, ic].mean():.4f}  "
                  f"<alpha>={alpha_arr[im, ic].mean():.3f}")

    out = DATA / "double_snapshot_nocone.npz"
    np.savez_compressed(
        out,
        mode_labels=np.array([m[0] for m in modes]),
        case_labels=np.array([c[0] for c in cases]),
        x=x_arr, y=y_arr, theta=theta_arr, v=v_arr, alpha=alpha_arr,
        eta=eta_arr, phi=phi_arr,
        params=np.array([N, L, 0.5, 0.7, n_warm, seed], dtype=float),
    )
    print(f"saved: {out.name}")


if __name__ == "__main__":
    main()
