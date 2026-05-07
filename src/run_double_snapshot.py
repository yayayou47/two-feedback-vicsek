"""
Real-space snapshots comparing the v3-limit (motility only) and the
full two-feedback model (motility + noise-shape) at three eta
values, $L = 30$. Goal: visualise whether the Gaussian
rectification of the heavy tails inside the dense phase produces
a sharper / more cohesive phase-separated structure than the
motility channel alone.

Output: data/double_snapshot.npz
   labels[mode, case], x[mode, case, n], y[mode, case, n],
   theta[mode, case, n], v[mode, case, n], alpha[mode, case, n],
   eta[case], phi[mode, case]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vicsek_double_adaptive import (DoubleAdaptiveParams,
                                     DoubleAdaptiveVicsek)


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def main():
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    R_r, R_a = 0.5, 0.7
    n_warm = 3000
    seed = 11

    # Eta values tuned from the pilot peaks: v3-limit and full both
    # peak around eta ~ 0.10-0.15, so we pick the same eta grid for
    # both modes to make the comparison clean.
    cases = [
        ("ordered",       0.020),
        ("near_critical", 0.100),
        ("disordered",    0.300),
    ]
    modes = [
        ("v3_limit", 0.005, 0.05, 1.0, 1.0),   # adaptive v, alpha=1 fixed
        ("full",     0.005, 0.05, 1.0, 2.0),   # both adaptive
    ]

    n_modes = len(modes)
    n_cases = len(cases)
    x_arr = np.zeros((n_modes, n_cases, N))
    y_arr = np.zeros((n_modes, n_cases, N))
    theta_arr = np.zeros((n_modes, n_cases, N))
    v_arr = np.zeros((n_modes, n_cases, N))
    alpha_arr = np.zeros((n_modes, n_cases, N))
    eta_arr = np.zeros(n_cases)
    phi_arr = np.zeros((n_modes, n_cases))
    mode_labels = np.array([m[0] for m in modes])
    case_labels = np.array([c[0] for c in cases])

    for im, (mlabel, vmn, vmx, amn, amx) in enumerate(modes):
        for ic, (clabel, eta) in enumerate(cases):
            p = DoubleAdaptiveParams(
                N=N, L=L,
                v_max=float(vmx), v_min=float(vmn),
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=R_r, R_a=R_a,
                beta=30.0, eta=float(eta),
                n_star=3.0, slope=2.0,
                seed=seed,
            )
            sim = DoubleAdaptiveVicsek(p)
            sim.theta[:] = 0.0
            for _ in range(n_warm):
                sim.step()
            x_arr[im, ic] = sim.x
            y_arr[im, ic] = sim.y
            theta_arr[im, ic] = sim.theta
            v_arr[im, ic] = sim.v_i
            alpha_arr[im, ic] = sim.alpha_i
            eta_arr[ic] = eta
            phi_arr[im, ic] = sim.polarisation()
            print(f"  {mlabel}/{clabel}: eta={eta:.3f}  "
                  f"phi={phi_arr[im, ic]:.3f}  "
                  f"<v>={v_arr[im, ic].mean():.4f}  "
                  f"<alpha>={alpha_arr[im, ic].mean():.3f}")

    np.savez_compressed(
        DATA / "double_snapshot.npz",
        mode_labels=mode_labels, case_labels=case_labels,
        x=x_arr, y=y_arr, theta=theta_arr, v=v_arr, alpha=alpha_arr,
        eta=eta_arr, phi=phi_arr,
        params=np.array([N, L, R_r, R_a, n_warm, seed], dtype=float),
    )
    print("saved:", DATA / "double_snapshot.npz")


if __name__ == "__main__":
    main()
