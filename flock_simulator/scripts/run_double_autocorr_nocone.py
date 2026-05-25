"""No-cone port of legacy run_double_autocorr.py.

Heading autocorrelation C(tau) at L=30, four modes at their
near-critical eta, 10 seeds. Dense / dilute / global versions.
Output: double_autocorr_nocone.npz.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, warm


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
        rho[i] = int(np.sum(dx * dx + dy * dy < R2)) - 1
    return rho


def main() -> None:
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm, n_meas = 2000, 20000
    n_density_skip = 50
    seeds = list(range(11, 11 + 30, 3))
    taus = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
                     2000, 5000])

    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0, 0.035),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0, 0.075),
        ("v3_limit", 0.005, 0.05,  1.0, 1.0, 0.100),
        ("full",     0.005, 0.05,  1.0, 2.0, 0.150),
    ]
    n_modes, n_seeds, n_taus = len(cases), len(seeds), len(taus)
    C_phi = np.zeros((n_modes, n_seeds, n_taus))
    C_dense = np.zeros((n_modes, n_seeds, n_taus))
    C_dilute = np.zeros((n_modes, n_seeds, n_taus))

    pbar = tqdm(total=n_modes * n_seeds, desc="autocorr_nocone")
    t0 = time.time()
    for im, (label, vmn, vmx, amn, amx, eta) in enumerate(cases):
        for isd, seed in enumerate(seeds):
            p = FlockParams(
                N=N, L=L,
                v_max=float(vmx), v_min=float(vmn),
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=0.5, R_a=0.7, eta=float(eta),
                n_star=3.0, slope=2.0, seed=int(seed),
            )
            sim = FlockSimulator(p)
            warm(sim, n_warm)

            phi_traj = np.empty(n_meas)
            theta_traj = np.empty((N, n_meas), dtype=np.float32)
            density_traj = np.empty((N, n_meas), dtype=np.uint8)
            current_class = np.zeros(N, dtype=np.uint8)
            for k in range(n_meas):
                sim.step()
                phi_traj[k] = sim.polarisation()
                theta_traj[:, k] = sim.state.theta.astype(np.float32)
                if k % n_density_skip == 0:
                    rho = local_density(sim.state.x, sim.state.y, L, 1.0)
                    hi = np.percentile(rho, 75)
                    lo = np.percentile(rho, 25)
                    current_class[:] = 2
                    current_class[rho >= hi] = 1
                    current_class[rho <= lo] = 0
                density_traj[:, k] = current_class

            phi_centered = phi_traj - phi_traj.mean()
            var_phi = float(phi_centered.var())
            for j, tau in enumerate(taus):
                if tau >= n_meas:
                    continue
                C_phi[im, isd, j] = (
                    np.mean(phi_centered[tau:] * phi_centered[:-tau])
                    / max(var_phi, 1e-12))

            for j, tau in enumerate(taus):
                if tau >= n_meas:
                    continue
                dtheta = (theta_traj[:, tau:].astype(np.float64)
                          - theta_traj[:, :-tau].astype(np.float64))
                cos_d = np.cos(dtheta)
                cls_t = density_traj[:, :-tau]
                dense_mask = (cls_t == 1)
                dilute_mask = (cls_t == 0)
                if dense_mask.any():
                    C_dense[im, isd, j] = cos_d[dense_mask].mean()
                if dilute_mask.any():
                    C_dilute[im, isd, j] = cos_d[dilute_mask].mean()
            del theta_traj, density_traj, dtheta, cos_d
            pbar.update(1)
    pbar.close()

    out = DATA / "double_autocorr_nocone.npz"
    np.savez_compressed(
        out,
        labels=np.array([c[0] for c in cases]),
        eta_per_mode=np.array([c[5] for c in cases]),
        seeds=np.array(seeds),
        taus=taus,
        C_phi_per_seed=C_phi,
        C_dense_per_seed=C_dense,
        C_dilute_per_seed=C_dilute,
        params=np.array([N, L, n_warm, n_meas, n_density_skip,
                         n_seeds], dtype=float),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")


if __name__ == "__main__":
    main()
