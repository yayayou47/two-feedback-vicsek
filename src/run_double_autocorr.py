"""
Polar-order autocorrelation $C(\\tau) = \\langle\\langle\\varphi
\\rangle(t+\\tau)\\langle\\varphi\\rangle(t)\\rangle / \\sigma^2_\\varphi$
for the four heavy-tailed modes at $L = 30$, ten seeds, long
trajectory ($n_{\\rm meas} = 20\\,000$). Tests the analytic
prediction of the hydrodynamic appendix $\\Gamma(\\rho, \\eta)
\\simeq \\eta^{\\alpha(\\rho)}$, namely that the full and motility
modes should differ in their decorrelation timescale by a factor
$\\eta^{\\alpha_{\\max} - \\alpha_{\\min}} = \\eta$ inside the
dense phase. We additionally compute the dense- and
dilute-quartile per-particle heading autocorrelation to make the
density stratification explicit.

Output: data/double_autocorr.npz
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
DATA.mkdir(exist_ok=True)


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


def main():
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm = 2000
    n_meas = 20000
    n_density_skip = 50      # reclassify particles every k steps
    seeds = list(range(11, 11 + 30, 3))   # 10 seeds
    taus = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
                      2000, 5000])

    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0, 0.035),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0, 0.075),
        ("v3_limit", 0.005, 0.05,  1.0, 1.0, 0.100),
        ("full",     0.005, 0.05,  1.0, 2.0, 0.150),
    ]

    n_modes = len(cases)
    n_seeds = len(seeds)
    n_taus = len(taus)
    # Global phi autocorrelation per seed
    C_phi = np.zeros((n_modes, n_seeds, n_taus))
    # Per-particle density-stratified heading autocorrelation
    C_dense = np.zeros((n_modes, n_seeds, n_taus))
    C_dilute = np.zeros((n_modes, n_seeds, n_taus))

    pbar = tqdm(total=n_modes * n_seeds, desc="autocorr")
    t0 = time.time()
    for im, (label, vmn, vmx, amn, amx, eta) in enumerate(cases):
        for isd, seed in enumerate(seeds):
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

            phi_traj = np.empty(n_meas)
            theta_traj = np.empty((N, n_meas), dtype=np.float32)
            density_traj = np.empty((N, n_meas), dtype=np.uint8)
            current_class = np.zeros(N, dtype=np.uint8)   # 1 dense, 0 dilute, 2 mid
            for k in range(n_meas):
                sim.step()
                phi_traj[k] = sim.polarisation()
                theta_traj[:, k] = sim.theta.astype(np.float32)
                if k % n_density_skip == 0:
                    rho = local_density(sim.x, sim.y, L, 1.0)
                    hi = np.percentile(rho, 75)
                    lo = np.percentile(rho, 25)
                    current_class[:] = 2
                    current_class[rho >= hi] = 1
                    current_class[rho <= lo] = 0
                density_traj[:, k] = current_class

            # Global phi autocorrelation.
            phi_centered = phi_traj - phi_traj.mean()
            var_phi = float(phi_centered.var())
            for j, tau in enumerate(taus):
                if tau >= n_meas:
                    continue
                C_phi[im, isd, j] = (
                    np.mean(phi_centered[tau:] * phi_centered[:-tau])
                    / max(var_phi, 1e-12))

            # Per-particle stratified heading autocorrelation.
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

    np.savez_compressed(
        DATA / "double_autocorr.npz",
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
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
    print()
    # Headline: dense-phase decorrelation halftime per mode.
    print("Dense-phase per-particle C(tau), mean over seeds:")
    print("  tau:     " + " ".join(f"{int(t):>5d}" for t in taus))
    for im, (lbl, *_) in enumerate(cases):
        line = " ".join(f"{C_dense[im, :, j].mean():+.3f}"
                         for j in range(n_taus))
        print(f"  {lbl:>10s}: {line}")
    print()
    # Pairwise full vs motility
    print("Gap (full - motility) on dense-phase C(tau):")
    for j, tau in enumerate(taus):
        diff = (C_dense[3, :, j] - C_dense[2, :, j])
        z = (diff.mean() / max(diff.std(ddof=1)
                                  / np.sqrt(n_seeds), 1e-9))
        print(f"  tau={int(tau):>5d}: delta = {diff.mean():+.3f} "
              f"(z = {z:+.2f})")


if __name__ == "__main__":
    main()
