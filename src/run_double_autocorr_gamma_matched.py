r"""B1 referee test (dynamic side): dense $C(\tau)$ at MATCHED $\Gamma$.

Companion to run_double_gr_gamma_matched.py. Re-measures the dense-quartile
heading autocorrelation $C_{\rm dense}(\tau)$ for the full mode at its
near-critical $\eta = 0.15$ (dense $\alpha \to 2$) against motility-only at
its own $\eta = 0.10$ and at the matched $\eta = 0.0225$ (so the $\alpha = 1$
dense kick decorrelates at the same rate $\Gamma = 1 - e^{-\eta^\alpha}$ as
the full mode's $\alpha = 2$ kick). Same L=30, rho0=2.22, ten-seed,
n_meas=20000 protocol as run_double_autocorr.py.

Output: data/double_autocorr_gamma_matched_nocone.npz
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
        rho[i] = int(np.sum(dx * dx + dy * dy < R2)) - 1
    return rho


def main():
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm = 2000
    n_meas = 20000
    n_density_skip = 50
    seeds = list(range(11, 11 + 30, 3))   # 10 seeds
    taus = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])

    cases = [
        ("full",       0.005, 0.05, 1.0, 2.0, 0.150),
        ("v3_own",     0.005, 0.05, 1.0, 1.0, 0.100),
        ("v3_matched", 0.005, 0.05, 1.0, 1.0, 0.0225),
    ]
    n_modes, n_seeds, n_taus = len(cases), len(seeds), len(taus)
    C_dense = np.zeros((n_modes, n_seeds, n_taus))
    C_dilute = np.zeros((n_modes, n_seeds, n_taus))

    pbar = tqdm(total=n_modes * n_seeds, desc="autocorr_gamma")
    t0 = time.time()
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
            theta_traj = np.empty((N, n_meas), dtype=np.float32)
            density_traj = np.empty((N, n_meas), dtype=np.uint8)
            current_class = np.zeros(N, dtype=np.uint8)
            for k in range(n_meas):
                sim.step()
                theta_traj[:, k] = sim.theta.astype(np.float32)
                if k % n_density_skip == 0:
                    rho = local_density(sim.x, sim.y, L, 1.0)
                    current_class[:] = 2
                    current_class[rho >= np.percentile(rho, 75)] = 1
                    current_class[rho <= np.percentile(rho, 25)] = 0
                density_traj[:, k] = current_class
            for j, tau in enumerate(taus):
                if tau >= n_meas:
                    continue
                dtheta = (theta_traj[:, tau:].astype(np.float64)
                          - theta_traj[:, :-tau].astype(np.float64))
                cos_d = np.cos(dtheta)
                cls_t = density_traj[:, :-tau]
                if (cls_t == 1).any():
                    C_dense[im, isd, j] = cos_d[cls_t == 1].mean()
                if (cls_t == 0).any():
                    C_dilute[im, isd, j] = cos_d[cls_t == 0].mean()
            del theta_traj, density_traj, dtheta, cos_d
            pbar.update(1)
    pbar.close()

    np.savez_compressed(
        DATA / "double_autocorr_gamma_matched_nocone.npz",
        labels=np.array([c[0] for c in cases]),
        eta_per_mode=np.array([c[5] for c in cases]),
        seeds=np.array(seeds), taus=taus,
        C_dense_per_seed=C_dense, C_dilute_per_seed=C_dilute,
        params=np.array([N, L, n_warm, n_meas, n_density_skip, n_seeds],
                        float))
    print(f"runtime {(time.time()-t0)/60:.1f} min -> "
          "double_autocorr_gamma_matched_nocone.npz")


if __name__ == "__main__":
    main()
