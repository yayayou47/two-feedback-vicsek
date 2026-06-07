"""Single-particle trajectory statistics, density-stratified.

For the four study modes at their near-critical eta (L = 30,
rho0 = 2.22), record per-particle dynamics over n_meas steps and
build two diagnostics, each split by the local-density quartile the
particle sits in (dense = top quartile, dilute = bottom quartile,
recomputed every n_density_skip steps and held constant in between,
exactly as in run_double_autocorr_nocone.py):

  1. Turning-angle pdf p(dtheta): the per-step heading increment
     dtheta_i(t) = wrap(theta_i(t+1) - theta_i(t)), accumulated into a
     histogram on (-pi, pi]. The alpha-stable gate predicts a heavy
     (Cauchy-like, alpha -> 1) tail in the dilute phase and a narrow
     (Gaussian, alpha -> 2) core in the dense phase.

  2. Mean squared displacement MSD(tau) = <|r(t0+tau) - r(t0)|^2>,
     conditioned on the density class at the origin t0, with positions
     unwrapped across the periodic box. Superdiffusion in the dilute
     phase vs caging/normal diffusion in the dense phase.

Output: data/double_traj_stats_nocone.npz.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, warm


def local_density(x, y, L, R=1.0):
    """Number of neighbours within R (periodic), per particle."""
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
    rho0 = 2.22
    N = int(round(rho0 * L * L))
    n_warm, n_meas = 2000, 8000
    n_density_skip = 50
    n_pos_skip = 4                       # record positions every 4 steps
    seeds = list(range(11, 11 + 18, 3))  # 6 seeds: 11,14,17,20,23,26

    # Same four modes / near-critical etas as the autocorrelation run.
    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0, 0.035),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0, 0.075),
        ("v3_limit", 0.005, 0.05,  1.0, 1.0, 0.100),
        ("full",     0.005, 0.05,  1.0, 2.0, 0.150),
    ]
    n_modes, n_seeds = len(cases), len(seeds)

    # Turning-angle histogram on (-pi, pi], per mode/seed/class.
    n_bins = 120
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # classes: 0 = dilute (bottom quartile), 1 = dense (top quartile)
    turn_hist = np.zeros((n_modes, n_seeds, 2, n_bins), dtype=np.float64)

    # MSD: log-spaced lags in *frame* units (1 frame = n_pos_skip steps).
    n_frames = n_meas // n_pos_skip
    lags_f = np.unique(np.round(
        np.logspace(0, np.log10(n_frames // 4), 14)).astype(int))
    lags_f = lags_f[lags_f >= 1]
    n_lags = len(lags_f)
    lags_steps = lags_f * n_pos_skip
    origin_skip = 20                     # sample time origins sparsely
    msd_sum = np.zeros((n_modes, n_seeds, 2, n_lags))
    msd_cnt = np.zeros((n_modes, n_seeds, 2, n_lags))

    phi_mean = np.zeros((n_modes, n_seeds))

    pbar = tqdm(total=n_modes * n_seeds, desc="traj_stats")
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

            # Unwrapped position tracker + recorded frames.
            unwrap = np.column_stack([sim.state.x.copy(),
                                      sim.state.y.copy()])
            prev_xy = unwrap.copy()
            prev_th = sim.state.theta.copy()
            current_class = np.full(N, 2, dtype=np.uint8)  # 2 = mid

            pos_frames = np.empty((N, n_frames, 2), dtype=np.float32)
            cls_frames = np.empty((N, n_frames), dtype=np.uint8)
            fi = 0
            phi_acc = 0.0

            for k in range(n_meas):
                sim.step()
                phi_acc += sim.polarisation()
                x = sim.state.x; y = sim.state.y
                th = sim.state.theta

                # Unwrap positions (minimal-image increment).
                dxy = np.column_stack([x, y]) - prev_xy
                dxy -= L * np.round(dxy / L)
                unwrap += dxy
                prev_xy = np.column_stack([x, y])

                # Refresh the held density class every n_density_skip.
                if k % n_density_skip == 0:
                    rho = local_density(x, y, L, 1.0)
                    hi = np.percentile(rho, 75)
                    lo = np.percentile(rho, 25)
                    current_class[:] = 2
                    current_class[rho >= hi] = 1
                    current_class[rho <= lo] = 0

                # Turning angle for this step, binned by held class.
                dth = th - prev_th
                dth = (dth + np.pi) % (2.0 * np.pi) - np.pi
                for c in (0, 1):
                    m = current_class == c
                    if m.any():
                        h, _ = np.histogram(dth[m], bins=edges)
                        turn_hist[im, isd, c] += h
                prev_th = th.copy()

                # Record a position/class frame.
                if k % n_pos_skip == 0 and fi < n_frames:
                    pos_frames[:, fi, 0] = unwrap[:, 0]
                    pos_frames[:, fi, 1] = unwrap[:, 1]
                    cls_frames[:, fi] = current_class
                    fi += 1

            phi_mean[im, isd] = phi_acc / n_meas

            # MSD conditioned on the density class at the origin.
            for li, lag in enumerate(lags_f):
                origins = range(0, fi - lag, origin_skip)
                for o in origins:
                    d = pos_frames[:, o + lag] - pos_frames[:, o]
                    disp2 = d[:, 0] ** 2 + d[:, 1] ** 2
                    cls0 = cls_frames[:, o]
                    for c in (0, 1):
                        m = cls0 == c
                        if m.any():
                            msd_sum[im, isd, c, li] += float(disp2[m].sum())
                            msd_cnt[im, isd, c, li] += int(m.sum())

            del pos_frames, cls_frames
            pbar.update(1)
    pbar.close()

    msd = np.where(msd_cnt > 0, msd_sum / np.maximum(msd_cnt, 1), np.nan)
    # Normalise the turning histograms to pdfs (per mode/seed/class).
    dx = edges[1] - edges[0]
    norm = turn_hist.sum(axis=-1, keepdims=True)
    turn_pdf = np.where(norm > 0, turn_hist / np.maximum(norm, 1) / dx,
                        np.nan)

    out = DATA / "double_traj_stats_nocone.npz"
    np.savez_compressed(
        out,
        labels=np.array([c[0] for c in cases]),
        eta_per_mode=np.array([c[5] for c in cases]),
        seeds=np.array(seeds),
        centers=centers, edges=edges,
        turn_pdf=turn_pdf, turn_hist=turn_hist,
        lags_steps=lags_steps,
        msd=msd, msd_cnt=msd_cnt,
        phi_mean=phi_mean,
        params=np.array([N, L, n_warm, n_meas, n_density_skip,
                         n_pos_skip, n_seeds], dtype=float),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")


if __name__ == "__main__":
    main()
