"""Giant Number Fluctuations (GNF).

Canonical active-matter diagnostic absent from the v4 set. We
count particles in nested square sub-boxes and measure
Var(N_box) vs <N_box>. A log-log fit gives the exponent zeta in

    Var(N_box) ~ <N_box>^{2 zeta},

with zeta = 1/2 the equilibrium / Poisson value and zeta > 1/2
the anomalous (giant) fluctuations expected in a MIPS-active or
Toner--Tu polar phase. Four modes at their near-critical eta,
L = 30, many snapshots over 10 seeds.

Output: data/double_gnf_nocone.npz with the box-size grid, the
mean and variance of N_box per mode, and the fitted zeta.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, warm


def box_counts(x, y, L, m):
    """Particle counts on an m x m partition of the box."""
    H, _, _ = np.histogram2d(
        x % L, y % L, bins=[m, m], range=[[0, L], [0, L]])
    return H.ravel()


def main() -> None:
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm, n_meas, n_skip = 2000, 4000, 50
    seeds = list(range(11, 11 + 30, 3))

    # Sub-box side lengths via the number of bins per side. <N_box>
    # = N / m^2, so m in {2,...,20} spans <N_box> from ~5 to ~1100.
    ms = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0, 0.035),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0, 0.075),
        ("v3_limit", 0.005, 0.05,  1.0, 1.0, 0.100),
        ("full",     0.005, 0.05,  1.0, 2.0, 0.150),
    ]
    n_modes, n_seeds, n_m = len(cases), len(seeds), len(ms)

    # accumulate sum and sum-of-squares of N_box per (mode, m)
    acc_sum = np.zeros((n_modes, n_m))
    acc_sq = np.zeros((n_modes, n_m))
    acc_cnt = np.zeros((n_modes, n_m))

    pbar = tqdm(total=n_modes * n_seeds, desc="gnf_nocone")
    t0 = time.time()
    for im, (name, vmn, vmx, amn, amx, eta) in enumerate(cases):
        for seed in seeds:
            p = FlockParams(
                N=N, L=L,
                v_max=float(vmx), v_min=float(vmn),
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=0.5, R_a=0.7, eta=float(eta),
                n_star=3.0, slope=2.0, seed=int(seed),
            )
            sim = FlockSimulator(p)
            warm(sim, n_warm)
            for k in range(n_meas):
                sim.step()
                if k % n_skip == 0:
                    for j, m in enumerate(ms):
                        c = box_counts(sim.state.x, sim.state.y, L, int(m))
                        acc_sum[im, j] += c.sum()
                        acc_sq[im, j] += (c ** 2).sum()
                        acc_cnt[im, j] += c.size
            pbar.update(1)
    pbar.close()

    mean_N = acc_sum / acc_cnt
    var_N = acc_sq / acc_cnt - mean_N ** 2

    # Fit Var ~ mean^(2 zeta) over the well-sampled range
    # <N_box> in [5, 0.3 N] (avoid the saturated large-box end).
    zeta = np.zeros(n_modes)
    fitmask = (mean_N[0] >= 5.0) & (mean_N[0] <= 0.3 * N)
    for im in range(n_modes):
        mk = fitmask & (var_N[im] > 0)
        if mk.sum() >= 3:
            slope, _ = np.polyfit(
                np.log(mean_N[im, mk]), np.log(var_N[im, mk]), 1)
            zeta[im] = 0.5 * slope
        else:
            zeta[im] = np.nan

    out = DATA / "double_gnf_nocone.npz"
    np.savez_compressed(
        out,
        labels=np.array([c[0] for c in cases]),
        etas=np.array([c[5] for c in cases]),
        ms=ms, mean_N=mean_N, var_N=var_N, zeta=zeta,
        params=np.array([N, L, n_warm, n_meas, n_skip, n_seeds],
                         dtype=float),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")
    print(f"\n{'mode':>10s}  {'zeta':>8s}")
    for im, c in enumerate(cases):
        print(f"  {c[0]:>10s}  {zeta[im]:6.3f}")


if __name__ == "__main__":
    main()
