r"""
Samples the polar order parameter at $L = 128$, $\sigma = 2.22$,
for the four heavy-tailed modes (baseline, v2 limit, motility, and
full), over five seeds with 2000 warm-up and a long
$n_{\rm meas} = 60\,000$ measurement steps per seed. Stores the
concatenated per-seed polarisation trajectories per mode and
prints their moments, Binder cumulant, and an estimated number of
histogram modes.

Output: data/double_orderpdf_L128.npz
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


def trace_phi(p, n_warm, n_meas):
    sim = DoubleAdaptiveVicsek(p)
    sim.theta[:] = 0.0
    for _ in range(n_warm):
        sim.step()
    out = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        out[k] = sim.polarisation()
    return out


def main():
    L = 128.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_warm = 2000
    n_meas = 60000
    seeds = list(range(11, 11 + 15, 3))   # 5 seeds

    cases = [
        ("baseline", 0.05,  0.05,  1.0, 1.0, 0.035),
        ("v2_limit", 0.05,  0.05,  1.0, 2.0, 0.075),
        ("motility", 0.005, 0.05,  1.0, 1.0, 0.100),
        ("full",     0.005, 0.05,  1.0, 2.0, 0.150),
    ]

    n_cases = len(cases)
    n_seeds = len(seeds)
    phi_traj = np.zeros((n_cases, n_seeds * n_meas))
    eta_arr = np.zeros(n_cases)

    pbar = tqdm(total=n_cases * n_seeds,
                desc="orderpdf_L128")
    t0 = time.time()
    for ic, (label, vmn, vmx, amn, amx, eta) in enumerate(cases):
        traj_list = []
        for seed in seeds:
            p = DoubleAdaptiveParams(
                N=N, L=float(L),
                v_max=float(vmx), v_min=float(vmn),
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=0.5, R_a=0.7, beta=30.0,
                eta=float(eta),
                n_star=3.0, slope=2.0,
                seed=int(seed),
            )
            traj_list.append(trace_phi(p, n_warm, n_meas))
            pbar.update(1)
        phi_traj[ic] = np.concatenate(traj_list)
        eta_arr[ic] = eta
    pbar.close()

    np.savez_compressed(
        DATA / "double_orderpdf_L128.npz",
        labels=np.array([c[0] for c in cases]),
        L=L,
        eta_per_case=eta_arr,
        phi_traj=phi_traj,
        seeds=np.array(seeds),
        params=np.array([sigma, N, n_warm, n_meas, n_seeds],
                         dtype=float),
    )

    from scipy.stats import skew, kurtosis
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
    print(f"\n=== L = {int(L)}, n_meas = {n_meas} ===")
    for ic, (lbl, *_) in enumerate(cases):
        tr = phi_traj[ic]
        U4 = (1.0 - (tr ** 4).mean()
              / (3.0 * (tr ** 2).mean() ** 2))
        # rough bimodality test: count modes via histogram
        # smoothed local maxima.
        hist, edges = np.histogram(tr, bins=80, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        # count local maxima with prominence > 10% of peak.
        peak_h = hist.max()
        n_peaks = 0
        for j in range(1, len(hist) - 1):
            if (hist[j] > hist[j-1] and hist[j] > hist[j+1]
                    and hist[j] > 0.10 * peak_h):
                n_peaks += 1
        print(f"  {lbl:>10s}  mean={tr.mean():.3f}  "
              f"std={tr.std():.3f}  "
              f"skew={skew(tr):+.2f}  kurt={kurtosis(tr):+.2f}  "
              f"U4={U4:.3f}  n_modes_est={n_peaks}")


if __name__ == "__main__":
    main()
