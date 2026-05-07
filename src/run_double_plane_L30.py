"""
Refined behavioural plane (n_star, s) at L = 30 with 5 seeds.
The original plane (run_double_plane.py) used L = 22 and 2 seeds,
which gave a clean s_sep landscape but a noisy chi_max landscape.
Here we re-sweep at L = 30 with five seeds so that the chi_max
landscape is statistically reliable, and we can test whether the
synergic corner (n_star <= 3, s >= 5) carries the steepest
chi_max(L) slope or whether the two landscapes diverge as the
split-synergy verdict on the central working point would suggest.

Output: data/double_plane_L30.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from vicsek_double_adaptive import (DoubleAdaptiveParams,
                                     DoubleAdaptiveVicsek)


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def measure_chi_sep(params: DoubleAdaptiveParams, n_warm: int,
                     n_meas: int):
    sim = DoubleAdaptiveVicsek(params)
    sim.theta[:] = 0.0
    for _ in range(n_warm):
        sim.step()
    phis = np.empty(n_meas)
    seps = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        phis[k] = sim.polarisation()
        seps[k] = sim.density_separation_index(n_bins=10)
    chi = params.N * phis.var()
    return chi, seps.mean()


def main():
    sigma = 2.22
    L = 30.0
    N = int(round(sigma * L * L))
    R_r, R_a = 0.5, 0.7
    n_warm, n_meas = 1500, 800
    seeds = [11, 23, 41, 59, 73]

    n_stars = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    slopes = np.array([1.0, 2.0, 5.0, 10.0])
    etas = np.array([0.05, 0.075, 0.10, 0.15, 0.20])

    n_ns, n_sl, n_e = len(n_stars), len(slopes), len(etas)
    chi = np.zeros((n_ns, n_sl, n_e))
    sep = np.zeros((n_ns, n_sl, n_e))

    pbar = tqdm(total=n_ns * n_sl * n_e * len(seeds),
                desc="plane_L30")
    for ins, n_star in enumerate(n_stars):
        for isl, slope in enumerate(slopes):
            for ie, eta in enumerate(etas):
                c_acc = 0.0
                s_acc = 0.0
                for seed in seeds:
                    p = DoubleAdaptiveParams(
                        N=N, L=L,
                        v_max=0.05, v_min=0.005,
                        alpha_min=1.0, alpha_max=2.0,
                        R_r=R_r, R_a=R_a,
                        beta=30.0, eta=float(eta),
                        n_star=float(n_star),
                        slope=float(slope),
                        seed=int(seed),
                    )
                    ch, sp = measure_chi_sep(p, n_warm, n_meas)
                    c_acc += ch
                    s_acc += sp
                    pbar.update(1)
                chi[ins, isl, ie] = c_acc / len(seeds)
                sep[ins, isl, ie] = s_acc / len(seeds)
    pbar.close()

    chi_peak = chi.max(axis=-1)
    sep_peak = sep.max(axis=-1)

    np.savez_compressed(
        DATA / "double_plane_L30.npz",
        n_stars=n_stars, slopes=slopes, etas=etas, L=L, N=N,
        chi=chi, sep=sep, chi_peak=chi_peak, sep_peak=sep_peak,
        params=np.array([sigma, R_r, R_a, n_warm, n_meas,
                         len(seeds)], dtype=float),
    )

    print()
    print("chi_peak grid (rows=n_star, cols=s), L=30, 5 seeds:")
    print("            s =", "  ".join(f"{s:>6.2f}" for s in slopes))
    for ins, n_star in enumerate(n_stars):
        row = "  ".join(f"{chi_peak[ins, isl]:>6.2f}"
                        for isl in range(n_sl))
        print(f"  n_star={n_star:>3.0f}: {row}")
    print()
    print("sep_peak grid:")
    print("            s =", "  ".join(f"{s:>6.2f}" for s in slopes))
    for ins, n_star in enumerate(n_stars):
        row = "  ".join(f"{sep_peak[ins, isl]:>6.2f}"
                        for isl in range(n_sl))
        print(f"  n_star={n_star:>3.0f}: {row}")
    print("saved:", DATA / "double_plane_L30.npz")


if __name__ == "__main__":
    main()
