"""No-cone port of legacy run_double_plane.py.

(n_star, s) sigmoid plane sweep at L=22 for the full mode, with
chi_max and s_sep_max over a small eta sub-grid. 5x4 grid x 5
etas x 2 seeds = 200 jobs, mostly small-N so cheap.
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, measure_phi_sep, warm


def main() -> None:
    sigma = 2.22
    L = 22.0
    N = int(round(sigma * L * L))
    n_warm, n_meas = 1500, 800
    seeds = [11, 23]

    n_stars = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    slopes = np.array([1.0, 2.0, 5.0, 10.0])
    etas = np.array([0.05, 0.075, 0.10, 0.15, 0.20])
    n_ns, n_sl, n_e = len(n_stars), len(slopes), len(etas)
    chi = np.zeros((n_ns, n_sl, n_e))
    sep = np.zeros((n_ns, n_sl, n_e))

    pbar = tqdm(total=n_ns * n_sl * n_e * len(seeds),
                desc="plane_nocone")
    for ins, n_star in enumerate(n_stars):
        for isl, slope in enumerate(slopes):
            for ie, eta in enumerate(etas):
                c_acc, s_acc = 0.0, 0.0
                for seed in seeds:
                    p = FlockParams(
                        N=N, L=L,
                        v_max=0.05, v_min=0.005,
                        alpha_min=1.0, alpha_max=2.0,
                        R_r=0.5, R_a=0.7, eta=float(eta),
                        n_star=float(n_star), slope=float(slope),
                        seed=int(seed),
                    )
                    sim = FlockSimulator(p)
                    warm(sim, n_warm)
                    phis, seps = measure_phi_sep(sim, n_meas)
                    c_acc += N * phis.var()
                    s_acc += seps.mean()
                    pbar.update(1)
                chi[ins, isl, ie] = c_acc / len(seeds)
                sep[ins, isl, ie] = s_acc / len(seeds)
    pbar.close()
    chi_peak = chi.max(axis=-1)
    sep_peak = sep.max(axis=-1)

    out = DATA / "double_plane_nocone.npz"
    np.savez_compressed(
        out,
        n_stars=n_stars, slopes=slopes, etas=etas, L=L, N=N,
        chi=chi, sep=sep, chi_peak=chi_peak, sep_peak=sep_peak,
        params=np.array([sigma, 0.5, 0.7, n_warm, n_meas,
                         len(seeds)], dtype=float),
    )
    print(f"\nsaved: {out.name}")
    print("chi_peak:")
    for ins, n_star in enumerate(n_stars):
        print(f"  n_star={n_star:>3.0f}: " +
              "  ".join(f"{chi_peak[ins, isl]:>6.2f}"
                        for isl in range(n_sl)))
    print("sep_peak:")
    for ins, n_star in enumerate(n_stars):
        print(f"  n_star={n_star:>3.0f}: " +
              "  ".join(f"{sep_peak[ins, isl]:>6.2f}"
                        for isl in range(n_sl)))


if __name__ == "__main__":
    main()
