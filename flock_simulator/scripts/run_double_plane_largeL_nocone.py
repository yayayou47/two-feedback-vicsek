"""Large-L (n_star, slope) plane for the full two-feedback model.

Extends the behavioural-plane survey to L=90 and L=128 on the same
fine 13x13 grid, five etas and three seeds used for the L=22/30
panels (run_double_plane_fine_nocone.py), so the four-size figure
carries identical statistics across rows. Output:
data/double_plane_largeL_nocone.npz with chi_peak / sep_peak maps
per size, matching the fine-file schema.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, measure_phi_sep, warm


def sweep_plane(L: float, n_stars, slopes, etas, seeds,
                n_warm: int, n_meas: int):
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_ns, n_sl, n_e = len(n_stars), len(slopes), len(etas)
    chi = np.zeros((n_ns, n_sl, n_e))
    sep = np.zeros((n_ns, n_sl, n_e))
    pbar = tqdm(total=n_ns * n_sl * n_e * len(seeds),
                desc=f"plane_largeL L={L:g}")
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
    return chi.max(axis=-1), sep.max(axis=-1), N


def main() -> None:
    n_stars = np.linspace(1.0, 8.0, 13)
    slopes = np.linspace(1.0, 10.0, 13)
    etas = np.array([0.05, 0.075, 0.10, 0.15, 0.20])
    seeds = [11, 23, 41]
    n_warm, n_meas = 1000, 600

    t0 = time.time()
    chi90, sep90, N90 = sweep_plane(90.0, n_stars, slopes, etas, seeds,
                                    n_warm, n_meas)
    chi128, sep128, N128 = sweep_plane(128.0, n_stars, slopes, etas,
                                       seeds, n_warm, n_meas)

    out = DATA / "double_plane_largeL_nocone.npz"
    np.savez_compressed(
        out,
        n_stars=n_stars, slopes=slopes, etas=etas,
        L90=90.0, L128=128.0, N90=N90, N128=N128,
        chi_peak_90=chi90, sep_peak_90=sep90,
        chi_peak_128=chi128, sep_peak_128=sep128,
        params=np.array([2.22, 0.5, 0.7, n_warm, n_meas, len(seeds)],
                        dtype=float),
    )
    print(f"\nruntime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")


if __name__ == "__main__":
    main()
