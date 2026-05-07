"""
High-statistics scan at three intermediate sizes L in {50, 64,
80} with 10 seeds. Goal: tighten the gap full - motility on
s_sep and discriminate between a deterministic peak buried in
seed noise and a genuinely flat advantage on the
intermediate-$L$ window. We also report seed-to-seed standard
errors so the gap can be quoted with statistical weight.

Output: data/double_L_highstat.npz
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


def measure(sim, n_meas):
    phi = np.empty(n_meas)
    s_sep = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        phi[k] = sim.polarisation()
        s_sep[k] = sim.density_separation_index(n_bins=10)
    return phi, s_sep


def main():
    sigma = 2.22
    Ls = np.array([50.0, 64.0, 80.0])
    etas = np.array([0.075, 0.100, 0.150, 0.200])
    seeds = list(range(11, 11 + 30, 3))   # 10 seeds: 11, 14, ..., 38
    n_warm = 1500
    n_meas = 1500

    modes = [
        ("motility", 0.005, 0.05, 1.0, 1.0),
        ("full",     0.005, 0.05, 1.0, 2.0),
    ]

    n_mode, n_L, n_eta, n_seed = (
        len(modes), len(Ls), len(etas), len(seeds))
    # Per-seed observables so we can compute SE.
    s_sep_per_seed = np.zeros((n_mode, n_L, n_eta, n_seed))
    chi_per_seed = np.zeros((n_mode, n_L, n_eta, n_seed))
    phi_per_seed = np.zeros((n_mode, n_L, n_eta, n_seed))

    pbar = tqdm(total=n_mode * n_L * n_eta * n_seed,
                desc="L_highstat")
    t0 = time.time()
    for im, (label, vmn, vmx, amn, amx) in enumerate(modes):
        for iL, L in enumerate(Ls):
            N = int(round(sigma * L * L))
            for ie, eta in enumerate(etas):
                for isd, seed in enumerate(seeds):
                    p = DoubleAdaptiveParams(
                        N=N, L=float(L),
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
                    phi_arr, s_arr = measure(sim, n_meas)
                    phi_per_seed[im, iL, ie, isd] = phi_arr.mean()
                    chi_per_seed[im, iL, ie, isd] = N * phi_arr.var()
                    s_sep_per_seed[im, iL, ie, isd] = s_arr.mean()
                    pbar.update(1)
    pbar.close()

    np.savez_compressed(
        DATA / "double_L_highstat.npz",
        modes=np.array([m[0] for m in modes]),
        Ls=Ls, etas=etas, seeds=np.array(seeds),
        phi_per_seed=phi_per_seed, chi_per_seed=chi_per_seed,
        s_sep_per_seed=s_sep_per_seed,
        params=np.array([sigma, n_warm, n_meas, n_seed]),
    )

    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
    # Summary: take max over eta of the seed-mean s_sep, then SE on seeds.
    print(f"\n{'L':>4s}  "
          f"{'mode':>10s}  "
          f"{'<s_sep>':>10s}  "
          f"{'eta*':>6s}  "
          f"{'SE':>6s}")
    for iL, L in enumerate(Ls):
        for im, (label, *_) in enumerate(modes):
            mean_e = s_sep_per_seed[im, iL].mean(axis=-1)  # over seeds
            ie_max = int(np.argmax(mean_e))
            seed_mean = mean_e[ie_max]
            seed_se = (s_sep_per_seed[im, iL, ie_max].std(ddof=1)
                       / np.sqrt(n_seed))
            print(f"  L={int(L):3d}  {label:>10s}  "
                  f"{seed_mean:.4f}  "
                  f"@{etas[ie_max]:.3f}  "
                  f"{seed_se:.4f}")
    print()
    print("Gap (full - motility) per L, with SE:")
    for iL, L in enumerate(Ls):
        m_mot = s_sep_per_seed[0, iL].mean(axis=-1)
        m_full = s_sep_per_seed[1, iL].mean(axis=-1)
        ie_mot = int(np.argmax(m_mot))
        ie_full = int(np.argmax(m_full))
        gap = m_full[ie_full] - m_mot[ie_mot]
        # SE on the difference at the same eta (use full's optimal eta)
        diff_at_full_eta = (s_sep_per_seed[1, iL, ie_full]
                            - s_sep_per_seed[0, iL, ie_full])
        se_gap = diff_at_full_eta.std(ddof=1) / np.sqrt(n_seed)
        print(f"  L={int(L):3d}  gap={gap:+.4f} +/- {se_gap:.4f}  "
              f"(z = {gap / max(se_gap, 1e-9):+.2f})")


if __name__ == "__main__":
    main()
