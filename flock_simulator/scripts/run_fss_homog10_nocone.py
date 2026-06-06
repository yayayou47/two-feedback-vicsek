"""Homogeneous ten-seed finite-size scan across all seven sizes.

Referee P1 fix: the published FSS slopes (Table I) rest on three seeds
per size, with only L=90 reseeded to ten and that single node spliced
into a three-seed fit. Here every size L in {15,22,30,45,64,90,128} is
run with the SAME ten seeds and the SAME 5-mode set, and the per-seed
chi(eta) is stored so chi_max, the FSS slopes, the Binder cumulant and
the synergy diagnostic Delta_n can all be refit on a homogeneous
ten-seed series with honest, seed-level error bars.

Protocol mirrors run_fss_large_nocone.py (10-eta grid, warm 1500 +
measure 1000, sigma = 2.22, omnidirectional). Output is written
incrementally after each size to
data/double_fss_homog10_nocone.npz, and a rerun skips sizes already
present, so the ~10 h sweep survives interruptions.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator

ETAS = np.array([0.005, 0.010, 0.020, 0.035, 0.050,
                 0.075, 0.100, 0.150, 0.200, 0.300])
SEEDS = list(range(11, 11 + 30, 3))          # 10 seeds: 11,14,...,38
N_WARM = 1500
N_MEAS = 1000
SIGMA = 2.22
LS = [15.0, 22.0, 30.0, 45.0, 64.0, 90.0, 128.0]

MODES = [
    ("vicsek_gauss", 0.05, 0.05, 2.0, 2.0),
    ("baseline",     0.05, 0.05, 1.0, 1.0),
    ("v2_limit",     0.05, 0.05, 1.0, 2.0),
    ("v3_limit",     0.005, 0.05, 1.0, 1.0),
    ("full",         0.005, 0.05, 1.0, 2.0),
]

OUT = DATA / "double_fss_homog10_nocone.npz"


def _u4(phi_series: np.ndarray) -> float:
    m2 = np.mean(phi_series ** 2)
    m4 = np.mean(phi_series ** 4)
    return float(1.0 - m4 / (3.0 * m2 ** 2)) if m2 > 0 else 0.0


def main() -> None:
    n_mode, n_L, n_eta, n_seed = len(MODES), len(LS), len(ETAS), len(SEEDS)
    shape = (n_mode, n_L, n_eta, n_seed)
    chi = np.full(shape, np.nan)
    phi = np.full(shape, np.nan)
    u4 = np.full(shape, np.nan)
    ssep = np.full(shape, np.nan)
    Ns = np.array([int(round(SIGMA * L * L)) for L in LS])

    # Resume: load any sizes already computed.
    done = set()
    if OUT.exists():
        prev = np.load(OUT, allow_pickle=True)
        if prev["chi"].shape == shape:
            chi, phi, u4, ssep = (prev["chi"].copy(), prev["phi"].copy(),
                                  prev["u4"].copy(), prev["ssep"].copy())
            for iL in range(n_L):
                if np.isfinite(chi[:, iL]).all():
                    done.add(iL)
            if done:
                print(f"resuming; sizes already done: "
                      f"{[int(LS[i]) for i in sorted(done)]}")

    mode_names = [m[0] for m in MODES]
    t0 = time.time()
    for iL, L in enumerate(LS):
        if iL in done:
            continue
        N = int(Ns[iL])
        pbar = tqdm(total=n_mode * n_eta * n_seed,
                    desc=f"FSS L={int(L)} N={N}")
        for im, (name, vmn, vmx, amn, amx) in enumerate(MODES):
            for ie, eta in enumerate(ETAS):
                for isd, seed in enumerate(SEEDS):
                    p = FlockParams(
                        N=N, L=float(L),
                        v_max=float(vmx), v_min=float(vmn),
                        alpha_min=float(amn), alpha_max=float(amx),
                        R_r=0.5, R_a=0.7, eta=float(eta),
                        n_star=3.0, slope=2.0, seed=int(seed),
                    )
                    sim = FlockSimulator(p)
                    for _ in range(N_WARM):
                        sim.step()
                    pser = np.empty(N_MEAS)
                    sser = np.empty(N_MEAS)
                    for k in range(N_MEAS):
                        sim.step()
                        pser[k] = sim.polarisation()
                        sser[k] = sim.density_separation_index(n_bins=10)
                    chi[im, iL, ie, isd] = N * pser.var()
                    phi[im, iL, ie, isd] = pser.mean()
                    u4[im, iL, ie, isd] = _u4(pser)
                    ssep[im, iL, ie, isd] = sser.mean()
                    pbar.update(1)
        pbar.close()
        # Incremental save after every completed size.
        np.savez_compressed(
            OUT, Ls=np.array(LS), Ns=Ns,
            modes=np.array(mode_names), etas=ETAS,
            seeds=np.array(SEEDS), chi=chi, phi=phi, u4=u4, ssep=ssep,
            params=np.array([SIGMA, 0.5, 0.7, N_WARM, N_MEAS,
                             len(SEEDS)], dtype=float),
        )
        print(f"  saved through L={int(L)} "
              f"({(time.time() - t0) / 60:.1f} min elapsed)")
    print(f"\nruntime: {(time.time() - t0) / 60:.1f} min  saved: {OUT.name}")


if __name__ == "__main__":
    main()
