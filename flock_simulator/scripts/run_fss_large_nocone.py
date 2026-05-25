"""
FSS extension of the four-mode pilot to L in {64, 90, 128} on the
omnidirectional simulator.

Mirrors the legacy ``version4/src/run_double_L{64,90,128}.py``
trio: same eta grid (10 points), same 3 seeds, same warmup +
measurement budget. Includes the Vicsek--Gaussian reference
(alpha_min = alpha_max = 2) so the original 5-mode comparison
table can be rebuilt at every L.

Outputs (one per L):
  data/double_L64_nocone.npz
  data/double_L90_nocone.npz
  data/double_L128_nocone.npz

Each .npz schema matches the legacy single-L files so that
existing figure renderers can swap them in by file-name suffix.

Sequential execution: L=64 first (cheapest, ~1-2 h), then L=90
(~3-4 h), then L=128 (~8-10 h). With ``--start-from`` you can
resume after an interruption (e.g. ``--start-from 90`` to skip
L=64 if it is already complete).

Usage:
  ../.venv/bin/python flock_simulator/scripts/run_fss_large_nocone.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flock_simulator import FlockParams, FlockSimulator   # noqa: E402

DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

ETAS = np.array([0.005, 0.010, 0.020, 0.035, 0.050,
                 0.075, 0.100, 0.150, 0.200, 0.300])
SEEDS = [11, 23, 41]
N_WARM = 1500
N_MEAS = 1000
SIGMA = 2.22

MODES = [
    ("vicsek_gauss", 0.05, 0.05, 2.0, 2.0),
    ("baseline",     0.05, 0.05, 1.0, 1.0),
    ("v2_limit",     0.05, 0.05, 1.0, 2.0),
    ("v3_limit",     0.005, 0.05, 1.0, 1.0),
    ("full",         0.005, 0.05, 1.0, 2.0),
]


def measure(sim: FlockSimulator, n_meas: int):
    phi = np.empty(n_meas)
    s_sep = np.empty(n_meas)
    v_mean = np.empty(n_meas)
    a_mean = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        phi[k] = sim.polarisation()
        s_sep[k] = sim.density_separation_index(n_bins=10)
        v_mean[k] = float(sim.state.v_i.mean())
        a_mean[k] = float(sim.state.alpha_i.mean())
    return phi, s_sep, v_mean, a_mean


def run_one_L(L: float) -> Path:
    N = int(round(SIGMA * L * L))
    n_mode, n_eta, n_seeds = len(MODES), len(ETAS), len(SEEDS)
    phi = np.zeros((n_mode, n_eta))
    chi = np.zeros_like(phi)
    U4 = np.zeros_like(phi)
    s_sep = np.zeros_like(phi)
    v_pop = np.zeros_like(phi)
    a_pop = np.zeros_like(phi)

    pbar = tqdm(total=n_mode * n_eta * n_seeds,
                desc=f"L{int(L)}_nocone")
    t0 = time.time()
    for im, (label, vmn, vmx, amn, amx) in enumerate(MODES):
        for ie, eta in enumerate(ETAS):
            phi_acc, s_acc, v_acc, a_acc = [], [], [], []
            for seed in SEEDS:
                p = FlockParams(
                    N=N, L=float(L),
                    v_max=float(vmx), v_min=float(vmn),
                    alpha_min=float(amn), alpha_max=float(amx),
                    R_r=0.5, R_a=0.7,
                    eta=float(eta),
                    n_star=3.0, slope=2.0,
                    seed=int(seed),
                )
                sim = FlockSimulator(p)
                sim.state.theta[:] = 0.0
                for _ in range(N_WARM):
                    sim.step()
                p_arr, s_arr, v_arr, a_arr = measure(sim, N_MEAS)
                phi_acc.append(p_arr)
                s_acc.append(s_arr)
                v_acc.append(v_arr)
                a_acc.append(a_arr)
                pbar.update(1)
            phi_all = np.concatenate(phi_acc)
            phi[im, ie] = phi_all.mean()
            chi[im, ie] = N * phi_all.var()
            U4[im, ie] = (
                1.0 - np.mean(phi_all ** 4)
                / (3.0 * np.mean(phi_all ** 2) ** 2)
            )
            s_sep[im, ie] = np.concatenate(s_acc).mean()
            v_pop[im, ie] = np.concatenate(v_acc).mean()
            a_pop[im, ie] = np.concatenate(a_acc).mean()
    pbar.close()

    out = DATA / f"double_L{int(L)}_nocone.npz"
    np.savez_compressed(
        out,
        modes=np.array([m[0] for m in MODES]),
        L=L, etas=ETAS, seeds=np.array(SEEDS),
        phi=phi, chi=chi, U4=U4, s_sep=s_sep,
        v_pop=v_pop, a_pop=a_pop,
        params=np.array([SIGMA, N_WARM, N_MEAS, n_seeds]),
    )
    print()
    print(f"L = {int(L)}: runtime {(time.time() - t0) / 60:.1f} min, "
          f"saved {out.name}")
    for im, (lbl, *_) in enumerate(MODES):
        ie = int(np.argmax(chi[im]))
        print(f"  {lbl:>14s}  chi_max={chi[im, ie]:8.2f} @eta={ETAS[ie]:.3f}  "
              f"phi={phi[im, ie]:.3f}  U4={U4[im, ie]:.3f}  "
              f"s_sep={s_sep[im, ie]:.2f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=int, default=64,
                        help="first L to run; remaining sizes follow")
    parser.add_argument("--Ls", type=int, nargs="*",
                        default=[64, 90, 128],
                        help="list of L values to run")
    args = parser.parse_args()
    Ls = [L for L in args.Ls if L >= args.start_from]

    print(f"Running FSS extension at L in {Ls} (sigma = {SIGMA}, "
          f"{N_WARM} warm + {N_MEAS} meas, {len(SEEDS)} seeds, "
          f"{len(MODES)} modes, {len(ETAS)} etas)")
    print()

    for L in Ls:
        run_one_L(float(L))


if __name__ == "__main__":
    main()
