"""
Pilot of the omnidirectional two-feedback Vicsek--Couzin model.

Mirrors ``version4/src/run_double_pilot.py`` but uses the new
no-cone simulator. Sweeps eta on the same 10-point grid for
L in {15, 22, 30, 45} at sigma = N/L^2 ~ 2.22, three seeds, four
modes (baseline / v2-limit / v3-limit / full).

Output: ``data/double_pilot_nocone.npz`` with the same schema as
``data/double_pilot.npz`` so figure renderers can swap one for the
other.
"""
from __future__ import annotations

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


def measure(sim: FlockSimulator, n_meas: int, every: int = 1):
    n = n_meas // every
    phi = np.empty(n)
    s_sep = np.empty(n)
    v_mean = np.empty(n)
    a_mean = np.empty(n)
    j = 0
    for k in range(n_meas):
        sim.step()
        if k % every == 0:
            phi[j] = sim.polarisation()
            s_sep[j] = sim.density_separation_index(n_bins=10)
            v_mean[j] = float(sim.state.v_i.mean())
            a_mean[j] = float(sim.state.alpha_i.mean())
            j += 1
    return phi[:j], s_sep[:j], v_mean[:j], a_mean[:j]


def main() -> None:
    Ls = np.array([15.0, 22.0, 30.0, 45.0])
    sigma = 2.22
    etas = np.array([0.005, 0.010, 0.020, 0.035, 0.050,
                     0.075, 0.100, 0.150, 0.200, 0.300])
    seeds = [11, 23, 41]
    n_warm = 1500
    n_meas = 1000

    modes = [
        ("baseline", 0.05, 0.05, 1.0, 1.0),
        ("v2_limit", 0.05, 0.05, 1.0, 2.0),
        ("v3_limit", 0.005, 0.05, 1.0, 1.0),
        ("full",     0.005, 0.05, 1.0, 2.0),
    ]
    n_mode, n_L, n_eta, n_seeds = (
        len(modes), len(Ls), len(etas), len(seeds),
    )

    phi = np.zeros((n_mode, n_L, n_eta))
    chi = np.zeros_like(phi)
    U4 = np.zeros_like(phi)
    s_sep = np.zeros_like(phi)
    v_pop = np.zeros_like(phi)
    a_pop = np.zeros_like(phi)
    phi_per_seed = np.zeros((n_mode, n_L, n_eta, n_seeds))
    chi_per_seed = np.zeros_like(phi_per_seed)
    U4_per_seed = np.zeros_like(phi_per_seed)
    s_sep_per_seed = np.zeros_like(phi_per_seed)
    v_pop_per_seed = np.zeros_like(phi_per_seed)
    a_pop_per_seed = np.zeros_like(phi_per_seed)

    pbar = tqdm(total=n_mode * n_L * n_eta * n_seeds,
                desc="pilot_nocone")
    t0 = time.time()
    for im, (label, vmn, vmx, amn, amx) in enumerate(modes):
        for iL, L in enumerate(Ls):
            N = int(round(sigma * L * L))
            for ie, eta in enumerate(etas):
                phi_acc, s_acc, v_acc, a_acc = [], [], [], []
                for isd, seed in enumerate(seeds):
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
                    for _ in range(n_warm):
                        sim.step()
                    p_arr, s_arr, v_arr, a_arr = measure(sim, n_meas)
                    phi_acc.append(p_arr)
                    s_acc.append(s_arr)
                    v_acc.append(v_arr)
                    a_acc.append(a_arr)
                    phi_per_seed[im, iL, ie, isd] = p_arr.mean()
                    chi_per_seed[im, iL, ie, isd] = N * p_arr.var()
                    U4_per_seed[im, iL, ie, isd] = (
                        1.0 - np.mean(p_arr ** 4)
                        / (3.0 * np.mean(p_arr ** 2) ** 2)
                    )
                    s_sep_per_seed[im, iL, ie, isd] = s_arr.mean()
                    v_pop_per_seed[im, iL, ie, isd] = v_arr.mean()
                    a_pop_per_seed[im, iL, ie, isd] = a_arr.mean()
                    pbar.update(1)
                phi_all = np.concatenate(phi_acc)
                phi[im, iL, ie] = phi_all.mean()
                chi[im, iL, ie] = N * phi_all.var()
                U4[im, iL, ie] = (
                    1.0 - np.mean(phi_all ** 4)
                    / (3.0 * np.mean(phi_all ** 2) ** 2)
                )
                s_sep[im, iL, ie] = np.concatenate(s_acc).mean()
                v_pop[im, iL, ie] = np.concatenate(v_acc).mean()
                a_pop[im, iL, ie] = np.concatenate(a_acc).mean()
    pbar.close()

    np.savez_compressed(
        DATA / "double_pilot_nocone.npz",
        modes=np.array([m[0] for m in modes]),
        Ls=Ls, etas=etas, seeds=np.array(seeds),
        phi=phi, chi=chi, U4=U4, s_sep=s_sep,
        v_pop=v_pop, a_pop=a_pop,
        phi_per_seed=phi_per_seed,
        chi_per_seed=chi_per_seed,
        U4_per_seed=U4_per_seed,
        s_sep_per_seed=s_sep_per_seed,
        v_pop_per_seed=v_pop_per_seed,
        a_pop_per_seed=a_pop_per_seed,
        params=np.array([sigma, n_warm, n_meas, n_seeds]),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min")
    print(f"saved: {DATA / 'double_pilot_nocone.npz'}")
    for im, (label, *_) in enumerate(modes):
        print(f"\n=== {label} ===")
        for iL, L in enumerate(Ls):
            ie_max = int(np.argmax(chi[im, iL]))
            print(f"  L={L:>5.1f}  chi_max={chi[im, iL, ie_max]:6.2f}  "
                  f"@eta={etas[ie_max]:.3f}  phi={phi[im, iL, ie_max]:.3f}  "
                  f"U4={U4[im, iL, ie_max]:.3f}  "
                  f"s_sep={s_sep[im, iL, ie_max]:.2f}")


if __name__ == "__main__":
    main()
