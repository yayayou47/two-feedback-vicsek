"""Quasi-static eta hysteresis loop for ALL FOUR modes.

Extends run_hysteresis_nocone.py (motility + full only) with the
two fixed-speed references (baseline / noise-shape) so Fig 9 can
show every mode in one figure. L = 64, 11-point eta ramp up then
down, 5 seeds. Output: data/double_hysteresis_allmodes_nocone.npz,
same schema as the legacy hysteresis file but with four modes.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator


def main() -> None:
    L = 64.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    n_relax, n_meas = 400, 400
    seeds = [11, 23, 41, 57, 73]

    eta_up = np.array([0.02, 0.035, 0.05, 0.075, 0.10, 0.125,
                       0.15, 0.175, 0.20, 0.25, 0.30])
    eta_down = eta_up[::-1]

    cases = [
        ("baseline", 0.05, 0.05, 1.0, 1.0),
        ("v2_limit", 0.05, 0.05, 1.0, 2.0),
        ("v3_limit", 0.005, 0.05, 1.0, 1.0),
        ("full",     0.005, 0.05, 1.0, 2.0),
    ]
    n_modes, n_seeds = len(cases), len(seeds)
    n_eta = len(eta_up)
    phi_up = np.zeros((n_modes, n_seeds, n_eta))
    phi_down = np.zeros((n_modes, n_seeds, n_eta))

    def sweep(sim, etas, store, im, isd):
        for ie, eta in enumerate(etas):
            sim.p.eta = float(eta)
            for _ in range(n_relax):
                sim.step()
            acc = np.empty(n_meas)
            for k in range(n_meas):
                sim.step()
                acc[k] = sim.polarisation()
            store[im, isd, ie] = acc.mean()

    pbar = tqdm(total=n_modes * n_seeds, desc="hysteresis_allmodes")
    t0 = time.time()
    for im, (name, vmn, vmx, amn, amx) in enumerate(cases):
        for isd, seed in enumerate(seeds):
            p = FlockParams(
                N=N, L=L,
                v_max=float(vmx), v_min=float(vmn),
                alpha_min=float(amn), alpha_max=float(amx),
                R_r=0.5, R_a=0.7, eta=float(eta_up[0]),
                n_star=3.0, slope=2.0, seed=int(seed),
            )
            sim = FlockSimulator(p)
            sim.state.theta[:] = 0.0
            for _ in range(1500):
                sim.step()
            sweep(sim, eta_up, phi_up, im, isd)
            sweep(sim, eta_down, phi_down, im, isd)
            pbar.update(1)
    pbar.close()

    phi_up_m = phi_up.mean(axis=1)
    phi_up_se = phi_up.std(axis=1, ddof=1) / np.sqrt(n_seeds)
    phi_dn_m = phi_down.mean(axis=1)[:, ::-1]
    phi_dn_se = phi_down.std(axis=1, ddof=1)[:, ::-1] / np.sqrt(n_seeds)
    loop_area = np.array([
        abs(np.trapezoid(phi_dn_m[im] - phi_up_m[im], eta_up))
        for im in range(n_modes)
    ])

    out = DATA / "double_hysteresis_allmodes_nocone.npz"
    np.savez_compressed(
        out,
        labels=np.array([c[0] for c in cases]),
        eta=eta_up, seeds=np.array(seeds),
        phi_up=phi_up_m, phi_up_se=phi_up_se,
        phi_down=phi_dn_m, phi_down_se=phi_dn_se,
        loop_area=loop_area,
        params=np.array([N, L, n_relax, n_meas, n_seeds], dtype=float),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")
    for im, c in enumerate(cases):
        print(f"  {c[0]:>10s}  loop={loop_area[im]:8.4f}")


if __name__ == "__main__":
    main()
