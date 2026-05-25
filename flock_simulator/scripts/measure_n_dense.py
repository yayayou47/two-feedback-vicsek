"""
Direct measurement of the dense-phase metric occupancy
<n_i>_dense, to be compared to the critical motility threshold
n_star^{v,crit}(sigma) extracted from the sigma sweep.

For each sigma in {1.0, 1.5, 2.22, 3.0}, run a full-mode
trajectory at L = 30, eta = 0.15, 5 seeds, 2000 warmup +
~3000 measurement steps, taking 12 snapshots spaced 200 steps
apart. At each snapshot:

  1. compute the local density rho_i (neighbour count within
     R = 1) of every particle;
  2. identify the dense quartile (top 25%) and the dilute
     quartile (bottom 25%);
  3. read off n_i, the metric annulus count
     R_r <= |r_ij| < R_a, that the shared sigmoid uses;
  4. record <n_i>_dense and <n_i>_dilute over the snapshot.

Average across seeds and snapshots gives <n_i>_dense(sigma).
The prediction is n_star^{v,crit}(sigma) =~ <n_i>_dense(sigma).

Output: data/n_dense_per_sigma.npz.
"""
from __future__ import annotations

import time
import numpy as np
from tqdm import tqdm

from _helpers import DATA, FlockParams, FlockSimulator, warm


def local_density(x, y, L, R):
    N = len(x)
    halfL = 0.5 * L
    R2 = R * R
    rho = np.empty(N, dtype=np.int64)
    for i in range(N):
        dx = x - x[i]; dy = y - y[i]
        dx = np.where(dx > halfL, dx - L, dx)
        dx = np.where(dx < -halfL, dx + L, dx)
        dy = np.where(dy > halfL, dy - L, dy)
        dy = np.where(dy < -halfL, dy + L, dy)
        rho[i] = int(np.count_nonzero(dx * dx + dy * dy < R2)) - 1
    return rho


def main() -> None:
    L = 30.0
    eta = 0.15
    sigmas = np.array([1.0, 1.5, 2.22, 3.0])
    seeds = [11, 14, 17, 20, 23]
    n_warm = 2000
    n_snap = 12
    n_skip = 200

    n_dense_full = np.zeros((len(sigmas), len(seeds), n_snap))
    n_dilute_full = np.zeros_like(n_dense_full)
    n_mean_global = np.zeros_like(n_dense_full)
    pbar = tqdm(total=len(sigmas) * len(seeds),
                desc="n_dense_per_sigma")
    t0 = time.time()
    for isg, sigma in enumerate(sigmas):
        N = int(round(sigma * L * L))
        for isd, seed in enumerate(seeds):
            p = FlockParams(
                N=N, L=L,
                v_max=0.05, v_min=0.005,
                alpha_min=1.0, alpha_max=2.0,
                R_r=0.5, R_a=0.7,
                eta=eta,
                n_star=3.0, slope=2.0,
                seed=int(seed),
            )
            sim = FlockSimulator(p)
            warm(sim, n_warm)
            for k in range(n_snap):
                for _ in range(n_skip):
                    sim.step()
                rho = local_density(
                    sim.state.x, sim.state.y, L, R=1.0)
                hi = np.percentile(rho, 75)
                lo = np.percentile(rho, 25)
                dense = sim.state.n_ali[rho >= hi]
                dilute = sim.state.n_ali[rho <= lo]
                n_dense_full[isg, isd, k] = float(dense.mean())
                n_dilute_full[isg, isd, k] = float(dilute.mean())
                n_mean_global[isg, isd, k] = float(sim.state.n_ali.mean())
            pbar.update(1)
    pbar.close()

    out = DATA / "n_dense_per_sigma.npz"
    np.savez_compressed(
        out,
        sigmas=sigmas, seeds=np.array(seeds),
        n_dense=n_dense_full, n_dilute=n_dilute_full,
        n_mean_global=n_mean_global,
        params=np.array([L, eta, n_warm, n_snap, n_skip],
                         dtype=float),
    )
    print()
    print(f"runtime: {(time.time() - t0) / 60:.1f} min  saved: {out.name}")
    print()
    print(f"{'sigma':>6s}  {'<n>_dense':>14s}  {'<n>_dilute':>14s}  "
          f"{'<n>_global':>14s}")
    for isg, sigma in enumerate(sigmas):
        nd = n_dense_full[isg]; nl = n_dilute_full[isg]
        ng = n_mean_global[isg]
        nd_mean = nd.mean(); nd_se = nd.std(ddof=1) / np.sqrt(len(seeds))
        nl_mean = nl.mean(); nl_se = nl.std(ddof=1) / np.sqrt(len(seeds))
        ng_mean = ng.mean(); ng_se = ng.std(ddof=1) / np.sqrt(len(seeds))
        print(f"  {sigma:>4.2f}  {nd_mean:>7.2f}+-{nd_se:.2f}  "
              f"{nl_mean:>7.2f}+-{nl_se:.2f}  "
              f"{ng_mean:>7.2f}+-{ng_se:.2f}")

    # Compare to n_v_crit from sigma sweep.
    try:
        sw = np.load(DATA / "sigma_sweep_summary.npz")
    except FileNotFoundError:
        print("\n[warn] sigma_sweep_summary.npz not found; "
              "run analyse_sigma_sweep.py first.")
        return
    sigmas_sw = sw["sigmas"]
    crit = sw["n_v_crit"]
    crit_se = sw["n_v_crit_se"]
    print()
    print(f"{'sigma':>6s}  {'n_v_crit':>16s}  {'<n>_dense':>14s}  "
          f"{'ratio':>8s}")
    for isg, sigma in enumerate(sigmas):
        # Locate matching sigma in the sweep.
        i_sw = int(np.argmin(np.abs(sigmas_sw - sigma)))
        if abs(sigmas_sw[i_sw] - sigma) > 1e-2:
            continue
        c = float(crit[i_sw]); cse = float(crit_se[i_sw])
        nd = n_dense_full[isg].mean()
        nd_se = n_dense_full[isg].std(ddof=1) / np.sqrt(len(seeds))
        ratio = c / nd if nd > 0 else float("nan")
        print(f"  {sigma:>4.2f}  {c:>7.2f}+-{cse:.2f}  "
              f"{nd:>7.2f}+-{nd_se:.2f}  {ratio:>8.2f}")


if __name__ == "__main__":
    main()
