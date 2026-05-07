"""
Final refit of FSS slopes using all three data sources:
  - data/double_pilot.npz  (4 sizes, 10 etas, 3 seeds)
  - data/double_L64.npz    (1 size = 64, 10 etas, 3 seeds)
  - data/double_finegrid.npz (5 sizes, 4 small etas, 2 seeds)

For each (mode, L), chi_max(L) is the maximum chi over the union
of eta grids, weighted by seed count via inverse-variance averaging
on overlapping etas (no overlap by construction here, all small etas
are new). Slopes are log-log least-squares on the 5 sizes.

Output: prints the merged chi_max table, slopes, Delta synergy.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def main():
    pilot = np.load(DATA / "double_pilot.npz", allow_pickle=True)
    big = np.load(DATA / "double_L64.npz", allow_pickle=True)
    fine = np.load(DATA / "double_finegrid.npz", allow_pickle=True)

    Ls = np.concatenate([pilot["Ls"], [float(big["L"])]])  # {15,22,30,45,64}
    modes = [str(s) if isinstance(s, str) else s.decode()
             for s in pilot["modes"]]

    # Pilot chi: shape (n_mode, 4, n_eta_pilot)
    chi_pilot = pilot["chi"]
    # Big chi at L=64: shape (n_mode, n_eta_pilot)
    chi_big = big["chi"]
    # Fine chi: shape (n_mode, 5, n_eta_fine)
    chi_fine = fine["chi"]

    print(f"Sizes: {Ls.tolist()}")
    print(f"Pilot etas: {pilot['etas'].tolist()}")
    print(f"Fine  etas: {fine['etas'].tolist()}")
    print()

    # Merge per (mode, L): take max over union of eta grids
    chi_max_5L = np.zeros((len(modes), len(Ls)))
    eta_at_max = np.zeros_like(chi_max_5L)
    src_at_max = np.empty(chi_max_5L.shape, dtype=object)
    for im, m in enumerate(modes):
        for iL in range(len(Ls)):
            if iL < 4:
                cand_main = chi_pilot[im, iL]
                etas_main = pilot["etas"]
            else:
                cand_main = chi_big[im]
                etas_main = pilot["etas"]
            cand_fine = chi_fine[im, iL]
            etas_fine = fine["etas"]
            # Take max over union
            all_chi = np.concatenate([cand_main, cand_fine])
            all_eta = np.concatenate([etas_main, etas_fine])
            all_src = (["coarse"] * len(cand_main)
                       + ["fine"] * len(cand_fine))
            j = int(np.argmax(all_chi))
            chi_max_5L[im, iL] = all_chi[j]
            eta_at_max[im, iL] = all_eta[j]
            src_at_max[im, iL] = all_src[j]

    print("Merged chi_max(L), peak eta, source:")
    for im, m in enumerate(modes):
        print(f"  {m:10s}", end="")
        for iL, L in enumerate(Ls):
            print(f"  L={int(L):2d}: {chi_max_5L[im, iL]:7.2f}@"
                  f"{eta_at_max[im, iL]:.4f}({src_at_max[im, iL][0]})",
                  end="")
        print()

    print()
    slopes = {}
    for im, m in enumerate(modes):
        a, _ = np.polyfit(np.log(Ls), np.log(chi_max_5L[im]), 1)
        slopes[m] = a
        print(f"  {m:10s}  slope = {a:+.3f}")

    delta = ((slopes["full"] - slopes["baseline"])
             - (slopes["v2_limit"] - slopes["baseline"])
             - (slopes["v3_limit"] - slopes["baseline"]))
    sign = ("SYNERGIC" if delta > 0
            else "ANTAGONISTIC" if delta < 0 else "ADDITIVE")
    print()
    print(f"  full - baseline = {slopes['full'] - slopes['baseline']:+.3f}")
    print(f"  v2 - baseline   = {slopes['v2_limit'] - slopes['baseline']:+.3f}")
    print(f"  v3 - baseline   = {slopes['v3_limit'] - slopes['baseline']:+.3f}")
    print(f"  Delta(refined)  = {delta:+.3f}  -> {sign}")


if __name__ == "__main__":
    main()
