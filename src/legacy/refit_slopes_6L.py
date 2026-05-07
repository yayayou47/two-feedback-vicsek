"""
Final 6-size FSS refit including L = 90. Combines:
  - data/double_pilot.npz       (4 sizes 15-45, 4 modes, 10 etas, 3 seeds)
  - data/double_L64.npz         (1 size 64,  4 modes, 10 etas, 3 seeds)
  - data/double_finegrid.npz    (5 sizes 15-64, 4 modes, 4 small etas, 2 seeds)
  - data/vicsek_gauss_ref.npz   (5 sizes 15-64, Vicsek-Gauss, 10 etas, 3 seeds)
  - data/double_L90.npz         (1 size 90,  5 modes, 10 etas, 3 seeds)

For each (mode, L), chi_max(L) is the maximum chi over the union
of available eta grids. Slopes are log-log least-squares on the
six sizes {15, 22, 30, 45, 64, 90} when available, else five.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def main():
    pilot = np.load(DATA / "double_pilot.npz", allow_pickle=True)
    big = np.load(DATA / "double_L64.npz", allow_pickle=True)
    fine = np.load(DATA / "double_finegrid.npz", allow_pickle=True)
    vg = np.load(DATA / "vicsek_gauss_ref.npz", allow_pickle=True)
    big90 = np.load(DATA / "double_L90.npz", allow_pickle=True)

    # Heavy-tailed modes order: baseline, v2_limit, v3_limit, full
    ht_modes = [str(s) if isinstance(s, str) else s.decode()
                for s in pilot["modes"]]
    Ls = np.array([15.0, 22.0, 30.0, 45.0, 64.0, 90.0])
    print(f"Sizes: {Ls.tolist()}")
    print()

    chi_max_6L = np.zeros((len(ht_modes) + 1, len(Ls)))
    eta_at_max = np.zeros_like(chi_max_6L)
    src_at_max = np.empty(chi_max_6L.shape, dtype=object)

    # Vicsek-Gauss row first (no fine eta grid for it).
    vg_chi = vg["chi"]                              # (5, 10)
    vg_etas = vg["etas"]
    vg_modes_L90_idx = list(big90["modes"]).index(b"vicsek_gauss"
                                                   if isinstance(big90["modes"][0], bytes)
                                                   else "vicsek_gauss")
    chi_vg_L90 = big90["chi"][vg_modes_L90_idx]
    for iL in range(5):
        j = int(np.argmax(vg_chi[iL]))
        chi_max_6L[0, iL] = vg_chi[iL, j]
        eta_at_max[0, iL] = vg_etas[j]
        src_at_max[0, iL] = "ref"
    j = int(np.argmax(chi_vg_L90))
    chi_max_6L[0, 5] = chi_vg_L90[j]
    eta_at_max[0, 5] = big90["etas"][j]
    src_at_max[0, 5] = "L90"

    # Heavy-tailed modes
    for im, m in enumerate(ht_modes, start=1):
        ht_im = im - 1
        # L = 15..45 from pilot, with finegrid added
        for iL in range(4):
            cand_main = pilot["chi"][ht_im, iL]
            cand_fine = fine["chi"][ht_im, iL]
            all_chi = np.concatenate([cand_main, cand_fine])
            all_eta = np.concatenate([pilot["etas"], fine["etas"]])
            all_src = (["c"] * len(cand_main)
                       + ["f"] * len(cand_fine))
            j = int(np.argmax(all_chi))
            chi_max_6L[im, iL] = all_chi[j]
            eta_at_max[im, iL] = all_eta[j]
            src_at_max[im, iL] = all_src[j]
        # L = 64 from big + finegrid
        cand_main = big["chi"][ht_im]
        cand_fine = fine["chi"][ht_im, 4]
        all_chi = np.concatenate([cand_main, cand_fine])
        all_eta = np.concatenate([big["etas"], fine["etas"]])
        all_src = (["c"] * len(cand_main)
                   + ["f"] * len(cand_fine))
        j = int(np.argmax(all_chi))
        chi_max_6L[im, 4] = all_chi[j]
        eta_at_max[im, 4] = all_eta[j]
        src_at_max[im, 4] = all_src[j]
        # L = 90 from big90
        l90_modes = list(big90["modes"])
        if isinstance(l90_modes[0], bytes):
            l90_modes = [b.decode() for b in l90_modes]
        try:
            i90 = l90_modes.index(m)
        except ValueError:
            chi_max_6L[im, 5] = np.nan
            continue
        cm90 = big90["chi"][i90]
        j = int(np.argmax(cm90))
        chi_max_6L[im, 5] = cm90[j]
        eta_at_max[im, 5] = big90["etas"][j]
        src_at_max[im, 5] = "L90"

    rows = ["vicsek_gauss"] + ht_modes
    print("Merged chi_max(L):")
    print("  " + " " * 13 + " ".join(f"L={int(L):>3d}" for L in Ls))
    for im, m in enumerate(rows):
        print(f"  {m:13s} "
              + " ".join(f"{chi_max_6L[im, iL]:7.2f}"
                          for iL in range(len(Ls))))
    print()
    print("Peak eta:")
    for im, m in enumerate(rows):
        print(f"  {m:13s} "
              + " ".join(f"{eta_at_max[im, iL]:.4f}"
                          for iL in range(len(Ls))))
    print()

    slopes = {}
    for im, m in enumerate(rows):
        cm = chi_max_6L[im]
        a, _ = np.polyfit(np.log(Ls), np.log(cm), 1)
        slopes[m] = a
        print(f"  {m:13s}  slope(6 sizes) = {a:+.3f}")

    # Synergy diagnostic on the heavy-tailed Cauchy backdrop
    delta = ((slopes["full"] - slopes["baseline"])
             - (slopes["v2_limit"] - slopes["baseline"])
             - (slopes["v3_limit"] - slopes["baseline"]))
    sign = ("SYNERGIC" if delta > 0
            else "ANTAGONISTIC" if delta < 0 else "ADDITIVE")
    print()
    print(f"  Delta(6 sizes)  = {delta:+.3f}   -> {sign}")
    print(f"    full   - baseline = {slopes['full'] - slopes['baseline']:+.3f}")
    print(f"    v2     - baseline = {slopes['v2_limit'] - slopes['baseline']:+.3f}")
    print(f"    v3     - baseline = {slopes['v3_limit'] - slopes['baseline']:+.3f}")

    # 5-size reference for comparison
    print()
    print("Reference: 5-size slopes on L<=64 (without L=90):")
    slopes5 = {}
    for im, m in enumerate(rows):
        a5, _ = np.polyfit(np.log(Ls[:-1]),
                            np.log(chi_max_6L[im, :-1]), 1)
        slopes5[m] = a5
        print(f"  {m:13s}  slope(5) = {a5:+.3f}    "
              f"slope(6) = {slopes[m]:+.3f}    "
              f"shift = {slopes[m] - a5:+.3f}")
    delta5 = ((slopes5["full"] - slopes5["baseline"])
              - (slopes5["v2_limit"] - slopes5["baseline"])
              - (slopes5["v3_limit"] - slopes5["baseline"]))
    print(f"  Delta(5 sizes) = {delta5:+.3f}    Delta(6 sizes) = {delta:+.3f}")

    # s_sep across L for each mode
    print()
    print("s_sep_max(L=90) per mode:")
    for im, m in enumerate(big90["modes"]):
        m_str = m.decode() if isinstance(m, bytes) else m
        smax = big90["s_sep"][im].max()
        ie = int(np.argmax(big90["s_sep"][im]))
        print(f"  {m_str:13s}  s_sep_max = {smax:.2f}  "
              f"@eta={big90['etas'][ie]:.3f}")


if __name__ == "__main__":
    main()
