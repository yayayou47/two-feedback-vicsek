"""
Final 7-size FSS refit: L = 15, 22, 30, 45, 64, 90, 128.
Combines double_pilot.npz (L 15-45), double_L64.npz (L=64),
double_finegrid.npz (small-eta), vicsek_gauss_ref.npz (Vicsek,
L 15-64), double_L90.npz (L=90), double_L128.npz (L=128).
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
    big128 = np.load(DATA / "double_L128.npz", allow_pickle=True)

    ht_modes = [str(s) if isinstance(s, str) else s.decode()
                for s in pilot["modes"]]
    Ls = np.array([15.0, 22.0, 30.0, 45.0, 64.0, 90.0, 128.0])
    print(f"Sizes: {Ls.tolist()}")
    print()

    chi_max = np.zeros((len(ht_modes) + 1, len(Ls)))
    eta_max = np.zeros_like(chi_max)
    s_sep_max = np.zeros_like(chi_max)

    def big_idx(arr, name):
        modes_arr = list(arr["modes"])
        if isinstance(modes_arr[0], bytes):
            modes_arr = [m.decode() for m in modes_arr]
        try:
            return modes_arr.index(name)
        except ValueError:
            return None

    # Vicsek-Gauss row
    vg_chi = vg["chi"]
    vg_etas = vg["etas"]
    vg_s = vg["s_sep"]
    for iL in range(5):
        j = int(np.argmax(vg_chi[iL]))
        chi_max[0, iL] = vg_chi[iL, j]
        eta_max[0, iL] = vg_etas[j]
        s_sep_max[0, iL] = vg_s[iL].max()
    for il_big, big_arr, idx_in_Ls in (
        (5, big90, 5), (6, big128, 6),
    ):
        idx = big_idx(big_arr, "vicsek_gauss")
        if idx is None:
            chi_max[0, idx_in_Ls] = np.nan
            continue
        j = int(np.argmax(big_arr["chi"][idx]))
        chi_max[0, idx_in_Ls] = big_arr["chi"][idx, j]
        eta_max[0, idx_in_Ls] = big_arr["etas"][j]
        s_sep_max[0, idx_in_Ls] = big_arr["s_sep"][idx].max()

    # Heavy-tailed modes
    for im, m in enumerate(ht_modes, start=1):
        ht_im = im - 1
        for iL in range(4):
            cm = pilot["chi"][ht_im, iL]
            cf = fine["chi"][ht_im, iL]
            all_chi = np.concatenate([cm, cf])
            all_eta = np.concatenate([pilot["etas"], fine["etas"]])
            j = int(np.argmax(all_chi))
            chi_max[im, iL] = all_chi[j]
            eta_max[im, iL] = all_eta[j]
            s_sep_max[im, iL] = pilot["s_sep"][ht_im, iL].max()
        # L = 64
        cm = big["chi"][ht_im]
        cf = fine["chi"][ht_im, 4]
        all_chi = np.concatenate([cm, cf])
        all_eta = np.concatenate([big["etas"], fine["etas"]])
        j = int(np.argmax(all_chi))
        chi_max[im, 4] = all_chi[j]
        eta_max[im, 4] = all_eta[j]
        s_sep_max[im, 4] = big["s_sep"][ht_im].max()
        # L = 90 and 128
        for big_arr, idx_in_Ls in ((big90, 5), (big128, 6)):
            idx = big_idx(big_arr, m)
            if idx is None:
                chi_max[im, idx_in_Ls] = np.nan
                s_sep_max[im, idx_in_Ls] = np.nan
                continue
            cm = big_arr["chi"][idx]
            j = int(np.argmax(cm))
            chi_max[im, idx_in_Ls] = cm[j]
            eta_max[im, idx_in_Ls] = big_arr["etas"][j]
            s_sep_max[im, idx_in_Ls] = big_arr["s_sep"][idx].max()

    rows = ["vicsek_gauss"] + ht_modes
    print("chi_max(L):")
    print("  " + " " * 13 + " ".join(f"L={int(L):>3d}" for L in Ls))
    for im, m in enumerate(rows):
        print(f"  {m:13s} "
              + " ".join(f"{chi_max[im, iL]:7.2f}"
                          for iL in range(len(Ls))))
    print()
    print("s_sep_max(L):")
    for im, m in enumerate(rows):
        print(f"  {m:13s} "
              + " ".join(f"{s_sep_max[im, iL]:7.2f}"
                          for iL in range(len(Ls))))
    print()

    slopes = {}
    for im, m in enumerate(rows):
        cm = chi_max[im]
        ok = np.isfinite(cm)
        a, _ = np.polyfit(np.log(Ls[ok]), np.log(cm[ok]), 1)
        slopes[m] = a
        print(f"  {m:13s}  slope(7) = {a:+.3f}")

    delta7 = ((slopes["full"] - slopes["baseline"])
              - (slopes["v2_limit"] - slopes["baseline"])
              - (slopes["v3_limit"] - slopes["baseline"]))
    sign = ("SYNERGIC" if delta7 > 0
            else "ANTAGONISTIC" if delta7 < 0 else "ADDITIVE")
    print()
    print(f"  Delta(7 sizes) = {delta7:+.3f}   -> {sign}")
    print(f"    full   - baseline = {slopes['full'] - slopes['baseline']:+.3f}")
    print(f"    v2     - baseline = {slopes['v2_limit'] - slopes['baseline']:+.3f}")
    print(f"    v3     - baseline = {slopes['v3_limit'] - slopes['baseline']:+.3f}")

    print()
    print("Convergence sequence of Delta:")
    for nL in (4, 5, 6, 7):
        slopes_n = {}
        for m in rows:
            cm = chi_max[rows.index(m)][:nL]
            a, _ = np.polyfit(np.log(Ls[:nL]), np.log(cm), 1)
            slopes_n[m] = a
        d = ((slopes_n["full"] - slopes_n["baseline"])
             - (slopes_n["v2_limit"] - slopes_n["baseline"])
             - (slopes_n["v3_limit"] - slopes_n["baseline"]))
        print(f"  Delta_{nL}(L<= {int(Ls[nL-1]):3d}) = {d:+.3f}")


if __name__ == "__main__":
    main()
