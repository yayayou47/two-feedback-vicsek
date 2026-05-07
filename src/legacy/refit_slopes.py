"""
Refit FSS slopes for the four-mode comparison after the L = 64
controlled run lands. Concatenates double_pilot.npz (L in {15, 22,
30, 45}) with double_L64.npz, prints the 5-size chi_max(L) per
mode, the log-log slope, and the synergy diagnostic Delta.

Run after data/double_L64.npz exists.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def main():
    pilot = np.load(DATA / "double_pilot.npz", allow_pickle=True)
    big = np.load(DATA / "double_L64.npz", allow_pickle=True)

    Ls_pilot = pilot["Ls"]
    L64 = float(big["L"])
    Ls = np.concatenate([Ls_pilot, [L64]])
    modes = [str(s) if isinstance(s, str) else s.decode()
             for s in pilot["modes"]]
    chi_pilot = pilot["chi"]                # [n_mode, n_L, n_eta]
    chi_big = big["chi"]                    # [n_mode, n_eta]

    print(f"Sizes: {Ls}")
    print()
    chi_max_5L = np.zeros((len(modes), len(Ls)))
    for im, m in enumerate(modes):
        cm = np.concatenate([chi_pilot[im].max(axis=1),
                              [chi_big[im].max()]])
        chi_max_5L[im] = cm
        a, b = np.polyfit(np.log(Ls), np.log(cm), 1)
        print(f"{m:10s}  chi_max= "
              + "  ".join(f"{v:7.2f}" for v in cm)
              + f"   slope= {a:+.3f}")

    print()
    slopes = {}
    for im, m in enumerate(modes):
        slopes[m], _ = np.polyfit(np.log(Ls),
                                   np.log(chi_max_5L[im]), 1)

    delta = ((slopes["full"] - slopes["baseline"])
             - (slopes["v2_limit"] - slopes["baseline"])
             - (slopes["v3_limit"] - slopes["baseline"]))
    print(f"Delta(5 sizes) = {delta:+.3f}")
    print(f"  full - baseline = {slopes['full'] - slopes['baseline']:+.3f}")
    print(f"  v2 - baseline   = {slopes['v2_limit'] - slopes['baseline']:+.3f}")
    print(f"  v3 - baseline   = {slopes['v3_limit'] - slopes['baseline']:+.3f}")
    sign = ("SYNERGIC" if delta > 0
            else "ANTAGONISTIC" if delta < 0 else "ADDITIVE")
    print(f"  -> {sign}")

    # Re-run with pilot 4 sizes only, for direct comparison
    print()
    print("(reference: pilot 4-size slopes for comparison)")
    slopes4 = {}
    for im, m in enumerate(modes):
        cm4 = chi_max_5L[im, :-1]
        a4, _ = np.polyfit(np.log(Ls_pilot), np.log(cm4), 1)
        slopes4[m] = a4
        print(f"  {m:10s}  4-size slope= {a4:+.3f}    "
              f"5-size slope= {slopes[m]:+.3f}    "
              f"shift= {slopes[m] - a4:+.3f}")
    delta4 = ((slopes4["full"] - slopes4["baseline"])
              - (slopes4["v2_limit"] - slopes4["baseline"])
              - (slopes4["v3_limit"] - slopes4["baseline"]))
    print(f"  Delta(4 sizes)= {delta4:+.3f}    "
          f"Delta(5 sizes)= {delta:+.3f}")

    # Headline numbers at L=64 specifically
    print()
    print("L=64 headline (largest size):")
    etas = pilot["etas"]
    for im, m in enumerate(modes):
        ie = int(np.argmax(chi_big[im]))
        print(f"  {m:10s}  chi_max={chi_big[im, ie]:7.2f}  @eta={etas[ie]:.3f}  "
              f"phi={big['phi'][im, ie]:.3f}  "
              f"U4_min={big['U4'][im].min():.3f}  "
              f"s_sep_max={big['s_sep'][im].max():.2f}")


if __name__ == "__main__":
    main()
