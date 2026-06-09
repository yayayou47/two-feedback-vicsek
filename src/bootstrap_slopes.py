"""
Bootstrap 95% confidence intervals on the seven-size FSS slopes
$\\chi_{\\max}\\sim L^a$ (sizes $L=15,22,30,45,64,90,128$) for the
four heavy-tailed modes and the Vicsek-Gaussian reference, and on
the synergy diagnostic $\\Delta_n = a_{\\rm full} + a_{\\rm Cauchy}
- a_{\\rm noise} - a_{\\rm motility}$ for $n \\in \\{4,5,6,7\\}$.
Resampling is with-replacement on the $(L, \\chi_{\\max})$ pairs
($B = 10\\,000$), with bootstrap indices shared across modes so
slope correlations are preserved. Reads the precomputed scan
data; prints slopes, CIs, and bootstrap std (no file written).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def _big_idx(arr, name: str) -> int | None:
    m = list(arr["modes"])
    if isinstance(m[0], bytes):
        m = [x.decode() for x in m]
    return m.index(name) if name in m else None


def main(B: int = 10_000, seed: int = 11) -> None:
    pilot = np.load(DATA / "double_pilot.npz", allow_pickle=True)
    big64 = np.load(DATA / "double_L64.npz", allow_pickle=True)
    fine = np.load(DATA / "double_finegrid.npz", allow_pickle=True)
    big90 = np.load(DATA / "double_L90.npz", allow_pickle=True)
    big128 = np.load(DATA / "double_L128.npz", allow_pickle=True)
    vg = np.load(DATA / "vicsek_gauss_ref.npz", allow_pickle=True)

    ht_modes = [s if isinstance(s, str) else s.decode()
                for s in pilot["modes"]]
    Ls = np.array([15., 22., 30., 45., 64., 90., 128.])

    chi_max = np.zeros((len(ht_modes), len(Ls)))
    chi_std = np.full_like(chi_max, np.nan)
    for im, m in enumerate(ht_modes):
        for iL in range(4):
            cm = pilot["chi"][im, iL]
            cf = fine["chi"][im, iL]
            cs_pilot = pilot["chi_per_seed"][im, iL]
            all_chi = np.concatenate([cm, cf])
            j = int(np.argmax(all_chi))
            chi_max[im, iL] = all_chi[j]
            if j < len(pilot["etas"]):
                chi_std[im, iL] = cs_pilot[j].std(ddof=1)
        cm = big64["chi"][im]
        cf = fine["chi"][im, 4]
        chi_max[im, 4] = np.concatenate([cm, cf]).max()
        chi_max[im, 5] = big90["chi"][_big_idx(big90, m)].max()
        chi_max[im, 6] = big128["chi"][_big_idx(big128, m)].max()

    chi_vg = np.zeros(len(Ls))
    for iL in range(5):
        chi_vg[iL] = vg["chi"][iL].max()
    chi_vg[5] = big90["chi"][_big_idx(big90, "vicsek_gauss")].max()
    chi_vg[6] = big128["chi"][_big_idx(big128, "vicsek_gauss")].max()

    rng = np.random.default_rng(seed)
    bs_idx = rng.choice(len(Ls), size=(B, len(Ls)), replace=True)

    def slope(y, idx):
        if np.unique(idx).size < 2:
            return np.nan
        return np.polyfit(np.log(Ls[idx]), np.log(y[idx]), 1)[0]

    print(f"chi_max(L) and per-seed std at peak (3 seeds, "
          f"NaN where peak in fine-grid eta region):")
    print("  ", "      ".join(f"L={int(L)}" for L in Ls))
    for im, m in enumerate(ht_modes):
        row = "  ".join(f"{c:7.2f}" for c in chi_max[im])
        print(f"  {m:>14s} chi: {row}")
        srow = "  ".join(f"{c:7.2f}" if not np.isnan(c)
                          else "    nan" for c in chi_std[im])
        print(f"  {'':>14s} std: {srow}")
    print(f"  {'Vicsek (Gauss)':>14s} chi: " +
          "  ".join(f"{c:7.2f}" for c in chi_vg))

    print()
    print(f"Slopes and 95% bootstrap CI (B = {B}):")
    point_slopes = {}
    for label, y in [
        ("Vicsek (Gauss)", chi_vg),
        *((m, chi_max[im]) for im, m in enumerate(ht_modes)),
    ]:
        bs = np.array([slope(y, bs_idx[b]) for b in range(B)])
        bs = bs[~np.isnan(bs)]
        pt = np.polyfit(np.log(Ls), np.log(y), 1)[0]
        point_slopes[label] = pt
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {label:>14s}: a = {pt:+.3f}   "
              f"95% CI = [{lo:+.3f}, {hi:+.3f}]   "
              f"bootstrap std = {bs.std(ddof=1):.3f}")

    print()
    print("Synergy Delta_n = a_full + a_cauchy - a_noise - a_motility:")
    i_c, i_n, i_m, i_f = 0, 1, 2, 3
    for n in (4, 5, 6, 7):
        Ls_n = Ls[:n]
        bs_idx_n = rng.choice(n, size=(B, n), replace=True)

        def slope_n(y, idx):
            if np.unique(idx).size < 2:
                return np.nan
            return np.polyfit(np.log(Ls_n[idx]),
                              np.log(y[:n][idx]), 1)[0]

        ds = []
        for b in range(B):
            idx = bs_idx_n[b]
            sl = [slope_n(chi_max[mm], idx)
                  for mm in (i_c, i_n, i_m, i_f)]
            if any(np.isnan(sl)):
                continue
            a_c, a_n_, a_m_, a_f = sl
            ds.append(a_f + a_c - a_n_ - a_m_)
        ds = np.array(ds)
        pt_slopes = [
            np.polyfit(np.log(Ls_n), np.log(chi_max[mm, :n]), 1)[0]
            for mm in (i_c, i_n, i_m, i_f)
        ]
        pt = pt_slopes[3] + pt_slopes[0] - pt_slopes[1] - pt_slopes[2]
        lo, hi = np.percentile(ds, [2.5, 97.5])
        print(f"  Delta_{n}: pt = {pt:+.3f}   "
              f"95% CI = [{lo:+.3f}, {hi:+.3f}]   "
              f"bootstrap std = {ds.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
