"""
Compute the dense-quartile gap profile $\\Delta g(r)$ between the
double-adaptive model and the motility-only ablation across the
full alignment-zone range, $r \\in [0.5, 6]$, on the four tested
sizes $L \\in \\{30, 64, 90, 128\\}$. Uses the existing per-seed
$g(r)$ arrays from the high-statistics scans; no new simulation
is needed.

Output: data/double_gr_decay.npz with the gap mean and SE per
size and per r-bin, plus the z-score profile.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def gap_profile(gr_dense_per_seed_full, gr_dense_per_seed_motility):
    """Per-bin gap mean, SE, z. Inputs are (n_seeds, n_bins)."""
    diff = gr_dense_per_seed_full - gr_dense_per_seed_motility
    ok_per_bin = np.isfinite(diff)
    n_per_bin = ok_per_bin.sum(axis=0)
    gap = np.where(n_per_bin > 1,
                    np.nanmean(diff, axis=0), np.nan)
    se = np.where(n_per_bin > 1,
                   np.nanstd(diff, axis=0, ddof=1)
                   / np.sqrt(np.maximum(n_per_bin, 1)), np.nan)
    z = np.where(se > 0, gap / np.maximum(se, 1e-9), 0.0)
    return gap, se, z


def main():
    # L = 30 from clusters/g(r) HS run.
    gr30 = np.load(DATA / "double_gr_hs.npz", allow_pickle=True)
    labels30 = [str(s) if isinstance(s, str) else s.decode()
                for s in gr30["labels"]]
    i_mot30 = labels30.index("v3_limit")
    i_full30 = labels30.index("full")
    r_centers = gr30["r_centers"]
    n_bins = len(r_centers)

    # L in {64, 90} from micro_hs_largeL.
    gr_big = np.load(DATA / "double_micro_hs_largeL.npz",
                      allow_pickle=True)
    big_modes = [str(s) if isinstance(s, str) else s.decode()
                 for s in gr_big["modes"]]
    i_mot_big = big_modes.index("motility")
    i_full_big = big_modes.index("full")
    Ls_big = gr_big["Ls"]

    # L = 128 from L128 run.
    gr128 = np.load(DATA / "double_micro_hs_L128.npz",
                     allow_pickle=True)
    modes128 = [str(s) if isinstance(s, str) else s.decode()
                for s in gr128["modes"]]
    i_mot128 = modes128.index("motility")
    i_full128 = modes128.index("full")

    Ls_all = [30.0, 64.0, 90.0, 128.0]
    gap_per_L = np.zeros((len(Ls_all), n_bins))
    se_per_L = np.zeros((len(Ls_all), n_bins))
    z_per_L = np.zeros((len(Ls_all), n_bins))

    # L = 30
    g, s, z = gap_profile(
        gr30["gr_dense_per_seed"][i_full30],
        gr30["gr_dense_per_seed"][i_mot30],
    )
    gap_per_L[0] = g; se_per_L[0] = s; z_per_L[0] = z

    # L = 64 and L = 90 from micro_hs_largeL  (shape: n_modes, n_L, n_seeds, n_bins)
    for iL_out, iL in enumerate([0, 1]):
        g, s, z = gap_profile(
            gr_big["gr_dense_per_seed"][i_full_big, iL],
            gr_big["gr_dense_per_seed"][i_mot_big, iL],
        )
        gap_per_L[iL_out + 1] = g
        se_per_L[iL_out + 1] = s
        z_per_L[iL_out + 1] = z

    # L = 128
    g, s, z = gap_profile(
        gr128["gr_dense_per_seed"][i_full128],
        gr128["gr_dense_per_seed"][i_mot128],
    )
    gap_per_L[3] = g; se_per_L[3] = s; z_per_L[3] = z

    np.savez_compressed(
        DATA / "double_gr_decay.npz",
        Ls=np.array(Ls_all),
        r_centers=r_centers,
        gap=gap_per_L,
        se=se_per_L,
        z=z_per_L,
    )

    print(f"\nGap profile $\\Delta g(r)$ across L:")
    print(f"r:        " + " ".join(f"{r:5.2f}" for r in r_centers))
    for iL, L in enumerate(Ls_all):
        print(f"L={int(L):3d} g:  "
              + " ".join(f"{v:+.3f}" for v in gap_per_L[iL]))
        print(f"      SE:  "
              + " ".join(f"{v:.3f}" for v in se_per_L[iL]))
        print(f"      z:   "
              + " ".join(f"{v:+.1f}" for v in z_per_L[iL]))
    print()

    # Headline numbers: gap at the inner alignment zone vs gap at
    # r >= 3 (well beyond the alignment cutoff R_a = 0.7).
    inner = (r_centers > 0.5) & (r_centers < 1.5)
    outer = r_centers > 3.0
    print("Inner (r in (0.5, 1.5)) and outer (r > 3) means:")
    for iL, L in enumerate(Ls_all):
        g_in = np.nanmean(gap_per_L[iL, inner])
        g_out = np.nanmean(gap_per_L[iL, outer])
        z_in = np.nanmean(z_per_L[iL, inner])
        z_out = np.nanmean(z_per_L[iL, outer])
        print(f"  L={int(L):3d}: <g>_inner={g_in:+.3f} (z~{z_in:+.1f})  "
              f"<g>_outer={g_out:+.3f} (z~{z_out:+.1f})")
    print(f"\nsaved: {DATA / 'double_gr_decay.npz'}")


if __name__ == "__main__":
    main()
