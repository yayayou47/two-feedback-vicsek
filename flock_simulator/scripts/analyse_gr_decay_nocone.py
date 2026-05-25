"""Compute the dense-quartile gap profile $\\Delta g(r)$ between
the double-adaptive model and the motility-only ablation across
the full alignment-zone range $r \\in [0.5, 6]$, on the four
no-cone sizes $L \\in \\{30, 64, 90, 128\\}$.

Reads:
  data/double_gr_hs_nocone.npz            (L = 30, 4 modes)
  data/double_micro_hs_largeL_nocone.npz  (L = 64, 90, 2 modes)
  data/double_micro_hs_L128_nocone.npz    (L = 128, 2 modes)

Writes:
  data/double_gr_decay_nocone.npz with the gap mean, SE, z-score
  profile per size.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def gap_profile(gr_full, gr_motility):
    diff = gr_full - gr_motility
    ok_per_bin = np.isfinite(diff)
    n_per_bin = ok_per_bin.sum(axis=0)
    gap = np.where(n_per_bin > 1, np.nanmean(diff, axis=0), np.nan)
    se = np.where(n_per_bin > 1,
                   np.nanstd(diff, axis=0, ddof=1)
                   / np.sqrt(np.maximum(n_per_bin, 1)), np.nan)
    z = np.where(se > 0, gap / np.maximum(se, 1e-9), 0.0)
    return gap, se, z


def main() -> None:
    gr30 = np.load(DATA / "double_gr_hs_nocone.npz", allow_pickle=True)
    labels30 = [str(s) if isinstance(s, str) else s.decode()
                for s in gr30["labels"]]
    i_mot30 = labels30.index("v3_limit")
    i_full30 = labels30.index("full")
    r_centers = gr30["r_centers"]
    n_bins = len(r_centers)

    gr_big = np.load(DATA / "double_micro_hs_largeL_nocone.npz",
                      allow_pickle=True)
    big_modes = [str(s) if isinstance(s, str) else s.decode()
                 for s in gr_big["modes"]]
    i_mot_big = big_modes.index("motility")
    i_full_big = big_modes.index("full")

    gr128 = np.load(DATA / "double_micro_hs_L128_nocone.npz",
                     allow_pickle=True)
    modes128 = [str(s) if isinstance(s, str) else s.decode()
                for s in gr128["modes"]]
    i_mot128 = modes128.index("motility")
    i_full128 = modes128.index("full")

    Ls_all = [30.0, 64.0, 90.0, 128.0]
    gap_per_L = np.zeros((len(Ls_all), n_bins))
    se_per_L = np.zeros((len(Ls_all), n_bins))
    z_per_L = np.zeros((len(Ls_all), n_bins))

    g, s, z = gap_profile(
        gr30["gr_dense_per_seed"][i_full30],
        gr30["gr_dense_per_seed"][i_mot30],
    )
    gap_per_L[0] = g; se_per_L[0] = s; z_per_L[0] = z

    for iL in (0, 1):
        g, s, z = gap_profile(
            gr_big["gr_dense_per_seed"][i_full_big, iL],
            gr_big["gr_dense_per_seed"][i_mot_big, iL],
        )
        gap_per_L[iL + 1] = g
        se_per_L[iL + 1] = s
        z_per_L[iL + 1] = z

    g, s, z = gap_profile(
        gr128["gr_dense_per_seed"][i_full128],
        gr128["gr_dense_per_seed"][i_mot128],
    )
    gap_per_L[3] = g; se_per_L[3] = s; z_per_L[3] = z

    out = DATA / "double_gr_decay_nocone.npz"
    np.savez_compressed(
        out,
        Ls=np.array(Ls_all),
        r_centers=r_centers,
        gap=gap_per_L,
        se=se_per_L,
        z=z_per_L,
    )
    print(f"saved {out.name}")

    print(f"\ninner mean (r < 1.5) and outer mean (r > 3) per L:")
    print(f"  L      inner      outer")
    in_mask = r_centers < 1.5
    out_mask = r_centers > 3.0
    for iL, L in enumerate(Ls_all):
        inner = float(np.nanmean(gap_per_L[iL, in_mask]))
        outer = float(np.nanmean(gap_per_L[iL, out_mask]))
        print(f"  {int(L):>4d}  {inner:+.3f}  {outer:+.3f}")


if __name__ == "__main__":
    main()
