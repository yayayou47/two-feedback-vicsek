"""
Analysis of the two-dimensional decoupled-sigmoid sweep.

Reads ``data/double_decoupled_2d_nocone.npz`` and produces:

  * ``data/decoupled_2d_summary.npz``: paired-diff gap Delta_g
    at r ~= 0.62 per cell, SE, bootstrap 95% CI, sign-of-gap
    flag, plus the inner (r < 1.5) and outer (r > 3) profile
    averages.
  * ``figures/fig_double_decoupled_2d.pdf``: a two-panel figure
    (heatmap of Delta_g on the (n_v, n_alpha) grid, and a
    scatter of Delta_g vs (n_v - n_alpha) showing the timing
    asymmetry).
  * a printed summary table for inclusion in Sec. 3.12.

Statistical convention: the paired diff is
``g_dense_full[seed] - g_dense_motility[seed]`` at every seed,
and the seed-level SE is reported. Bootstrap CI is at 95 %
with B = 10 000 paired-bootstrap resamples.

Run from version4/ as:
  ../.venv/bin/python flock_simulator/scripts/analyse_decoupled_2d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import style as st  # type: ignore[import]

st.apply()

DATA = ROOT / "data"
FIG = ROOT / "figures"


def main() -> None:
    src = DATA / "double_decoupled_2d_nocone.npz"
    z = np.load(src, allow_pickle=True)
    n_v = z["n_v_grid"]
    n_a = z["n_a_grid"]
    seeds = z["seeds"]
    r_centers = z["r_centers"]
    gr_dense = z["gr_dense"]          # (n_v, n_a, n_seed, n_bins)
    gr_dense_mot = z["gr_dense_motility"]  # (n_seed, n_bins)
    n_seeds = gr_dense.shape[2]
    j = int(np.argmin(np.abs(r_centers - 0.62)))
    print(f"r-bin = {r_centers[j]:.3f}, {n_seeds} seeds, "
          f"grid {len(n_v)} x {len(n_a)}")

    # Paired diff per (cell, seed).
    diff = gr_dense[:, :, :, j] - gr_dense_mot[None, None, :, j]
    mean_g = np.nanmean(diff, axis=2)
    se_g = (np.nanstd(diff, axis=2, ddof=1)
            / np.sqrt(np.sum(np.isfinite(diff), axis=2)))
    z_g = mean_g / np.maximum(se_g, 1e-9)

    # Bootstrap CI per cell.
    rng = np.random.default_rng(11)
    B = 10_000
    bs_idx = rng.choice(n_seeds, size=(B, n_seeds), replace=True)
    ci_lo = np.zeros_like(mean_g)
    ci_hi = np.zeros_like(mean_g)
    for iv in range(len(n_v)):
        for ia in range(len(n_a)):
            samples = diff[iv, ia]
            bs = samples[bs_idx].mean(axis=1)
            ci_lo[iv, ia], ci_hi[iv, ia] = np.percentile(bs, [2.5, 97.5])

    # Inner / outer profile means.
    inner_mask = r_centers < 1.5
    outer_mask = r_centers > 3.0
    inner_diff = gr_dense[:, :, :, inner_mask] - gr_dense_mot[None, None, :, inner_mask]
    outer_diff = gr_dense[:, :, :, outer_mask] - gr_dense_mot[None, None, :, outer_mask]
    inner_mean = np.nanmean(inner_diff, axis=(2, 3))
    outer_mean = np.nanmean(outer_diff, axis=(2, 3))

    # Save summary.
    out = DATA / "decoupled_2d_summary.npz"
    np.savez_compressed(
        out,
        n_v=n_v, n_a=n_a,
        mean_g=mean_g, se_g=se_g, z_g=z_g,
        ci_lo=ci_lo, ci_hi=ci_hi,
        inner_mean=inner_mean, outer_mean=outer_mean,
    )
    print(f"saved {out.name}")

    # Headline table.
    print(f"\nDelta_g(r ~ 0.62), paired diff vs motility-only, "
          f"10-seed mean +/- SE (z):")
    print(f"{'n_v\\n_a':>10s}  " +
          "  ".join(f"{na:>14.0f}" for na in n_a))
    for iv, nsv in enumerate(n_v):
        cells = []
        for ia in range(len(n_a)):
            cells.append(f"{mean_g[iv, ia]:+.3f}+-{se_g[iv, ia]:.3f}"
                         f"({z_g[iv, ia]:+.1f})")
        print(f"{nsv:>10.0f}  " + "  ".join(cells))

    # Sign-flip line: cells with CI strictly above 0 vs strictly below.
    above = (ci_lo > 0).astype(int)
    below = (ci_hi < 0).astype(int)
    n_above = int(above.sum())
    n_below = int(below.sum())
    n_bracket = int(((ci_lo <= 0) & (ci_hi >= 0)).sum())
    print(f"\nCell classification (95% bootstrap CI):")
    print(f"  cells with Delta_g > 0 (CI excludes 0): {n_above}")
    print(f"  cells with Delta_g < 0 (CI excludes 0): {n_below}")
    print(f"  cells bracketing 0: {n_bracket}")

    # Diagonal (shared-sigmoid) is n_v == n_a.
    diag_idx = [(i, i) for i in range(len(n_v)) if i < len(n_a)]
    diag_vals = [mean_g[i, j_] for i, j_ in diag_idx]
    diag_n = [(int(n_v[i]), int(n_a[j_])) for i, j_ in diag_idx]
    print(f"\nDiagonal (shared sigmoid n_v = n_a):")
    for (i, j_), val in zip(diag_idx, diag_vals):
        print(f"  n_star = {int(n_v[i])}: Delta_g = {val:+.3f}")

    # Variable importance: which threshold controls the gap?
    pairs = []
    for iv, nsv in enumerate(n_v):
        for ia, nsa in enumerate(n_a):
            pairs.append((float(nsv), float(nsa),
                          mean_g[iv, ia], se_g[iv, ia]))
    pairs = np.array(pairs)
    xv, xa, y, sigma = pairs[:, 0], pairs[:, 1], pairs[:, 2], pairs[:, 3]

    def r2(yhat, y):
        return 1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    cv = np.polyfit(xv, y, 1)
    yh_v = cv[0] * xv + cv[1]
    print(f"\nFit Delta_g = {cv[0]:+.4f} * n_v + {cv[1]:+.3f}  "
          f"(R^2 = {r2(yh_v, y):.3f})")
    ca = np.polyfit(xa, y, 1)
    yh_a = ca[0] * xa + ca[1]
    print(f"Fit Delta_g = {ca[0]:+.4f} * n_a + {ca[1]:+.3f}  "
          f"(R^2 = {r2(yh_a, y):.3f})")
    cd = np.polyfit(xv - xa, y, 1)
    yh_d = cd[0] * (xv - xa) + cd[1]
    print(f"Fit Delta_g = {cd[0]:+.4f} * (n_v-n_a) + {cd[1]:+.3f}  "
          f"(R^2 = {r2(yh_d, y):.3f})")
    A = np.column_stack([xv, xa, np.ones_like(xv)])
    cb, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh_b = A @ cb
    print(f"Fit Delta_g = {cb[0]:+.4f}*n_v + {cb[1]:+.4f}*n_a + {cb[2]:+.3f}  "
          f"(R^2 = {r2(yh_b, y):.3f})  bivariate")
    # n_v is dominant; use it for panel (b).
    a, b_int = cv[0], cv[1]
    r2_v = r2(yh_v, y)

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(st.DOUBLE_COL[0], 3.0))

    # Panel (a): heatmap of Delta_g on (n_v, n_a).
    ax = axes[0]
    vmax = float(np.nanmax(np.abs(mean_g)))
    im = ax.imshow(
        mean_g, origin="lower", aspect="auto",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=[n_a.min() - 0.5, n_a.max() + 0.5,
                n_v.min() - 0.5, n_v.max() + 0.5],
    )
    ax.set_xticks(n_a)
    ax.set_yticks(n_v)
    ax.set_xlabel(r"$n_\star^\alpha$ (noise-shape threshold)")
    ax.set_ylabel(r"$n_\star^v$ (motility threshold)")
    ax.set_title(r"(a) $\Delta g(r\!\simeq\!0.6)$ vs motility, $L = 30$, 10 seeds",
                 fontsize=8, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\Delta g$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    # Cell labels with z-score in parentheses.
    for iv, nsv in enumerate(n_v):
        for ia, nsa in enumerate(n_a):
            txt_color = "white" if abs(mean_g[iv, ia]) > 0.6 * vmax else "black"
            ax.text(nsa, nsv,
                    f"{mean_g[iv, ia]:+.2f}",
                    ha="center", va="center",
                    fontsize=6.5, color=txt_color)
    # Diagonal line (shared-sigmoid locus).
    diag_line_x = np.linspace(n_a.min(), n_a.max(), 100)
    ax.plot(diag_line_x, diag_line_x, ":", color="grey", lw=0.6,
            alpha=0.8)

    # Panel (b): Delta_g vs n_v with linear fit, n_a as color.
    ax = axes[1]
    palette = {1.0: st.WONG["blue"], 2.0: st.WONG["sky"],
               3.0: st.WONG["green"], 5.0: st.WONG["orange"],
               8.0: st.WONG["vermil"]}
    for iv in range(len(n_v)):
        for ia in range(len(n_a)):
            ax.errorbar(
                xv[iv * len(n_a) + ia], y[iv * len(n_a) + ia],
                yerr=sigma[iv * len(n_a) + ia],
                fmt="o", markersize=5,
                color=palette[float(n_a[ia])],
                ecolor="grey", elinewidth=0.6, capsize=2,
                markeredgecolor="black", markeredgewidth=0.3,
            )
    xf = np.linspace(0.5, 8.5, 200)
    ax.plot(xf, a * xf + b_int, "-", color="grey", lw=0.8,
            label=fr"fit $-{abs(a):.3f}\,n_\star^v + {b_int:+.2f}$")
    ax.axhline(0, color="black", lw=0.4, ls="--", alpha=0.4)
    ax.set_xlabel(r"$n_\star^v$ (motility threshold)")
    ax.set_ylabel(r"$\Delta g(r\!\simeq\!0.6)$")
    ax.set_title(fr"(b) Motility threshold dominates, $R^2 = {r2_v:.2f}$ on $n_\star^v$ alone",
                 fontsize=8, loc="left")
    for v, c in palette.items():
        ax.scatter([], [], color=c, s=20, edgecolor="black",
                   linewidth=0.3, label=fr"$n_\star^\alpha = {int(v)}$")
    ax.legend(fontsize=6.5, loc="lower left", frameon=False, ncol=2)

    fig.tight_layout()
    out_pdf = FIG / "fig_double_decoupled_2d.pdf"
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    print(f"\nsaved figure: {out_pdf.name}")


if __name__ == "__main__":
    main()
