"""Build the merged decoupled-sigmoid figure (former Figs. 14/15/16)
as a single 2x3 panel grid.

Row 1 groups the three Delta_g heatmaps: (a) Delta_g on (n_a, n_v)
at L=30, (d) metric-kernel and (e) topological-kernel maps at
sigma=2.22. They share one colour scale, a single y-label on the
left and a single vertical Delta_g colorbar at the right of the row.
Row 2 groups the curves: (b) Delta_g vs n_v colour-coded by n_a with
the motility-threshold fit, (c) Delta_g vs n_v at four densities,
and (f) critical threshold n_v_crit vs sigma with the linear fit.

Output: figures/fig_double_decoupled_2d.pdf (overwrites the old
two-panel file so the manuscript include keeps working).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import style as st  # noqa: E402

# Cream figure background, neutral (white) panel interiors, framed axes.
plt.rcParams.update({
    "figure.facecolor": st.CREAM,
    "savefig.facecolor": st.CREAM,
    "savefig.edgecolor": st.CREAM,
    "axes.facecolor": st.WHITE,
})

DATA = ROOT / "data"
FIG = ROOT / "figures"

# Navy-mauve-vermillion snapshot colormap, shared with Figs. 2 and 4.
# Used here with a symmetric vmin=-vmax so Delta_g = 0 lands on the
# mauve midpoint.
R_CMP = LinearSegmentedColormap.from_list(
    "snapshot_speed", ["#1f4ea1", "#7e3a8a", "#D55E00"])


def _heatmap(ax, n_a, n_v, data, title, vmax, show_ylabel=True):
    AA, VV = np.meshgrid(n_a, n_v)
    pm = ax.pcolormesh(AA, VV, data, cmap=R_CMP, vmin=-vmax, vmax=vmax,
                       shading="gouraud")
    ax.set_xlabel(r"$n_\star^\alpha$")
    if show_ylabel:
        ax.set_ylabel(r"$n_\star^v$")
    ax.set_title(title, fontsize=8, loc="left")
    diag = np.linspace(n_a.min(), n_a.max(), 100)
    ax.plot(diag, diag, ":", color="grey", lw=0.6, alpha=0.8)
    return pm


def main() -> None:
    # --- data ---
    zd = np.load(DATA / "decoupled_2d_summary.npz", allow_pickle=True)
    mean_g = zd["mean_g"]
    se_g = zd["se_g"]
    n_a = zd["n_a"]
    n_v = zd["n_v"]

    zt = np.load(DATA / "decoupled_2d_topo_summary.npz", allow_pickle=True)
    m_mean = zt["metric_mean"]
    t_mean = zt["topo_mean"]
    n_a_t = zt["n_a"]
    n_v_t = zt["n_v"]

    zs = np.load(DATA / "sigma_sweep_summary.npz", allow_pickle=True)
    sigmas = zs["sigmas"]
    crit = zs["n_v_crit"]
    crit_se = zs["n_v_crit_se"]
    cc = float(zs["scaling_c"])
    dd = float(zs["scaling_d"])
    R2s = float(zs["scaling_R2"])
    A = float(zs["A"])

    # sigma-sweep Delta_g(n_v) series, recomputed from raw gr data.
    swp = np.load(DATA / "double_sigma_sweep_nocone.npz", allow_pickle=True)
    twod = np.load(DATA / "double_decoupled_2d_nocone.npz", allow_pickle=True)
    r_centers = swp["r_centers"]
    j = int(np.argmin(np.abs(r_centers - 0.62)))
    sig_swp = swp["sigmas"]
    nvg = swp["n_v_grid"]
    gr_full = swp["gr_dense_full"]            # (n_s, n_v, n_seed, n_bins)
    gr_mot = swp["gr_dense_motility"]         # (n_s, n_seed, n_bins)
    diff = gr_full[..., j] - gr_mot[:, None, :, j]
    mean_swp = np.nanmean(diff, axis=2)       # (n_s, n_v)
    se_swp = (np.nanstd(diff, axis=2, ddof=1)
              / np.sqrt(np.sum(np.isfinite(diff), axis=2)))
    twod_na = twod["n_a_grid"]
    ia3 = int(np.where(twod_na == 3.0)[0][0])
    twod_nv = twod["n_v_grid"]
    gr_full_2d = twod["gr_dense"][:, ia3, :, j]
    gr_mot_2d = twod["gr_dense_motility"][:, j]
    diff_2d = gr_full_2d - gr_mot_2d[None, :]
    mean_2d = np.nanmean(diff_2d, axis=1)
    se_2d = (np.nanstd(diff_2d, axis=1, ddof=1)
             / np.sqrt(np.sum(np.isfinite(diff_2d), axis=1)))
    series = [(float(s), nvg, mean_swp[i], se_swp[i])
              for i, s in enumerate(sig_swp)]
    series.append((2.22, twod_nv, mean_2d, se_2d))
    series.sort(key=lambda t: t[0])

    # --- figure: row 0 = heatmaps (a, d, e); row 1 = curves (b, c, f) ---
    # Square panels packed tight; a slim spacer column carries the one
    # heatmap colorbar so the three data columns stay aligned across
    # both rows.
    fig = plt.figure(figsize=(st.DOUBLE_COL[0], 4.7), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.015, h_pad=0.015,
                                    wspace=0.015, hspace=0.03)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)]
                     for r in range(2)], dtype=object)
    cax = fig.add_subplot(gs[0, 3])
    vmax = max(float(np.nanmax(np.abs(mean_g))),
               float(np.nanmax(np.abs(m_mean))),
               float(np.nanmax(np.abs(t_mean))))

    # Row 0: the three Delta_g heatmaps on one shared colour scale,
    # a single y-label on the left and one vertical colorbar at right.
    pm = _heatmap(axes[0, 0], n_a, n_v, mean_g,
                  r"(a) $\Delta g(r\!\simeq\!0.6)$, $L = 30$", vmax,
                  show_ylabel=True)
    _heatmap(axes[0, 1], n_a_t, n_v_t, m_mean,
             r"(d) metric alignment", vmax, show_ylabel=False)
    _heatmap(axes[0, 2], n_a_t, n_v_t, t_mean,
             r"(e) topological ($k = 4$)", vmax, show_ylabel=False)
    cbar = fig.colorbar(pm, cax=cax)
    cbar.set_label(r"$\Delta g$", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    # (b) Delta_g vs n_v, colour-coded by n_a.
    ax = axes[1, 0]
    pal = {1.0: st.WONG["blue"], 2.0: st.WONG["sky"],
           3.0: st.WONG["green"], 5.0: st.WONG["orange"],
           8.0: st.WONG["vermil"]}
    xv, yv = [], []
    for iv, nv in enumerate(n_v):
        for ia, na in enumerate(n_a):
            ax.errorbar(nv, mean_g[iv, ia], yerr=se_g[iv, ia],
                        fmt="o", ms=4.5, color=pal[float(na)],
                        ecolor="grey", elinewidth=0.6, capsize=2,
                        markeredgecolor="black", markeredgewidth=0.3)
            xv.append(float(nv)); yv.append(float(mean_g[iv, ia]))
    a_fit, b_fit = np.polyfit(xv, yv, 1)
    xf = np.linspace(0.5, 8.5, 100)
    ax.plot(xf, a_fit * xf + b_fit, "-", color="grey", lw=0.8)
    ax.axhline(0, color="black", lw=0.4, ls="--", alpha=0.4)
    ax.set_xlabel(r"$n_\star^v$")
    ax.set_ylabel(r"$\Delta g(r\!\simeq\!0.6)$")
    ax.set_title(r"(b) motility threshold dominates", fontsize=8,
                 loc="left")
    for v, c in pal.items():
        ax.scatter([], [], color=c, s=18, edgecolor="black",
                   linewidth=0.3, label=fr"$n_\star^\alpha = {int(v)}$")
    ax.legend(fontsize=6, loc="upper right", frameon=False, ncol=2)

    # (c) Delta_g(n_v) at four densities.
    ax = axes[1, 1]
    spal = {1.0: st.WONG["blue"], 1.5: st.WONG["sky"],
            2.22: st.WONG["green"], 3.0: st.WONG["vermil"]}
    for sigma, nvs, ms, ses in series:
        col = spal.get(round(sigma, 2), st.WONG["black"])
        ax.errorbar(nvs, ms, yerr=ses, fmt="o", ms=4,
                    color=col, ecolor="grey", elinewidth=0.6, capsize=2,
                    markeredgecolor="black", markeredgewidth=0.3,
                    label=fr"$\rho_0 = {sigma:g}$")
        aa, bb = np.polyfit(nvs.astype(float), ms, 1,
                            w=1.0 / np.maximum(ses, 1e-6) ** 2)
        xf2 = np.linspace(0.5, 12.5, 100)
        ax.plot(xf2, aa * xf2 + bb, "-", color=col, lw=0.8, alpha=0.7)
    ax.axhline(0, color="black", lw=0.4, ls="--", alpha=0.4)
    ax.set_xlabel(r"$n_\star^v$")
    ax.set_ylabel(r"$\Delta g(r\!\simeq\!0.6)$")
    ax.set_title(r"(c) four densities", fontsize=8, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="upper right")

    # (f) n_v_crit vs sigma.
    ax = axes[1, 2]
    for sigma, cr, cse in zip(sigmas, crit, crit_se):
        col = spal.get(round(float(sigma), 2), st.WONG["black"])
        ax.errorbar(sigma, cr, yerr=cse, fmt="o", ms=6, color=col,
                    ecolor="grey", elinewidth=0.7, capsize=3,
                    markeredgecolor="black", markeredgewidth=0.4)
    xf3 = np.linspace(0.7, 3.3, 100)
    ax.plot(xf3, cc * xf3 + dd, "-", color="grey", lw=1.0,
            label=(fr"$n^{{v,{{\rm crit}}}}_\star = {cc:+.2f}\,\rho_0 "
                   fr"{dd:+.2f}$ ($R^2 = {R2s:.2f}$)"))
    ax.plot(xf3, A * xf3, ":", color="black", lw=0.8, alpha=0.6,
            label=fr"$A\,\rho_0 = {A:.2f}\,\rho_0$")
    ax.set_xlabel(r"$\rho_0$ (mean density)")
    ax.set_ylabel(r"$n^{v,\,{\rm crit}}_\star$")
    ax.set_title(r"(f) density scaling", fontsize=8, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="lower left")

    # Square every panel (y size matched to x size).
    for a in axes.ravel():
        a.set_box_aspect(1)

    out = FIG / "fig_double_decoupled_2d.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
