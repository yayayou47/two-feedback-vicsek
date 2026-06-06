"""Figure renderer for the double-adaptive Vicsek paper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import style

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIGS = HERE.parent / "figures"
FIGS.mkdir(exist_ok=True)

style.apply()
# Cream figure background with neutral (white) panel interiors and
# framed axes, applied uniformly to every figure in this module.
plt.rcParams.update({
    "figure.facecolor": style.CREAM,
    "savefig.facecolor": style.CREAM,
    "savefig.edgecolor": style.CREAM,
    "axes.facecolor": style.WHITE,
})


def _save(fig, name):
    out = FIGS / name
    # dpi controls the resolution of any rasterized=True artists (e.g.
    # the dense large-L quiver layers) embedded in the vector PDF.
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


SNAPSHOT_BLUE = "#1f4ea1"  # v1 navy: best contrast on cream at print size

# Speed-encoding colormap for snapshots: navy (slow, trapped in
# the dense phase) -> vermillion (fast, free in the dilute phase).
# Both endpoints read well on cream and stay legible at print
# size; the midpoint mauve also stays distinct from the cream BG.
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap

SPEED_CMAP = _LinearSegmentedColormap.from_list(
    "snapshot_speed", [SNAPSHOT_BLUE, "#7e3a8a", "#D55E00"])
SPEED_VMIN = 0.005
SPEED_VMAX = 0.05


def _cream_panel(ax):
    """v1 snapshot aesthetic: cream face on the panel."""
    ax.set_facecolor(style.CREAM)


def _zoom_inset(ax, xs, ys, ths, L, zoom_size=5.0, loc="upper right",
                inset_frac=0.38, arrow_len=0.45, speeds=None,
                bg=None):
    """Add a square zoom inset showing a clearly resolved sub-region.

    The inset is anchored in `loc` corner of the parent axes and shows
    a `zoom_size`-wide square centred on a dense patch of the snapshot
    (chosen automatically as the densest sub-window). Arrows in the
    inset use the same colour and geometry as v1 fig_snapshots."""
    from matplotlib.patches import Rectangle
    H, xe, ye = np.histogram2d(xs, ys, bins=10, range=[[0, L], [0, L]])
    ix, iy = np.unravel_index(np.argmax(H), H.shape)
    cx = 0.5 * (xe[ix] + xe[ix + 1])
    cy = 0.5 * (ye[iy] + ye[iy + 1])
    half = zoom_size / 2
    x0 = max(0.0, min(L - zoom_size, cx - half))
    y0 = max(0.0, min(L - zoom_size, cy - half))
    x1 = x0 + zoom_size
    y1 = y0 + zoom_size

    # Anchor (x, y, w, h) in parent-axes coords for upper-right /
    # upper-left corners (we only use upper-right here).
    if loc == "upper right":
        anchor = (1.0 - inset_frac - 0.005, 1.0 - inset_frac - 0.005,
                  inset_frac, inset_frac)
    else:
        anchor = (0.005, 1.0 - inset_frac - 0.005,
                  inset_frac, inset_frac)
    iax = ax.inset_axes(anchor)
    iax.set_facecolor(style.CREAM if bg is None else bg)
    sel = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    if speeds is None:
        iax.quiver(
            xs[sel], ys[sel], np.cos(ths[sel]), np.sin(ths[sel]),
            color=SNAPSHOT_BLUE,
            scale=1.0 / arrow_len, scale_units="xy",
            angles="xy", width=0.012,
            headwidth=3.5, headlength=4.0,
        )
    else:
        iax.quiver(
            xs[sel], ys[sel], np.cos(ths[sel]), np.sin(ths[sel]),
            speeds[sel],
            cmap=SPEED_CMAP, clim=(SPEED_VMIN, SPEED_VMAX),
            scale=1.0 / arrow_len, scale_units="xy",
            angles="xy", width=0.012,
            headwidth=3.5, headlength=4.0,
        )
    iax.set_xlim(x0, x1); iax.set_ylim(y0, y1)
    iax.set_aspect("equal")
    iax.set_xticks([]); iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_edgecolor("#444")
        spine.set_linewidth(0.6)

    rect = Rectangle((x0, y0), zoom_size, zoom_size,
                     fill=False, edgecolor="#444", linewidth=0.6)
    ax.add_patch(rect)


def _cream_save(fig, name):
    """Save a snapshot figure with v1's cream background."""
    fig.patch.set_facecolor(style.CREAM)
    out = FIGS / name
    fig.savefig(out, facecolor=style.CREAM)
    fig.savefig(out.with_suffix(".png"), facecolor=style.CREAM)
    plt.close(fig)
    print(f"  saved {out}")


# Paper-wide colour code: imported from style.py so every figure
# in the project resolves a mode -> colour through the same dict.
PALETTE = style.PALETTE

LABELS = {
    "baseline":     r"Cauchy ref. ($\alpha\!=\!1$, $v\!=\!v_{\max}$)",
    "v2_limit":     r"noise-shape adaptive",
    "v3_limit":     r"motility adaptive",
    "full":         r"double-adaptive (this work)",
    "vicsek_gauss": r"Vicsek ($\alpha\!=\!2$, $v\!=\!v_{\max}$)",
}

DISPLAY = {
    "baseline":     "Cauchy reference",
    "v2_limit":     "noise-shape adaptive",
    "v3_limit":     "motility adaptive",
    "full":         "double-adaptive",
    "vicsek_gauss": "Vicsek (Gaussian)",
    "fixed_v":      "fixed speed",
    "adaptive":     "adaptive speed",
    "fixed_v_eta035": "fixed speed",
    "adaptive_eta100": "adaptive speed",
    "adaptive_eta150": "adaptive speed",
}


def _disp(label):
    return DISPLAY.get(label, label.replace("_", " "))


def fig_double_schematic():
    """Schematic of the shared sigmoid: v(n)/v_max and alpha(n) on
    the same n-axis, plus a small Couzin zonal cartoon."""
    import matplotlib.patches as patches

    n = np.linspace(0, 8, 200)
    n_star, slope = 3.0, 2.0
    sig = 1.0 / (1.0 + np.exp(-(n - n_star) * slope))
    v_max, v_min = 0.05, 0.005
    a_min, a_max = 1.0, 2.0
    v = v_max - (v_max - v_min) * sig
    a = a_min + (a_max - a_min) * sig

    fig, axes = plt.subplots(1, 2,
                             figsize=(style.DOUBLE_COL[0], 2.4))

    # (a) shared sigmoid
    ax = axes[0]
    ax2 = ax.twinx()
    l1, = ax.plot(n, v, "-", color="#7a3aa0", lw=1.6,
                  label=r"$v_i(n_i)$")
    l2, = ax2.plot(n, a, "-", color="#3aa040", lw=1.6,
                   label=r"$\alpha_i(n_i)$")
    ax.axvline(n_star, ls=":", c="grey", lw=0.8)
    ax.axhline(v_max, ls=":", c="#7a3aa0", lw=0.4)
    ax.axhline(v_min, ls=":", c="#7a3aa0", lw=0.4)
    ax2.axhline(a_min, ls=":", c="#3aa040", lw=0.4)
    ax2.axhline(a_max, ls=":", c="#3aa040", lw=0.4)
    ax.set_xlabel(r"local neighbour count $n_i$")
    ax.set_ylabel(r"speed $v_i$", color="#7a3aa0")
    ax2.set_ylabel(r"stability $\alpha_i$", color="#3aa040")
    ax.tick_params(axis="y", colors="#7a3aa0")
    ax2.tick_params(axis="y", colors="#3aa040")
    ax.text(n_star, v_max * 1.02, r"$n_\star$",
            fontsize=8, ha="center", color="grey")
    ax.text(0.5, 1.02, "(a) shared sigmoid", fontsize=8,
            transform=ax.transAxes, ha="center")

    # (b) Couzin zonal cartoon, v1 "definition-sketch" visual
    # language: a red repulsion disc inside a light-blue alignment
    # annulus, a focal blue arrow, labelled neighbours casting an
    # alignment or a turn-away response, and dashed R_r / R_a radii.
    from matplotlib.patches import Wedge

    ax = axes[1]
    ax.set_aspect("equal")
    R_r, R_a = 0.5, 0.8
    rep_color, ali_color = "#e07b7b", "#9bb8de"
    blue = style.PARTICLE_BLUE

    ax.add_patch(Wedge((0, 0), R_a, 0, 360, width=R_a - R_r,
                       facecolor=ali_color, alpha=0.55,
                       edgecolor="#3a4a78", lw=0.8, zorder=1))
    ax.add_patch(Wedge((0, 0), R_r, 0, 360,
                       facecolor=rep_color, alpha=0.55,
                       edgecolor="#9c3a3a", lw=0.8, zorder=1))

    # focal particle i with its heading
    hl = 0.30
    ax.annotate("", xy=(hl, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=blue, lw=2.4),
                zorder=6)
    ax.scatter([0], [0], s=70, color=blue, edgecolor="white",
               lw=0.8, zorder=7)
    ax.text(0.04, -0.14, r"$i$", fontsize=11, fontweight="bold",
            zorder=7)
    ax.text(hl + 0.02, 0.06, r"$\vec e_i$", fontsize=10, color=blue,
            zorder=7)

    al = 0.20

    def _nb(x, y, ang, label, color=blue, alpha=1.0):
        th = np.deg2rad(ang)
        ax.annotate("", xy=(x + al * np.cos(th), y + al * np.sin(th)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.5, alpha=alpha), zorder=6)
        ax.scatter([x], [y], s=34, color=color, alpha=alpha,
                   edgecolor="white", lw=0.6, zorder=7)
        ax.text(x + 0.04, y + 0.07, label, fontsize=9, alpha=alpha,
                zorder=7)

    # repulsion neighbour -> turn-away vector
    j1 = (-0.16, 0.20)
    _nb(*j1, 90, r"$j_1$")
    nrm = np.hypot(*j1)
    away = (-j1[0] / nrm * 0.34, -j1[1] / nrm * 0.34)
    ax.annotate("", xy=away, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#9c3a3a",
                                lw=1.5, ls=(0, (3, 2))), zorder=6)
    ax.text(away[0] + 0.02, away[1] - 0.08, "repulse", fontsize=8,
            color="#9c3a3a", style="italic", zorder=7)

    # alignment neighbours (omnidirectional)
    _nb(0.44, 0.40, 20, r"$j_2$")
    _nb(-0.50, -0.32, 200, r"$j_3$")
    _nb(-0.54, 0.06, 0, r"$j_4$")
    # out-of-range neighbour, greyed
    _nb(0.86, 0.50, 60, r"$j_\infty$", color="#888", alpha=0.55)

    ax.annotate(r"$R_r$",
                xy=(R_r * np.cos(np.deg2rad(-55)),
                    R_r * np.sin(np.deg2rad(-55))),
                xytext=(0.30, -0.92), fontsize=10,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444"))
    ax.annotate(r"$R_a$",
                xy=(R_a * np.cos(np.deg2rad(-32)),
                    R_a * np.sin(np.deg2rad(-32))),
                xytext=(0.92, -0.66), fontsize=10,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444"))

    ax.set_xlim(-1.05, 1.15)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 1.02, "(b) zonal Couzin neighbourhood",
            fontsize=8, transform=ax.transAxes, ha="center")

    fig.tight_layout()
    _save(fig, "fig_double_schematic.pdf")


def fig_double_pilot(npz_path: Path):
    """Six-panel pilot summary across the four modes. If
    data/double_L64.npz exists, the FSS panel uses 5 sizes."""
    z = np.load(npz_path, allow_pickle=True)
    modes = [str(s) if isinstance(s, str) else s.decode()
             for s in z["modes"]]
    Ls = z["Ls"]
    etas = z["etas"]
    phi = z["phi"]
    chi = z["chi"]
    U4 = z["U4"]
    s_sep = z["s_sep"]
    v_pop = z["v_pop"]
    a_pop = z["a_pop"]

    # Merge the controlled FSS sizes L=64, 90, 128 (no-cone series,
    # matching Table I). The cone-era double_L*.npz files carry a
    # different protocol --- their reference-mode chi_max is spuriously
    # large --- and must NOT be mixed into this no-cone pilot.
    for big_name in ("double_L64_nocone.npz", "double_L90_nocone.npz",
                     "double_L128_nocone.npz"):
        big_path = DATA / big_name
        if not big_path.exists():
            continue
        big = np.load(big_path, allow_pickle=True)
        Ls = np.concatenate([Ls, [float(big["L"])]])
        # These files carry 5 modes (vicsek_gauss + 4 ht); the
        # pilot-derived figure only carries the 4 heavy-tailed modes,
        # so we slice big arrays to those four.
        big_modes = [str(s) if isinstance(s, str) else s.decode()
                     for s in big["modes"]]
        idx = [big_modes.index(m) for m in modes if m in big_modes]
        chi_extra = big["chi"][idx][:, None, :]
        chi = np.concatenate([chi, chi_extra], axis=1)
        phi = np.concatenate([phi, big["phi"][idx][:, None, :]], axis=1)
        U4 = np.concatenate([U4, big["U4"][idx][:, None, :]], axis=1)
        s_sep = np.concatenate(
            [s_sep, big["s_sep"][idx][:, None, :]], axis=1)
        v_pop = np.concatenate(
            [v_pop, big["v_pop"][idx][:, None, :]], axis=1)
        a_pop = np.concatenate(
            [a_pop, big["a_pop"][idx][:, None, :]], axis=1)

    fig, axes = plt.subplots(2, 3,
                             figsize=(style.DOUBLE_COL[0] * 1.20, 5.8),
                             gridspec_kw=dict(wspace=0.22, hspace=0.28))
    for _ax in axes.ravel():
        _ax.set_box_aspect(1)
        for _sp in _ax.spines.values():
            _sp.set_visible(True); _sp.set_linewidth(0.8)
            _sp.set_edgecolor("#333333")
    panel_titles = [
        r"(a) $\langle\varphi\rangle(\eta)$, $L\!=\!L_{\max}$",
        r"(b) $\chi(\eta)$, $L\!=\!L_{\max}$",
        r"(c) $s_{\rm sep}(\eta)$, $L\!=\!L_{\max}$",
        r"(d) $U_4(\eta)$, $L\!=\!L_{\max}$",
        r"(e) $\chi_{\max}$ vs $L$ (log-log)",
        r"(f) $\langle v_i\rangle$, $\langle\alpha_i\rangle$ vs $\eta$",
    ]
    # Panels (a)-(d) and (f) show one representative L. We pick the
    # largest L for which per-seed data is available so the SE bands
    # are drawn; otherwise fall back to the largest L in the merged
    # array.
    has_per_seed_pre = "phi_per_seed" in z.files
    n_pilot_L_pre = (z["phi_per_seed"].shape[1]
                      if has_per_seed_pre else 0)
    iL = (n_pilot_L_pre - 1 if has_per_seed_pre and n_pilot_L_pre > 0
           else len(Ls) - 1)
    L_panel = float(Ls[iL])

    # (a) phi
    # Per-seed SE bands when available. The pilot file may carry
    # per-seed for the first n_pilot_L sizes only; later sizes
    # (merged in from L=64/90/128 files) have no SE so we skip
    # the bands at those iL.
    has_per_seed = "phi_per_seed" in z.files
    n_pilot_L = (z["phi_per_seed"].shape[1] if has_per_seed else 0)

    def _se(arr_per_seed, im, iL):
        if iL >= arr_per_seed.shape[1]:
            return None
        a = arr_per_seed[im, iL]      # (n_eta, n_seed)
        ok = np.isfinite(a)
        n = ok.sum(axis=-1)
        return np.where(n > 1,
                         np.nanstd(a, axis=-1, ddof=1)
                         / np.sqrt(np.maximum(n, 1)),
                         np.nan)

    if has_per_seed:
        phi_seed = z["phi_per_seed"]
        chi_seed = z["chi_per_seed"]
        U4_seed = z["U4_per_seed"]
        s_sep_seed = z["s_sep_per_seed"]

    ax = axes[0, 0]
    for im, m in enumerate(modes):
        c = PALETTE[m]
        ax.plot(etas, phi[im, iL], "o-", color=c, label=LABELS[m],
                lw=1.2, ms=3)
        if has_per_seed:
            se = _se(phi_seed, im, iL)
            if se is None: continue
            ax.fill_between(etas, phi[im, iL] - se, phi[im, iL] + se,
                             color=c, alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\langle\varphi\rangle$")
    ax.set_title(r"(a) $\langle\varphi\rangle(\eta)$", fontsize=9, loc="left")

    # (b) chi -- log-y so the heavy spread does not flatten everything else.
    ax = axes[0, 1]
    for im, m in enumerate(modes):
        c = PALETTE[m]
        chi_pos = np.where(chi[im, iL] > 0, chi[im, iL], np.nan)
        ax.plot(etas, chi_pos, "o-", color=c, lw=1.2, ms=3)
        if has_per_seed:
            se = _se(chi_seed, im, iL)
            if se is None: continue
            lo = np.maximum(chi_pos - se, 1e-3)
            ax.fill_between(etas, lo, chi_pos + se,
                             color=c, alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\chi$")
    ax.set_title(r"(b) $\chi(\eta)$", fontsize=9, loc="left")

    # (c) s_sep
    ax = axes[0, 2]
    for im, m in enumerate(modes):
        c = PALETTE[m]
        ax.plot(etas, s_sep[im, iL], "o-", color=c, lw=1.2, ms=3)
        if has_per_seed:
            se = _se(s_sep_seed, im, iL)
            if se is None: continue
            ax.fill_between(etas, s_sep[im, iL] - se,
                             s_sep[im, iL] + se,
                             color=c, alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$s_{\rm sep}$")
    ax.set_title(r"(c) $s_{\rm sep}(\eta)$", fontsize=9, loc="left")

    # (d) U4
    ax = axes[1, 0]
    for im, m in enumerate(modes):
        c = PALETTE[m]
        ax.plot(etas, U4[im, iL], "o-", color=c, lw=1.2, ms=3)
        if has_per_seed:
            se = _se(U4_seed, im, iL)
            if se is None: continue
            ax.fill_between(etas, U4[im, iL] - se, U4[im, iL] + se,
                             color=c, alpha=0.18, lw=0)
    ax.axhline(2.0 / 3.0, ls="--", c="black", lw=0.5, alpha=0.7)
    ax.text(etas[-1] * 1.02, 2.0 / 3.0, r"$2/3$",
             fontsize=7, va="center", ha="left", color="black")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$U_4$")
    ax.set_title(r"(d) $U_4(\eta)$", fontsize=9, loc="left")

    # (e) chi_max FSS with bootstrap CI on the slope.
    ax = axes[1, 1]
    rng = np.random.default_rng(0)
    for im, m in enumerate(modes):
        chi_max = chi[im].max(axis=1)
        ax.plot(Ls, chi_max, "o", color=PALETTE[m], ms=5, zorder=3)
        a, b = np.polyfit(np.log(Ls), np.log(chi_max), 1)
        # bootstrap the slope by resampling sizes with replacement
        boot = []
        for _ in range(2000):
            idx = rng.integers(0, len(Ls), len(Ls))
            try:
                ai, _ = np.polyfit(np.log(Ls[idx]),
                                    np.log(chi_max[idx]), 1)
                boot.append(ai)
            except (np.linalg.LinAlgError, ValueError):
                continue
        a_lo, a_hi = np.percentile(boot, [16, 84])
        a_err = 0.5 * (a_hi - a_lo)
        L_grid = np.linspace(Ls.min(), Ls.max(), 50)
        ax.plot(L_grid, np.exp(b) * L_grid ** a, "-",
                 color=PALETTE[m], lw=1.0, alpha=0.85, zorder=2)
        ax.text(0.03, 0.97 - 0.07 * im,
                fr"$a={a:+.2f}\pm{a_err:.2f}$",
                transform=ax.transAxes,
                fontsize=7, color=PALETTE[m],
                ha="left", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\chi_{\max}$")
    ax.set_title(r"(e) $\chi_{\max}(L)$", fontsize=9, loc="left")

    # (f) v_pop and a_pop overlay
    ax = axes[1, 2]
    ax2 = ax.twinx()
    for im, m in enumerate(modes):
        ax.plot(etas, v_pop[im, iL], "-", color=PALETTE[m], lw=1.2)
        ax2.plot(etas, a_pop[im, iL], ":", color=PALETTE[m], lw=1.0)
    ax.axhline(0.05, ls=":", c="grey", lw=0.5)
    ax.axhline(0.005, ls=":", c="grey", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\langle v_i\rangle$  (solid)")
    ax2.set_ylabel(r"$\langle\alpha_i\rangle$  (dotted)")
    ax2.set_ylim(0.95, 2.05)
    ax.set_title(r"(f) $\langle v\rangle,\,\langle\alpha\rangle(\eta)$", fontsize=9, loc="left")

    # Shared legend at the top of the figure, no stack on panel a.
    handles = [plt.Line2D([0], [0], color=PALETTE[m], marker="o",
                            ms=4, label=_disp(m))
                for m in modes]
    axes[0, 0].legend(handles=handles, loc="lower left", fontsize=6,
                      frameon=True, framealpha=0.9, handlelength=1.2,
                      borderpad=0.3, labelspacing=0.3)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.07,
                        wspace=0.42, hspace=0.40)
    # Enclosing "card" frame around each of the six subfigures (the
    # data axes plus their title, ticks and axis labels), drawn on the
    # cream background like Fig. 1. Read from each panel's tight bbox so
    # the box hugs the whole subfigure; panel (f) unions its twin axis.
    from matplotlib.patches import Rectangle as _Rect
    from matplotlib.transforms import Bbox as _Bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pad = 0.010
    twin_of = {id(axes[1, 2]): ax2}
    for a in axes.ravel():
        bb = a.get_tightbbox(renderer)
        if id(a) in twin_of:
            bb = _Bbox.union([bb, twin_of[id(a)].get_tightbbox(renderer)])
        (x0, y0), (x1, y1) = inv.transform(bb)
        fig.add_artist(_Rect((x0 - pad, y0 - pad),
                             (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad,
                             transform=fig.transFigure, fill=False,
                             edgecolor="#333333", lw=0.9, zorder=0))
    _save(fig, "fig_double_pilot.pdf")


def fig_double_snapshot(npz_path: Path):
    """Real-space snapshot grid on a white background, arrows coloured
    by local speed v_i. When the multi-size file
    (double_snapshot_multiL_nocone.npz) is present, the grid spans
    three sizes L in {30, 90, 128} (two mode-rows each) and three
    noise columns (ordered / near-critical / disordered); otherwise a
    single-size 2x3 grid is drawn. Each panel carries a 5x5 zoom
    inset on its densest sub-region."""
    multi = DATA / "double_snapshot_multiL_nocone.npz"
    arrow_len = 0.45

    if multi.exists():
        z = np.load(multi, allow_pickle=True)
        L_list = [float(v) for v in z["L_list"]]
        mode_labels = [str(s) if isinstance(s, str) else s.decode()
                       for s in z["mode_labels"]]
        case_labels = [str(s) if isinstance(s, str) else s.decode()
                       for s in z["case_labels"]]
        x = z["x"]; y = z["y"]; theta = z["theta"]; v = z["v"]
        eta = z["eta"]; phi = z["phi"]; counts = z["counts"]

        n_L, nm, nc = len(L_list), len(mode_labels), len(case_labels)
        # One row per size; within a row each noise case occupies an
        # adjacent (motility, double-adaptive) column pair, so a row
        # reads left to right as the three cases side by side.
        nrow = n_L
        ncol = nc * nm
        # Canvas scaled up (and the manuscript \includegraphics width to
        # match) plus tight inter-panel spacing so each snapshot panel
        # renders ~10% larger than the previous version while the
        # rasterised quiver keeps its ~200 dpi at the bigger display size.
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(1.32 * style.DOUBLE_COL[0],
                                          0.66 * style.DOUBLE_COL[0]),
                                 gridspec_kw={"wspace": 0.05,
                                              "hspace": 0.18})
        last_q = None
        # Columns left to right in data order: ordered -> near-critical
        # -> disordered.
        case_order = list(range(nc))
        for iL in range(n_L):
            for jc, ic in enumerate(case_order):
                for im in range(nm):
                    col = jc * nm + im
                    ax = axes[iL, col]
                    L = L_list[iL]
                    N = int(counts[iL, im, ic])
                    xs = x[iL, im, ic, :N]
                    ys = y[iL, im, ic, :N]
                    ths = theta[iL, im, ic, :N]
                    vs = v[iL, im, ic, :N]
                    wid = 0.004 * 30.0 / L
                    q = ax.quiver(xs, ys, np.cos(ths), np.sin(ths), vs,
                                  cmap=SPEED_CMAP,
                                  clim=(SPEED_VMIN, SPEED_VMAX),
                                  scale=1.0 / arrow_len,
                                  scale_units="xy", angles="xy",
                                  width=wid, headwidth=3.5,
                                  headlength=4.0, rasterized=True)
                    last_q = q
                    ax.set_xlim(0, L); ax.set_ylim(0, L)
                    ax.set_aspect("equal")
                    ax.set_xticks([]); ax.set_yticks([])
                    _zoom_inset(ax, xs, ys, ths, L, zoom_size=5.0,
                                arrow_len=arrow_len, speeds=vs,
                                bg=style.WHITE)
                    # Mode name on the top row only; size label on the
                    # leftmost column only (no repeats elsewhere).
                    if iL == 0:
                        ax.set_title(_disp(mode_labels[im]), fontsize=7)
                    if col == 0:
                        ax.set_ylabel(fr"$L = {int(L)}$", fontsize=8)
                    ax.text(0.03, 0.97,
                            fr"$\langle\varphi\rangle="
                            fr"{phi[iL, im, ic]:.2f}$",
                            transform=ax.transAxes, fontsize=6,
                            fontweight="bold", ha="left", va="top",
                            bbox=dict(facecolor=style.CREAM, alpha=0.9,
                                      edgecolor="none", pad=1))
        fig.tight_layout(rect=(0.0, 0.0, 0.96, 0.93), w_pad=0.25,
                         h_pad=0.4)
        # A single case/eta header centred over each column pair so the
        # noise level is written once, not once per mode.
        for jc, ic in enumerate(case_order):
            p_l = axes[0, jc * nm].get_position()
            p_r = axes[0, jc * nm + nm - 1].get_position()
            fig.text(0.5 * (p_l.x0 + p_r.x1), p_l.y1 + 0.055,
                     fr"({chr(97 + jc)}) {_disp(case_labels[ic])}, "
                     fr"$\eta={eta[ic]:g}$",
                     ha="center", va="bottom", fontsize=8,
                     fontweight="bold")
        cbar = fig.colorbar(last_q, ax=axes, orientation="vertical",
                            fraction=0.012, pad=0.012)
        cbar.set_label(r"local speed $v_i$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        _save(fig, "fig_double_snapshot.pdf")
        return

    z = np.load(npz_path, allow_pickle=True)
    mode_labels = [str(s) if isinstance(s, str) else s.decode()
                   for s in z["mode_labels"]]
    case_labels = [str(s) if isinstance(s, str) else s.decode()
                   for s in z["case_labels"]]
    x = z["x"]; y = z["y"]; theta = z["theta"]
    v = z["v"]; eta = z["eta"]; phi = z["phi"]
    L = float(z["params"][1])

    nm, nc = len(mode_labels), len(case_labels)
    fig, axes = plt.subplots(nm, nc,
                             figsize=(style.DOUBLE_COL[0], 5.0))
    last_q = None
    for im in range(nm):
        for ic in range(nc):
            ax = axes[im, ic]
            q = ax.quiver(x[im, ic], y[im, ic],
                          np.cos(theta[im, ic]), np.sin(theta[im, ic]),
                          v[im, ic],
                          cmap=SPEED_CMAP,
                          clim=(SPEED_VMIN, SPEED_VMAX),
                          scale=1.0 / arrow_len, scale_units="xy",
                          angles="xy", width=0.004,
                          headwidth=3.5, headlength=4.0)
            ax.set_xlim(0, L); ax.set_ylim(0, L)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            _zoom_inset(ax, x[im, ic], y[im, ic], theta[im, ic],
                        L, zoom_size=5.0, arrow_len=arrow_len,
                        speeds=v[im, ic], bg=style.WHITE)
            ax.set_title(
                fr"{_disp(mode_labels[im])} | {_disp(case_labels[ic])}, "
                fr"$\eta={eta[ic]:g}$" "\n"
                fr"$\langle\varphi\rangle = {phi[im, ic]:.2f}$",
                fontsize=7,
            )
            last_q = q
    fig.tight_layout(rect=(0.0, 0.0, 0.94, 1.0))
    cbar = fig.colorbar(last_q, ax=axes, orientation="vertical",
                        fraction=0.020, pad=0.02)
    cbar.set_label(r"local speed $v_i$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    _save(fig, "fig_double_snapshot.pdf")




def _plane_panel(ax, fig, data, n_stars, slopes, cmap, vlabel,
                 title=None, letter=None,
                 show_xlabel=True, show_ylabel=True):
    """Render one (n_star, s) heat-map panel with a smooth, continuous
    colour field (Gouraud-shaded pcolormesh) and no cell annotations.
    Axis labels are drawn only where requested so a grid of panels can
    keep the y-label on the left column and the x-label on the bottom
    row."""
    S, NS = np.meshgrid(slopes, n_stars)
    pm = ax.pcolormesh(S, NS, data, cmap=cmap, shading="gouraud")
    if show_xlabel:
        ax.set_xlabel(r"sigmoid slope $s$")
    if show_ylabel:
        ax.set_ylabel(r"threshold $n_\star$")
    if title is not None:
        ax.set_title(title, fontsize=9)
    if letter is not None:
        ax.text(0.04, 0.96, letter, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.75,
                          edgecolor="none", pad=1))
    cb = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(vlabel, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def fig_double_plane(npz_L22: Path, npz_L30: Path):
    """Behavioural (n_star, s) plane of the full two-feedback model,
    one row per system size (chi_max, s_sep). The fine-grid file
    (double_plane_fine_nocone.npz) supplies L=30 and the large-L file
    adds L=90 and L=128, all on the same 13x13 grid so the colour
    fields read as continuous; otherwise the legacy 5x5 files are
    used. Maps share the navy-mauve-vermillion snapshot colormap."""
    fine = DATA / "double_plane_fine_nocone.npz"
    large = DATA / "double_plane_largeL_nocone.npz"
    if fine.exists():
        zf = np.load(fine, allow_pickle=True)
        n_stars = zf["n_stars"]
        slopes = zf["slopes"]
        # One row per size; the large-L re-sweep adds L=90 and L=128
        # once data/double_plane_largeL_nocone.npz is present.
        rows = [
            (float(zf["L30"]), zf["chi_peak_30"], zf["sep_peak_30"]),
        ]
        if large.exists():
            zl = np.load(large, allow_pickle=True)
            rows.append((float(zl["L90"]), zl["chi_peak_90"],
                         zl["sep_peak_90"]))
            rows.append((float(zl["L128"]), zl["chi_peak_128"],
                         zl["sep_peak_128"]))
        nrow = len(rows)
        fig, axes = plt.subplots(nrow, 2,
                                 figsize=(style.DOUBLE_COL[0],
                                          2.5 * nrow))
        axes = np.atleast_2d(axes)
        letters = "abcdefghijkl"
        for ir, (L, chi, sep) in enumerate(rows):
            # y-label only on the left column (a, c, e); x-label only
            # on the bottom row (e, f).
            bottom = (ir == nrow - 1)
            _plane_panel(axes[ir, 0], fig, chi, n_stars, slopes,
                         SPEED_CMAP, r"$\chi_{\max}$",
                         title=(r"$\chi_{\max}$" if ir == 0 else None),
                         letter=fr"({letters[2 * ir]}) $L={int(L)}$",
                         show_xlabel=bottom, show_ylabel=True)
            _plane_panel(axes[ir, 1], fig, sep, n_stars, slopes,
                         SPEED_CMAP, r"$s_{\rm sep}$",
                         title=(r"$s_{\rm sep}$" if ir == 0 else None),
                         letter=fr"({letters[2 * ir + 1]})",
                         show_xlabel=bottom, show_ylabel=False)
        fig.tight_layout()
        _save(fig, "fig_double_plane.pdf")
        return

    z22 = np.load(npz_L22, allow_pickle=True)
    z30 = np.load(npz_L30, allow_pickle=True)
    n_stars = z22["n_stars"]
    slopes = z22["slopes"]

    fig, axes = plt.subplots(2, 2, figsize=(style.DOUBLE_COL[0], 5.0))
    panels = (
        (axes[0, 0], z22["chi_peak"],
         fr"(a) $\chi_{{\max}}$ ($L = {float(z22['L']):g}$)",
         "RdBu_r", r"$\chi_{\max}$"),
        (axes[0, 1], z22["sep_peak"], r"(b) $s_{\rm sep}$",
         "RdBu_r", r"$s_{\rm sep}$"),
        (axes[1, 0], z30["chi_peak"],
         fr"(c) $\chi_{{\max}}$ ($L = {float(z30['L']):g}$, 5 seeds)",
         "RdBu_r", r"$\chi_{\max}$"),
        (axes[1, 1], z30["sep_peak"], r"(d) $s_{\rm sep}$",
         "RdBu_r", r"$s_{\rm sep}$"),
    )
    for ax, data, title, cmap, vlabel in panels:
        _plane_panel(ax, fig, data, n_stars, slopes, cmap, vlabel,
                     title=title)
    fig.tight_layout()
    _save(fig, "fig_double_plane.pdf")


def _orderpdf_traj(z, labels, key):
    """Return the phi trajectory for a mode, tolerating the
    motility/v3_limit alias across the order-parameter npz files."""
    aliases = {"motility": ("motility", "v3_limit"),
               "full": ("full",)}.get(key, (key,))
    for a in aliases:
        if a in labels:
            return labels.index(a)
    raise KeyError(f"{key} not in {labels}")


def fig_double_orderpdf(npz_L30: Path, npz_largeL: Path,
                        npz_L128: Path | None = None):
    r"""Order-parameter distribution $P(\langle\varphi\rangle)$ of
    the motility-only and double-adaptive modes, one panel per size
    $L \in \{30, 64, 90, 128\}$, motility and full overlaid so the
    macroscopic mean shift reads as a scale-invariant signature."""
    from scipy.stats import gaussian_kde

    panel_data = []
    # L = 30 (motility label is v3_limit here).
    z0 = np.load(npz_L30, allow_pickle=True)
    labs0 = [str(s) if isinstance(s, str) else s.decode()
             for s in z0["labels"]]
    eta0 = z0["eta"]
    i_m0 = _orderpdf_traj(z0, labs0, "motility")
    i_f0 = _orderpdf_traj(z0, labs0, "full")
    panel_data.append((30.0, float(eta0[i_m0]), float(eta0[i_f0]),
                       z0["phi_traj"][i_m0], z0["phi_traj"][i_f0]))
    # L = 64, 90.
    zL = np.load(npz_largeL, allow_pickle=True)
    labsL = [str(s) if isinstance(s, str) else s.decode()
             for s in zL["labels"]]
    etaL = zL["eta_per_case"]
    i_mL = _orderpdf_traj(zL, labsL, "motility")
    i_fL = _orderpdf_traj(zL, labsL, "full")
    for iL, L in enumerate(zL["Ls"]):
        panel_data.append((float(L), float(etaL[i_mL]), float(etaL[i_fL]),
                           zL["phi_traj"][i_mL, iL],
                           zL["phi_traj"][i_fL, iL]))
    # L = 128 (optional).
    if npz_L128 is not None and npz_L128.exists():
        z2 = np.load(npz_L128, allow_pickle=True)
        labs2 = [str(s) if isinstance(s, str) else s.decode()
                 for s in z2["labels"]]
        eta2 = z2["eta_per_case"]
        i_m2 = _orderpdf_traj(z2, labs2, "motility")
        i_f2 = _orderpdf_traj(z2, labs2, "full")
        panel_data.append((float(z2["L"]), float(eta2[i_m2]),
                           float(eta2[i_f2]),
                           z2["phi_traj"][i_m2], z2["phi_traj"][i_f2]))

    # Height trimmed 20% (2.8 -> 2.24) and a shared y-axis so the
    # P-axis label and ticks appear only on the leftmost panel.
    fig, axes = plt.subplots(1, len(panel_data),
                             figsize=(style.DOUBLE_COL[0], 2.24),
                             sharey=True)
    if len(panel_data) == 1:
        axes = [axes]
    bins = np.linspace(0, 1, 60)
    centers_b = 0.5 * (bins[:-1] + bins[1:])
    width_b = bins[1] - bins[0]
    grid = np.linspace(0, 1, 200)
    u4 = lambda a: 1.0 - (a ** 4).mean() / (3.0 * (a ** 2).mean() ** 2)
    for ip, (L, eta_mot, eta_full, m_mot, m_full) in enumerate(panel_data):
        ax = axes[ip]
        # First pass: build histograms and KDE curves, find the panel
        # peak so everything can be normalised to [0, 1].
        series = []
        peak = 0.0
        for lbl, data in (("motility", m_mot), ("full", m_full)):
            c = PALETTE.get(lbl, "#1f4ea1")
            hist, _ = np.histogram(data, bins=bins, density=True)
            kde = gaussian_kde(data, bw_method="scott")
            curve = kde(grid)
            peak = max(peak, float(hist.max()), float(curve.max()))
            series.append((lbl, data, c, hist, curve))
        # Second pass: plot normalised to the panel peak.
        for lbl, data, c, hist, curve in series:
            ax.bar(centers_b, hist / peak, width=width_b, color=c,
                   alpha=0.45, edgecolor="black", linewidth=0.3,
                   label=_disp(lbl))
            ax.plot(grid, curve / peak, "-", color=c, lw=1.4)
            ax.axvline(float(data.mean()), color=c, lw=0.7,
                       ls="--", alpha=0.85)
        gap = m_full.mean() - m_mot.mean()
        ax.text(0.03, 0.97,
                fr"$\langle\varphi\rangle_{{\rm mot}} = "
                fr"{m_mot.mean():.2f}$" "\n"
                fr"$\langle\varphi\rangle_{{\rm full}} = "
                fr"{m_full.mean():.2f}$" "\n"
                fr"$\Delta = {gap:+.2f}$" "\n"
                fr"$U_{{4,{{\rm full}}}} = {u4(m_full):.2f}$",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="none", pad=2))
        ax.set_xlabel(r"$\langle\varphi\rangle$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.0)
        ax.set_title(fr"$L = {int(L)}$", fontsize=8)
        if ip == 0:
            ax.set_ylabel(r"$P(\langle\varphi\rangle)$ (normalised)")
            ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.text(-0.18, 1.04, f"({chr(97 + ip)})",
                transform=ax.transAxes,
                fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig_double_orderpdf.pdf")


def fig_double_profile(npz_path: Path):
    """Polar-axis profiles of the four modes, overlaid on two axes.
    Panel (a): normalised density; panel (b): mean local speed. Each
    panel is y-limited to the span of the plotted curves; the
    band-soliton index b per mode is quoted in the legend."""
    z = np.load(npz_path, allow_pickle=True)
    labels = [str(s) if isinstance(s, str) else s.decode()
              for s in z["labels"]]
    centers = z["centers"]
    profiles = z["profiles"]
    v_profiles = z["v_profiles"]
    band_idx = z["band_idx"]
    L = float(z["params"][1])

    n = len(labels)
    fig, axes = plt.subplots(1, 2,
                             figsize=(style.DOUBLE_COL[0], 2.8))
    # Panel (a): density, each profile normalised to its own peak.
    dens = [profiles[ic] / profiles[ic].max() for ic in range(n)]

    ax = axes[0]
    for ic in range(n):
        ax.plot(centers, dens[ic], "-", lw=1.4,
                color=PALETTE.get(labels[ic], "#1f4ea1"),
                label=fr"{_disp(labels[ic])} ($b={band_idx[ic]:.2f}$)")
    ax.set_xlim(0, L)
    ax.set_ylim(0.85, 1.0)
    ax.set_xlabel(r"$x_\parallel$")
    ax.set_ylabel(r"$\rho(x_\parallel)/\rho_{\max}$")
    ax.set_title("(a)", fontsize=9, loc="left")
    # Four entries stacked on four lines, anchored at the panel's
    # lower centre and growing downward.
    ax.legend(fontsize=6, frameon=False, loc="lower center",
              ncol=1, bbox_to_anchor=(0.5, 0.0),
              borderaxespad=0.3, handlelength=1.6,
              labelspacing=0.35)

    # Panel (b): local speed, each profile normalised to its peak.
    ax = axes[1]
    for ic in range(n):
        vp = v_profiles[ic]
        vp = vp / vp.max() if vp.max() > 0 else vp
        ax.plot(centers, vp, "-", lw=1.4,
                color=PALETTE.get(labels[ic], "#1f4ea1"),
                label=_disp(labels[ic]))
    ax.set_xlim(0, L)
    ax.set_ylim(0.85, 1.0)
    ax.set_xlabel(r"$x_\parallel$")
    ax.set_ylabel(r"$\langle v_i\rangle(x_\parallel)/\langle v_i\rangle_{\max}$")
    ax.set_title("(b)", fontsize=9, loc="left")

    fig.tight_layout()
    _save(fig, "fig_double_profile.pdf")


def fig_double_clusters(npz_path: Path):
    r"""Cluster-size complementary CDF $P(s' \ge s)$ per mode.
    Modes whose dense fluctuations cap at the threshold floor
    (Cauchy reference, noise-shape only) are still drawn but
    pushed to the background."""
    z = np.load(npz_path, allow_pickle=True)
    labels = [str(s) if isinstance(s, str) else s.decode()
              for s in z["labels"]]
    fig, ax = plt.subplots(figsize=(style.SINGLE_COL[0] * 1.4,
                                     style.SINGLE_COL[1] * 1.3))
    # First the reference modes in muted form, then the
    # phase-separated modes on top.
    for m in labels:
        sizes = np.asarray(z[f"sizes_{m}"], dtype=int)
        if len(sizes) == 0:
            continue
        sizes = np.sort(sizes)
        ccdf = 1.0 - np.arange(len(sizes)) / len(sizes)
        col = PALETTE.get(m, "#1f4ea1")
        is_phase_separated = sizes.max() >= 8
        ax.loglog(sizes, ccdf,
                   lw=1.5 if is_phase_separated else 0.9,
                   color=col, alpha=1.0 if is_phase_separated else 0.55,
                   label=_disp(m))
    ax.set_xlabel(r"cluster size $s$ (particles)")
    ax.set_ylabel(r"$P(s' \geq s)$")
    ax.set_title("$L = 30$", fontsize=8, loc="left")
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    fig.tight_layout()
    _save(fig, "fig_double_clusters.pdf")


def fig_double_clusters_hs(npz_path: Path):
    r"""Cluster diagnostics from the high-statistics summary file
    (10 seeds per mode at $L = 30$). Three side-by-side bar
    panels: number of clusters per snapshot, maximum cluster
    size, and mean cluster size, each with seed-level standard
    error bars.

    This renderer replaces the legacy
    :func:`fig_double_clusters` cluster-size CCDF when the
    no-cone pipeline is in use; the underlying npz only stores
    seed-level reductions, not the full size distribution.
    """
    z = np.load(npz_path, allow_pickle=True)
    labels = [str(s) if isinstance(s, str) else s.decode()
              for s in z["labels"]]
    n_per_seed = z["n_per_seed"].astype(float)
    max_per_seed = z["max_per_seed"].astype(float)
    mean_per_seed = z["mean_per_seed"]
    n_seeds = n_per_seed.shape[1]

    fig, axes = plt.subplots(1, 3,
                              figsize=(style.DOUBLE_COL[0], 2.4))
    panels = (
        (axes[0], n_per_seed,    "(a) clusters per snapshot",
         r"$\langle n_{\rm cl}\rangle$"),
        (axes[1], max_per_seed,  "(b) maximum cluster size",
         r"$\langle s_{\max}\rangle$ (particles)"),
        (axes[2], mean_per_seed, "(c) mean cluster size",
         r"$\langle s\rangle$ (particles)"),
    )
    for ax, arr, title, ylabel in panels:
        means = arr.mean(axis=1)
        ses = arr.std(axis=1, ddof=1) / np.sqrt(n_seeds)
        colors = [PALETTE.get(m, "#1f4ea1") for m in labels]
        positions = np.arange(len(labels))
        ax.bar(positions, means, yerr=ses, capsize=2.5,
               color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(positions)
        ax.set_xticklabels([_disp(m) for m in labels],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8, loc="left")
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    _save(fig, "fig_double_clusters.pdf")


def _gr_decay_panel(ax, npz_decay: Path):
    """Panel (c): dense-quartile gap profile Delta g(r) across the
    four high-statistics sizes."""
    z = np.load(npz_decay, allow_pickle=True)
    Ls = z["Ls"]
    r = z["r_centers"]
    gap = z["gap"]
    se = z["se"]
    cmap = plt.get_cmap("plasma")
    for iL, L in enumerate(Ls):
        c = cmap(0.15 + 0.7 * iL / max(len(Ls) - 1, 1))
        ax.fill_between(r, gap[iL] - se[iL], gap[iL] + se[iL],
                        color=c, alpha=0.20)
        ax.plot(r, gap[iL], "-", lw=1.4, color=c,
                label=fr"$L={int(L)}$")
    ax.axhline(0.0, color="grey", ls="-", lw=0.4)
    ax.axvline(0.7, color="grey", ls=":", lw=0.5)
    ax.set_xlabel(r"distance $r$")
    ax.set_ylabel(r"$\Delta g(r) = g_{\rm full} - g_{\rm motility}$")
    ax.set_title("dense-quartile gap", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right", ncol=2)


def fig_double_gr(npz_path: Path, npz_decay: Path | None = None):
    """Density-stratified heading correlation $g(r)$. Panels (a) and
    (b): $g(r)$ in the dense and dilute quartiles for the four modes
    with seed-level error bands. Panel (c): the dense-quartile gap
    $\\Delta g(r)$ between double-adaptive and motility-only across
    four sizes."""
    z = np.load(npz_path, allow_pickle=True)
    labels = [str(s) if isinstance(s, str) else s.decode()
              for s in z["labels"]]
    r_centers = z["r_centers"]

    has_per_seed = "gr_dense_per_seed" in z.files
    if has_per_seed:
        gd = z["gr_dense_per_seed"]
        gl = z["gr_dilute_per_seed"]
        gr_dense = np.nanmean(gd, axis=1)
        gr_dilute = np.nanmean(gl, axis=1)
        n_d = np.sum(~np.isnan(gd), axis=1)
        n_l = np.sum(~np.isnan(gl), axis=1)
        gr_dense_se = np.nanstd(gd, axis=1, ddof=1) / np.sqrt(np.maximum(n_d, 1))
        gr_dilute_se = np.nanstd(gl, axis=1, ddof=1) / np.sqrt(np.maximum(n_l, 1))
    else:
        gr_dense = z["gr_dense"]
        gr_dilute = z["gr_dilute"]
        gr_dense_se = None

    has_decay = npz_decay is not None and npz_decay.exists()
    ncols = 3 if has_decay else 2
    fig, axes = plt.subplots(1, ncols,
                             figsize=(style.DOUBLE_COL[0], 2.8))
    for im, m in enumerate(labels):
        col = PALETTE.get(m, "#1f4ea1")
        axes[0].plot(r_centers, gr_dense[im], "-", lw=1.4,
                      color=col, label=_disp(m))
        axes[1].plot(r_centers, gr_dilute[im], "-", lw=1.4,
                      color=col, label=_disp(m))
        if gr_dense_se is not None:
            axes[0].fill_between(r_centers,
                                  gr_dense[im] - gr_dense_se[im],
                                  gr_dense[im] + gr_dense_se[im],
                                  color=col, alpha=0.18, lw=0)
            axes[1].fill_between(r_centers,
                                  gr_dilute[im] - gr_dilute_se[im],
                                  gr_dilute[im] + gr_dilute_se[im],
                                  color=col, alpha=0.18, lw=0)
    for ax in (axes[0], axes[1]):
        ax.axhline(0.0, ls=":", color="grey", lw=0.6)
        ax.axvline(0.7, ls=":", color="grey", lw=0.6)
        ax.set_xlabel(r"distance $r$")
        ax.set_xlim(0.5, r_centers[-1])
    axes[1].sharey(axes[0])
    axes[0].set_ylabel(r"$g(r) = \langle\cos[\theta_i - \theta_j]\rangle$")
    axes[0].set_title("dense quartile", fontsize=8)
    axes[1].set_title("dilute quartile", fontsize=8)
    # annotate the dense gap full vs motility at the inner bin.
    if has_per_seed and "v3_limit" in labels and "full" in labels:
        i_m = labels.index("v3_limit")
        i_f = labels.index("full")
        delta = gr_dense[i_f, 0] - gr_dense[i_m, 0]
        diff = gd[i_f, :, 0] - gd[i_m, :, 0]
        ok = np.isfinite(diff)
        z_score = diff[ok].mean() / (
            diff[ok].std(ddof=1) / np.sqrt(ok.sum()))
        axes[0].annotate(
            fr"$\Delta g = {delta:+.3f}$" "\n"
            fr"$z = {z_score:+.1f}$",
            xy=(r_centers[0], gr_dense[i_f, 0]),
            xytext=(0.45 * r_centers[-1], 0.28),
            fontsize=7, color=PALETTE["full"],
            arrowprops=dict(arrowstyle="-", lw=0.6,
                              color=PALETTE["full"]))
    axes[0].legend(fontsize=6, loc="lower left", frameon=False)
    if has_decay:
        _gr_decay_panel(axes[2], npz_decay)
    for ip in range(ncols):
        axes[ip].text(-0.18, 1.04, f"({chr(97 + ip)})",
                      transform=axes[ip].transAxes,
                      fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig_double_gr.pdf")


def fig_double_Lfine_gap(npz_fine, npz_pilot, npz_L64, npz_L90,
                          npz_L128):
    """Combine all measurements into a 12-size series of
    s_sep_max(L) for the double-adaptive and motility-only modes,
    and plot the gap full - motility."""
    pilot = np.load(npz_pilot, allow_pickle=True)
    big64 = np.load(npz_L64, allow_pickle=True)
    big90 = np.load(npz_L90, allow_pickle=True)
    big128 = np.load(npz_L128, allow_pickle=True)
    fine = np.load(npz_fine, allow_pickle=True)

    def big_idx(arr, name):
        ms = list(arr["modes"])
        if isinstance(ms[0], bytes):
            ms = [m.decode() for m in ms]
        try:
            return ms.index(name)
        except ValueError:
            return None

    # Canonical modes plotted in panel (a). "motility" is the
    # display name; in the data files it is stored as v3_limit
    # (pilot / big files) or motility (the all-modes fine scan).
    CANON = ["baseline", "v2_limit", "motility", "full"]
    ALIASES = {
        "baseline": ("baseline",),
        "v2_limit": ("v2_limit",),
        "motility": ("v3_limit", "motility"),
        "full": ("full",),
    }

    def _idx(modes_list, canon):
        for a in ALIASES[canon]:
            if a in modes_list:
                return modes_list.index(a)
        return None

    pilot_modes = [str(s) if isinstance(s, str) else s.decode()
                   for s in pilot["modes"]]
    fine_modes = [str(s) if isinstance(s, str) else s.decode()
                  for s in fine["modes"]]

    pts: list[tuple] = []   # (L, mode, s_sep_max)

    # Pilot covers L = 15, 22, 30, 45.
    for canon in CANON:
        ip = _idx(pilot_modes, canon)
        if ip is None:
            continue
        for iL, L in enumerate(pilot["Ls"]):
            pts.append((float(L), canon,
                        float(pilot["s_sep"][ip, iL].max())))

    # Big files: L = 64, 90, 128.
    for big in (big64, big90, big128):
        L = float(big["L"])
        big_modes = [str(s) if isinstance(s, str) else s.decode()
                     for s in big["modes"]]
        for canon in CANON:
            ib = _idx(big_modes, canon)
            if ib is not None:
                pts.append((L, canon, float(big["s_sep"][ib].max())))

    # Fine scan (all four modes): L = 38, 50, 60, 75, 105.
    for canon in CANON:
        iff = _idx(fine_modes, canon)
        if iff is None:
            continue
        for iL, L in enumerate(fine["Ls"]):
            pts.append((float(L), canon,
                        float(fine["s_sep"][iff, iL].max())))

    pts.sort()

    def _series(mode):
        Ls = sorted({p[0] for p in pts if p[1] == mode})
        s = [next(p[2] for p in pts if p[0] == L and p[1] == mode)
             for L in Ls]
        return np.array(Ls), np.array(s)

    series = {m: _series(m) for m in CANON}
    L_arr, s_full_arr = series["full"]
    _, s_mot_arr = series["motility"]
    gap = s_full_arr - s_mot_arr

    # High-statistics 10-seed reference values from
    # double_L_highstat.npz (overplotted with error bars).
    hs_path = Path(npz_fine).parent / "double_L_highstat.npz"
    has_hs = hs_path.exists()
    if has_hs:
        hs = np.load(hs_path, allow_pickle=True)
        hs_modes = [str(s) if isinstance(s, str) else s.decode()
                    for s in hs["modes"]]
        i_m_hs = hs_modes.index("motility")
        i_f_hs = hs_modes.index("full")
        Ls_hs = hs["Ls"]
        # Reduce s_sep_per_seed[mode, L, eta, seed] to per-(mode, L)
        # by taking the max over eta of the seed-mean, then SE over
        # seeds at that eta.
        s_per_seed = hs["s_sep_per_seed"]   # (n_mode, n_L, n_eta, n_seed)
        n_seed = s_per_seed.shape[-1]
        gap_hs = np.zeros(len(Ls_hs))
        gap_hs_se = np.zeros(len(Ls_hs))
        for iL in range(len(Ls_hs)):
            mean_e_m = s_per_seed[i_m_hs, iL].mean(axis=-1)
            mean_e_f = s_per_seed[i_f_hs, iL].mean(axis=-1)
            ie_m = int(np.argmax(mean_e_m))
            ie_f = int(np.argmax(mean_e_f))
            diff = (s_per_seed[i_f_hs, iL, ie_f]
                     - s_per_seed[i_m_hs, iL, ie_m])
            gap_hs[iL] = diff.mean()
            gap_hs_se[iL] = diff.std(ddof=1) / np.sqrt(n_seed)

    fig, axes = plt.subplots(1, 2,
                              figsize=(style.DOUBLE_COL[0], 2.8))
    ax = axes[0]
    markers = {"baseline": "v", "v2_limit": "^",
               "motility": "o", "full": "s"}
    pal = {"baseline": PALETTE["baseline"], "v2_limit": PALETTE["v2_limit"],
           "motility": PALETTE["v3_limit"], "full": PALETTE["full"]}
    for m in CANON:
        Lm, sm = series[m]
        if len(Lm) == 0:
            continue
        ax.plot(Lm, sm, markers[m], color=pal[m],
                label=_disp(m), ms=4)
    ax.axhline(1.73, color="black", ls="--", lw=0.5, alpha=0.6)
    ax.text(L_arr[-1] * 1.05, 1.73,
             r"$\simeq 1.73$ plateau",
             fontsize=7, va="center", ha="left", color="black")
    ax.set_xscale("log")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$s_{\rm sep}^{\max}$")
    ax.set_title(r"(a)", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="lower right", frameon=False)

    ax = axes[1]
    ax.axhline(0.0, color="grey", ls="-", lw=0.4)
    ax.plot(L_arr, gap, "D", color=style.WONG["blue"], ms=4,
             label="3 to 5 seeds")
    if has_hs:
        ax.errorbar(Ls_hs, gap_hs, yerr=gap_hs_se, fmt="o",
                     color=style.WONG["vermil"], ms=5, capsize=3,
                     lw=0.8, label="10 seeds")
    ax.set_xscale("log")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$s_{\rm sep}^{\rm full} - s_{\rm sep}^{\rm motility}$")
    ax.set_title(r"(b)", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="lower left", frameon=False)

    fig.tight_layout()
    _save(fig, "fig_double_Lfine_gap.pdf")


def fig_double_3regimes(npz_path: Path):
    """Three-regime snapshot of the proposed double-adaptive
    model (ordered, near-critical, disordered) in the minimalist
    early-Vicsek snapshot style on a white background. Panels are
    labelled (a)--(c); a single speed colorbar matched to the box
    height sits at the right."""
    z = np.load(npz_path, allow_pickle=True)
    mode_labels = [str(s) if isinstance(s, str) else s.decode()
                   for s in z["mode_labels"]]
    case_labels = [str(s) if isinstance(s, str) else s.decode()
                   for s in z["case_labels"]]
    x = z["x"]; y = z["y"]; theta = z["theta"]
    vs = z["v"]; eta = z["eta"]; phi = z["phi"]
    L = float(z["params"][1])

    im_full = mode_labels.index("full")
    nice_cases = ["Ordered", "Near-critical", "Disordered"]
    tags = ["(a)", "(b)", "(c)"]

    fig, axes = plt.subplots(1, 3,
                             figsize=(style.DOUBLE_COL[0], 2.7),
                             constrained_layout=True)
    arrow_len = 0.45
    last_q = None
    for ic, ax in enumerate(axes):
        u = np.cos(theta[im_full, ic])
        w = np.sin(theta[im_full, ic])
        q = ax.quiver(
            x[im_full, ic], y[im_full, ic], u, w,
            vs[im_full, ic],
            cmap=SPEED_CMAP,
            clim=(SPEED_VMIN, SPEED_VMAX),
            scale=1.0 / arrow_len, scale_units="xy",
            angles="xy", width=0.004,
            headwidth=3.5, headlength=4.0,
        )
        ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        _zoom_inset(ax, x[im_full, ic], y[im_full, ic], theta[im_full, ic],
                    L, zoom_size=5.0, arrow_len=arrow_len,
                    speeds=vs[im_full, ic], bg=style.WHITE)
        ax.set_title(
            f"{tags[ic]} {nice_cases[ic]}\n"
            fr"$\eta={eta[ic]:g}$, "
            fr"$\langle\varphi\rangle={phi[im_full, ic]:.2f}$",
            fontsize=8,
        )
        last_q = q
    cbar = fig.colorbar(last_q, ax=axes, orientation="vertical",
                        fraction=0.046, pad=0.02, aspect=16)
    cbar.set_label(r"local speed $v_i$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    _save(fig, "fig_double_3regimes.pdf")


def fig_double_cluster_map(npz_path: Path):
    """Real-space cluster identification at the near-critical point
    for motility-only (top row) and double-adaptive (bottom row)
    across system sizes L in {30, 90, 128} (columns). Snapshot
    aesthetic on a white background: arrows are coloured by the local
    speed v_i on the shared navy--mauve--vermillion colormap, exactly
    as in the snapshot figures, and a fixed 5x5 zoom inset accompanies
    each panel. The connected-component dense clusters (bins above
    1.5x the median occupancy on a 10x10 grid) are highlighted with a
    light green shading. Panels are labelled (a)-(f)."""
    from scipy.ndimage import label as nd_label
    from matplotlib.patches import Rectangle
    import matplotlib.colors as mcolors

    z = np.load(npz_path, allow_pickle=True)
    mode_labels = [str(s) if isinstance(s, str) else s.decode()
                   for s in z["mode_labels"]]
    L_list = [float(v) for v in z["L_list"]]
    counts = z["counts"]
    x = z["x"]; y = z["y"]; theta = z["theta"]; vv = z["v"]
    eta = float(z["eta"]); phi = z["phi"]

    n_bin = 10
    factor = 1.5
    arrow_len = 0.45
    zoom_size = 5.0      # fixed, as in the snapshot figures
    green_shade = (0.0, 0.6, 0.3, 0.18)   # light green dense-zone fill

    order = ["v3_limit", "full"]
    row_idx = [mode_labels.index(m) for m in order if m in mode_labels]
    row_title = {"v3_limit": "motility adaptive",
                 "full": "double-adaptive"}

    nrow, ncol = len(row_idx), len(L_list)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(style.DOUBLE_COL[0],
                                      1.8 * nrow + 0.6))
    if nrow == 1:
        axes = axes[None, :]
    tags = "abcdefghi"
    last_q = None

    for r, im in enumerate(row_idx):
        for c, L in enumerate(L_list):
            ax = axes[r, c]
            N = int(counts[im, c])
            xs = x[im, c, :N]
            ys = y[im, c, :N]
            ths = theta[im, c, :N]
            vs = vv[im, c, :N]

            H, _, _ = np.histogram2d(
                xs, ys, bins=[n_bin, n_bin], range=[[0, L], [0, L]])
            nz = H[H > 0]
            threshold = factor * np.median(nz) if len(nz) else 0.0
            mask = H > threshold
            tiled = np.tile(mask, (3, 3))
            labelled, _ = nd_label(tiled)
            central = labelled[n_bin:2 * n_bin, n_bin:2 * n_bin]
            n_clusters = len(np.unique(central[central > 0]))

            # Green shading of the connected dense clusters.
            ax.imshow(mask.T, extent=[0, L, 0, L], origin="lower",
                      cmap=mcolors.ListedColormap(
                          [(0, 0, 0, 0), green_shade]),
                      aspect="auto", interpolation="nearest")

            # Snapshot-style quiver coloured by local speed; the
            # marker width shrinks with system size for legibility.
            wid = 0.004 * 30.0 / L
            q = ax.quiver(xs, ys, np.cos(ths), np.sin(ths), vs,
                          cmap=SPEED_CMAP,
                          clim=(SPEED_VMIN, SPEED_VMAX),
                          scale=1.0 / arrow_len, scale_units="xy",
                          angles="xy", width=wid,
                          headwidth=3.5, headlength=4.0, zorder=3)
            last_q = q
            ax.set_xlim(0, L); ax.set_ylim(0, L)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])

            # Fixed 5x5 zoom inset on the densest sub-region.
            Hz, xe, ye = np.histogram2d(xs, ys, bins=10,
                                        range=[[0, L], [0, L]])
            zix, ziy = np.unravel_index(np.argmax(Hz), Hz.shape)
            zcx = 0.5 * (xe[zix] + xe[zix + 1])
            zcy = 0.5 * (ye[ziy] + ye[ziy + 1])
            z0x = max(0.0, min(L - zoom_size, zcx - zoom_size / 2))
            z0y = max(0.0, min(L - zoom_size, zcy - zoom_size / 2))
            iax = ax.inset_axes((1.0 - 0.38 - 0.005,
                                 1.0 - 0.38 - 0.005, 0.38, 0.38))
            iax.set_facecolor(style.WHITE)
            sel = ((xs >= z0x) & (xs <= z0x + zoom_size)
                   & (ys >= z0y) & (ys <= z0y + zoom_size))
            iax.quiver(xs[sel], ys[sel],
                       np.cos(ths[sel]), np.sin(ths[sel]), vs[sel],
                       cmap=SPEED_CMAP, clim=(SPEED_VMIN, SPEED_VMAX),
                       scale=1.0 / arrow_len, scale_units="xy",
                       angles="xy", width=0.012,
                       headwidth=3.5, headlength=4.0)
            iax.set_xlim(z0x, z0x + zoom_size)
            iax.set_ylim(z0y, z0y + zoom_size)
            iax.set_aspect("equal")
            iax.set_xticks([]); iax.set_yticks([])
            for spine in iax.spines.values():
                spine.set_edgecolor("#444")
                spine.set_linewidth(0.6)
            ax.add_patch(Rectangle((z0x, z0y), zoom_size, zoom_size,
                                   fill=False, edgecolor="#444",
                                   linewidth=0.6))

            # Per-panel quantities annotated inside the panel; the
            # size goes in the column header and the mode in the row
            # label so neither is repeated across the grid.
            ax.text(0.03, 0.97,
                    fr"({tags[r * ncol + c]}) "
                    fr"$\langle\varphi\rangle={phi[im, c]:.2f}$, "
                    fr"$n_{{\rm cl}}={n_clusters}$",
                    transform=ax.transAxes, fontsize=6,
                    fontweight="bold", ha="left", va="top",
                    bbox=dict(facecolor=style.CREAM, alpha=0.9,
                              edgecolor="none", pad=1))
            if r == 0:
                ax.set_title(fr"$L = {int(L)}$", fontsize=8)
            if c == 0:
                ax.set_ylabel(row_title.get(order[r], order[r]),
                              fontsize=8)
    fig.tight_layout(rect=(0.0, 0.0, 0.93, 1.0))
    cbar = fig.colorbar(last_q, ax=axes, orientation="vertical",
                        fraction=0.020, pad=0.02)
    cbar.set_label(r"local speed $v_i$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    _save(fig, "fig_double_cluster_map.pdf")


def fig_double_cluster_psd(npz_path: Path):
    """Dense-phase cluster-size distribution P(s) for every mode
    across the order--disorder window, all curves on one axes.
    Colour encodes the mode (paper palette); the line style of each
    curve encodes the noise level eta (solid to dotted for increasing
    eta). The two fixed-speed references form no dense phase at any
    eta; the motility-active modes keep a heavy-tailed dense phase
    throughout."""
    z = np.load(npz_path, allow_pickle=True)
    labels = [s if isinstance(s, str) else s.decode()
              for s in z["labels"]]
    etas = z["etas"]
    centers = z["centers"]
    pdf = z["pdf"]
    smax = z["smax_frac"]            # (n_modes, n_eta, n_seeds)
    n_modes, n_eta = len(labels), len(etas)

    # noise level -> distinct dash pattern (low eta solid, high eta sparse)
    eta_styles = ["solid", (0, (5, 1)), (0, (4, 1, 1, 1)), (0, (1, 1.4))]

    fig, ax = plt.subplots(figsize=(style.DOUBLE_COL[0],
                                    style.DOUBLE_COL[0] * 0.52))
    for im, m in enumerate(labels):
        for ie in range(n_eta):
            y = pdf[im, ie]
            ok = y > 0
            if ok.any():
                ax.plot(centers[ok], y[ok],
                        color=PALETTE[m], lw=1.1, alpha=0.9,
                        ls=eta_styles[ie % len(eta_styles)])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"dense-cluster size $s$ (particles)")
    ax.set_ylabel(r"$P(s)$")

    # Two legends in the cleared lower-left/upper-right of a decaying
    # P(s): one keying mode to hue, one keying eta to line style.
    from matplotlib.lines import Line2D
    mode_handles = [Line2D([], [], color=PALETTE[m], lw=2,
                           label=_disp(m)) for m in labels]
    eta_handles = [Line2D([], [], color="0.30", lw=1.4,
                          ls=eta_styles[ie % len(eta_styles)],
                          label=fr"$\eta = {etas[ie]:g}$")
                   for ie in range(n_eta)]
    leg1 = ax.legend(handles=mode_handles, fontsize=7, frameon=False,
                     loc="upper right", title="mode")
    ax.add_artist(leg1)
    ax.legend(handles=eta_handles, fontsize=7, frameon=False,
              loc="lower left", title="noise", ncol=2)
    fig.tight_layout()
    _save(fig, "fig_double_cluster_psd.pdf")


def fig_double_gnf(npz_path: Path):
    """Giant number fluctuations: Var(N_box) vs <N_box> per mode
    with the fitted exponent zeta in Var ~ <N>^{2 zeta}. The
    Poisson line zeta = 1/2 is drawn for reference."""
    z = np.load(npz_path, allow_pickle=True)
    labels = [s if isinstance(s, str) else s.decode()
              for s in z["labels"]]
    mean_N = z["mean_N"]
    var_N = z["var_N"]
    zeta = z["zeta"]

    fig, ax = plt.subplots(figsize=(style.SINGLE_COL[0] * 1.5,
                                    style.SINGLE_COL[1] * 1.4))
    for im, m in enumerate(labels):
        mk = var_N[im] > 0
        ax.plot(mean_N[im, mk], var_N[im, mk], "o-", color=PALETTE[m],
                ms=4, lw=1.0,
                label=fr"{_disp(m)}  ($\zeta={zeta[im]:.2f}$)")
    # Poisson reference Var = <N>
    xr = np.array([mean_N[mean_N > 0].min(), mean_N.max()])
    ax.plot(xr, xr, "k--", lw=0.7, label=r"Poisson ($\zeta=1/2$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\langle N_{\rm box}\rangle$")
    ax.set_ylabel(r"$\mathrm{Var}(N_{\rm box})$")
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    fig.tight_layout()
    _save(fig, "fig_double_gnf.pdf")


def fig_double_cluster_summary(npz_clusters: Path, npz_gnf: Path,
                               npz_psd: Path):
    """Merged dense-phase structure figure (former Figs. on cluster
    statistics, number fluctuations, and the cluster-size
    distribution). Panels (a)-(c): seed-averaged cluster count, max
    and mean cluster size per mode with +-1 SE bars. Panel (d):
    giant number fluctuations Var(N_box) vs <N_box> with the fitted
    exponent zeta. Panel (e): dense-phase P(s), hue keying the mode
    and line style the noise level eta."""
    from matplotlib.lines import Line2D

    # --- cluster summary statistics ---
    zc = np.load(npz_clusters, allow_pickle=True)
    clabels = [s if isinstance(s, str) else s.decode()
               for s in zc["labels"]]
    n_clusters = zc["n_per_seed"].astype(float)
    max_size = zc["max_per_seed"].astype(float)
    mean_size = zc["mean_per_seed"].astype(float)
    n_seeds = n_clusters.shape[1]

    def _ms(arr):
        return (arr.mean(axis=1),
                arr.std(axis=1, ddof=1) / np.sqrt(n_seeds))

    counts_m, counts_se = _ms(n_clusters)
    maxsz_m, maxsz_se = _ms(max_size)
    meansz_m, meansz_se = _ms(mean_size)
    disp = [_disp(m) for m in clabels]
    xpos = np.arange(len(clabels))
    ccolors = [PALETTE[m] for m in clabels]

    # --- GNF ---
    zg = np.load(npz_gnf, allow_pickle=True)
    glabels = [s if isinstance(s, str) else s.decode()
               for s in zg["labels"]]
    mean_N = zg["mean_N"]; var_N = zg["var_N"]; zeta = zg["zeta"]

    # --- P(s) ---
    zp = np.load(npz_psd, allow_pickle=True)
    plabels = [s if isinstance(s, str) else s.decode()
               for s in zp["labels"]]
    etas = zp["etas"]; centers = zp["centers"]; pdf = zp["pdf"]
    eta_styles = ["solid", (0, (5, 1)), (0, (4, 1, 1, 1)), (0, (1, 1.4))]

    fig = plt.figure(figsize=(style.DOUBLE_COL[0], 5.0))
    gs = fig.add_gridspec(2, 3)

    # Panels (a)-(c): bar summaries.
    bar_panels = (
        (fig.add_subplot(gs[0, 0]), counts_m, counts_se,
         "(a) cluster count"),
        (fig.add_subplot(gs[0, 1]), maxsz_m, maxsz_se,
         "(b) max cluster size"),
        (fig.add_subplot(gs[0, 2]), meansz_m, meansz_se,
         "(c) mean cluster size"),
    )
    for ax, m, se, ylab in bar_panels:
        ax.bar(xpos, m, yerr=se, color=ccolors, capsize=3,
               alpha=0.85, error_kw=dict(lw=0.8))
        ax.set_xticks(xpos)
        ax.set_xticklabels(disp, rotation=30, ha="right", fontsize=6)
        ax.set_ylabel(ylab, fontsize=8)

    # Panel (d): GNF.
    ax = fig.add_subplot(gs[1, 0])
    for im, m in enumerate(glabels):
        mk = var_N[im] > 0
        ax.plot(mean_N[im, mk], var_N[im, mk], "o-", color=PALETTE[m],
                ms=3, lw=1.0,
                label=fr"{_disp(m)} ($\zeta={zeta[im]:.2f}$)")
    xr = np.array([mean_N[mean_N > 0].min(), mean_N.max()])
    ax.plot(xr, xr, "k--", lw=0.7, label=r"Poisson")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\langle N_{\rm box}\rangle$")
    ax.set_ylabel(r"$\mathrm{Var}(N_{\rm box})$")
    ax.legend(fontsize=5.5, loc="upper left", frameon=False)
    ax.set_title("(d)", fontsize=9, loc="left")

    # Panel (e): P(s) overlay, spanning the remaining two cells.
    axe = fig.add_subplot(gs[1, 1:])
    for im, m in enumerate(plabels):
        for ie in range(len(etas)):
            y = pdf[im, ie]
            ok = y > 0
            if ok.any():
                axe.plot(centers[ok], y[ok], color=PALETTE[m], lw=1.1,
                         alpha=0.9,
                         ls=eta_styles[ie % len(eta_styles)])
    axe.set_xscale("log"); axe.set_yscale("log")
    axe.set_xlabel(r"dense-cluster size $s$ (particles)")
    axe.set_ylabel(r"$P(s)$")
    axe.set_title("(e)", fontsize=9, loc="left")
    mode_handles = [Line2D([], [], color=PALETTE[m], lw=2,
                           label=_disp(m)) for m in plabels]
    eta_handles = [Line2D([], [], color="0.30", lw=1.4,
                          ls=eta_styles[ie % len(eta_styles)],
                          label=fr"$\eta = {etas[ie]:g}$")
                   for ie in range(len(etas))]
    leg1 = axe.legend(handles=mode_handles, fontsize=6, frameon=False,
                      loc="upper right", title="mode")
    axe.add_artist(leg1)
    axe.legend(handles=eta_handles, fontsize=6, frameon=False,
               loc="lower left", title="noise", ncol=2)

    fig.tight_layout()
    _save(fig, "fig_double_cluster_summary.pdf")


def fig_double_hysteresis(npz_path: Path):
    """Quasi-static eta ramp up (filled circles, solid) and down
    (open squares, dashed) at L=64, all four modes overlaid on a
    single axes with hue keying the mode. Overlapping up/down
    branches and negligible loop areas indicate a continuous
    transition."""
    from matplotlib.lines import Line2D

    z = np.load(npz_path, allow_pickle=True)
    labels = [s if isinstance(s, str) else s.decode()
              for s in z["labels"]]
    eta = z["eta"]
    phi_up = z["phi_up"]; phi_up_se = z["phi_up_se"]
    phi_dn = z["phi_down"]; phi_dn_se = z["phi_down_se"]
    loop = z["loop_area"]

    fig, ax = plt.subplots(figsize=(style.SINGLE_COL[0] * 1.5,
                                    style.SINGLE_COL[1] * 1.15))
    mode_handles = []
    for im, m in enumerate(labels):
        c = PALETTE[m]
        ax.errorbar(eta, phi_up[im], yerr=phi_up_se[im], fmt="o-",
                    color=c, ms=3, lw=1.0, capsize=2)
        ax.errorbar(eta, phi_dn[im], yerr=phi_dn_se[im], fmt="s--",
                    color=c, ms=3, lw=1.0, capsize=2, alpha=0.6,
                    mfc="white")
        mode_handles.append(
            Line2D([], [], color=c, lw=1.6,
                   label=fr"{_disp(m)} (area $={loop[im]:.3f}$)"))
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\langle\varphi\rangle$")
    # Two legends: hue keys the mode, marker keys the ramp direction.
    ramp_handles = [
        Line2D([], [], color="0.30", lw=1.0, marker="o", ls="-",
               ms=4, label="up-ramp"),
        Line2D([], [], color="0.30", lw=1.0, marker="s", ls="--",
               ms=4, mfc="white", label="down-ramp"),
    ]
    leg_modes = ax.legend(handles=mode_handles, fontsize=6.5,
                          frameon=False, loc="lower left")
    ax.add_artist(leg_modes)
    ax.legend(handles=ramp_handles, fontsize=6.5, frameon=False,
              loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_double_hysteresis.pdf")


def fig_double_autocorr(npz_path: Path):
    r"""Polar/heading autocorrelation $C(\tau)$ for the four
    heavy-tailed modes at $L = 30$. Panel (a) shows the
    dense-quartile per-particle $\langle\cos\Delta\theta\rangle$
    versus $\tau$ for the four modes with seed-standard-error
    bands. Panel (b) shows the gap full $-$ motility-only
    against $\tau$, both for dense and dilute quartiles, with
    z-scores annotated for the dense gap."""
    z = np.load(npz_path, allow_pickle=True)
    labels = [str(s) if isinstance(s, str) else s.decode()
              for s in z["labels"]]
    taus = z["taus"]
    C_dense = z["C_dense_per_seed"]      # (n_modes, n_seeds, n_taus)
    C_dilute = z["C_dilute_per_seed"]
    n_seeds = C_dense.shape[1]

    mean_dense = C_dense.mean(axis=1)
    se_dense = C_dense.std(axis=1, ddof=1) / np.sqrt(n_seeds)
    mean_dilute = C_dilute.mean(axis=1)
    se_dilute = C_dilute.std(axis=1, ddof=1) / np.sqrt(n_seeds)

    fig, axes = plt.subplots(1, 2,
                              figsize=(style.DOUBLE_COL[0], 2.8))

    ax = axes[0]
    for im, lbl in enumerate(labels):
        c = PALETTE.get(lbl, "#1f4ea1")
        ax.plot(taus, mean_dense[im], "o-", color=c, lw=1.2, ms=3,
                label=_disp(lbl))
        ax.fill_between(taus,
                         mean_dense[im] - se_dense[im],
                         mean_dense[im] + se_dense[im],
                         color=c, alpha=0.20, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"lag $\tau$ (steps)")
    ax.set_ylabel(r"$\langle\cos[\theta_i(t+\tau) - \theta_i(t)]"
                  r"\rangle_{\rm dense}$")
    ax.set_title(r"(a) dense-quartile heading autocorrelation",
                 fontsize=8)
    ax.axhline(0.0, color="grey", lw=0.4)
    ax.legend(fontsize=7, frameon=False, loc="upper right")

    # Panel (b): gap full - motility, dense and dilute.
    i_full = labels.index("full")
    i_mot = labels.index("v3_limit")
    diff_dense = (C_dense[i_full] - C_dense[i_mot])
    diff_dilute = (C_dilute[i_full] - C_dilute[i_mot])
    g_dense_mean = diff_dense.mean(axis=0)
    g_dense_se = diff_dense.std(axis=0, ddof=1) / np.sqrt(n_seeds)
    g_dilute_mean = diff_dilute.mean(axis=0)
    g_dilute_se = diff_dilute.std(axis=0, ddof=1) / np.sqrt(n_seeds)

    ax = axes[1]
    c_full = PALETTE["full"]
    c_mot = PALETTE["v3_limit"]
    ax.fill_between(taus,
                     g_dense_mean - g_dense_se,
                     g_dense_mean + g_dense_se,
                     color=c_full, alpha=0.20, lw=0)
    ax.plot(taus, g_dense_mean, "o-", color=c_full, lw=1.4, ms=3,
            label="dense quartile")
    ax.fill_between(taus,
                     g_dilute_mean - g_dilute_se,
                     g_dilute_mean + g_dilute_se,
                     color=c_mot, alpha=0.20, lw=0)
    ax.plot(taus, g_dilute_mean, "s--", color=c_mot, lw=1.0, ms=3,
            label="dilute quartile")
    ax.axhline(0.0, color="grey", lw=0.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"lag $\tau$ (steps)")
    ax.set_ylabel(r"$\Delta C(\tau) = C_{\rm full} - C_{\rm motility}$")
    ax.set_title(r"(b) full $-$ motility-only gap",
                 fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    # z-score annotation on a few representative lags.
    for j_tau in (2, 6, 9):       # tau = 5, 100, 1000
        z_val = (g_dense_mean[j_tau]
                  / max(g_dense_se[j_tau], 1e-9))
        ax.annotate(fr"$z\!=\!{z_val:.0f}$",
                     xy=(taus[j_tau], g_dense_mean[j_tau]),
                     xytext=(0, -10), textcoords="offset points",
                     fontsize=6, ha="center", va="top",
                     color=c_full)
    fig.tight_layout()
    _save(fig, "fig_double_autocorr.pdf")


def _pick(stem: str) -> Path:
    """Prefer the no-cone version of an npz file when it exists.

    The figure pipeline now defaults to the omnidirectional
    simulator's outputs. Any file with the same stem and the
    ``_nocone`` suffix overrides the legacy with-cone version.
    """
    no_cone = DATA / f"{stem}_nocone.npz"
    legacy = DATA / f"{stem}.npz"
    if no_cone.exists():
        return no_cone
    return legacy


def main():
    # Fig. 1 (fig_double_schematic.pdf) is now produced by
    # flock_simulator/scripts/make_setup_figure.py (Vicsek--Couzin
    # setup + alpha-stable pdfs + shared sigmoid + worked rules), not
    # by the legacy two-panel fig_double_schematic() below.
    fig_double_pilot(_pick("double_pilot"))
    fig_double_snapshot(_pick("double_snapshot"))
    # fig_double_3regimes was promoted out: the two-mode snapshot now
    # serves as the early real-space figure.
    fig_double_plane(_pick("double_plane"), _pick("double_plane_L30"))
    npz_op = _pick("double_orderpdf")
    npz_op_large = _pick("double_orderpdf_largeL")
    npz_op_128 = _pick("double_orderpdf_L128")
    fig_double_orderpdf(npz_op, npz_op_large, npz_op_128)
    fig_double_profile(_pick("double_profile"))
    npz_clmap = DATA / "double_cluster_snaps_multiL_nocone.npz"
    fig_double_cluster_map(npz_clmap if npz_clmap.exists()
                           else _pick("double_snapshot"))
    fig_double_gr(_pick("double_gr_hs"), _pick("double_gr_decay"))
    npz_lfine = DATA / "double_Lfine_allmodes_nocone.npz"
    fig_double_Lfine_gap(
        npz_lfine if npz_lfine.exists() else _pick("double_Lfine"),
        _pick("double_pilot"),
        _pick("double_L64"),
        _pick("double_L90"),
        _pick("double_L128"),
    )
    npz_psd = DATA / "double_cluster_psd_eta_nocone.npz"
    if not npz_psd.exists():
        npz_psd = _pick("double_cluster_psd")
    npz_gnf = _pick("double_gnf")
    npz_hyst = DATA / "double_hysteresis_allmodes_nocone.npz"
    if not npz_hyst.exists():
        npz_hyst = _pick("double_hysteresis")
    fig_double_cluster_summary(_pick("double_clusters_hs"), npz_gnf, npz_psd)
    fig_double_hysteresis(npz_hyst)
    fig_double_autocorr(_pick("double_autocorr"))
    # The decoupled-2D, sigma-sweep and topological-kernel figures
    # (Figs. 14-16) are produced by the analyse_*.py scripts, not by
    # this renderer.


if __name__ == "__main__":
    main()
