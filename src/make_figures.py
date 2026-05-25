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


def _save(fig, name):
    out = FIGS / name
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
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
                inset_frac=0.38, arrow_len=0.45, speeds=None):
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
    iax.set_facecolor(style.CREAM)
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

    # (b) Couzin zonal cartoon -- omnidirectional vision (no blind sector)
    ax = axes[1]
    ax.set_aspect("equal")
    R_r, R_a = 0.5, 0.7
    cx, cy = 0.0, 0.0
    ax.add_patch(patches.Annulus((cx, cy), R_a, R_a - R_r,
                                  facecolor="#1f4ea1", alpha=0.15,
                                  edgecolor="#1f4ea1", lw=0.6))
    ax.add_patch(patches.Circle((cx, cy), R_r,
                                 facecolor="#e07a3a", alpha=0.18,
                                 edgecolor="#e07a3a", lw=0.6))
    ax.arrow(cx, cy, 0.45, 0, head_width=0.07, head_length=0.06,
             fc="black", ec="black", lw=0.8)
    ax.scatter([cx], [cy], c="black", s=14, zorder=5)
    rng = np.random.default_rng(7)
    for _ in range(12):
        r = rng.uniform(R_r, R_a * 0.95)
        th = rng.uniform(-np.pi, np.pi)
        ax.scatter([r * np.cos(th)], [r * np.sin(th)],
                    c="#1f4ea1", s=12, alpha=0.8, zorder=4)
    for _ in range(2):
        r = rng.uniform(0.05, R_r * 0.9)
        th = rng.uniform(-np.pi, np.pi)
        ax.scatter([r * np.cos(th)], [r * np.sin(th)],
                    c="#e07a3a", s=12, alpha=0.85, zorder=4)
    ax.text(0.0, R_a * 0.98, "alignment", fontsize=7,
            color="#1f4ea1", ha="center")
    ax.text(0.0, -R_a * 0.5, "repulsion", fontsize=7,
            color="#e07a3a", ha="center")
    ax.set_xlim(-R_a * 1.15, R_a * 1.15)
    ax.set_ylim(-R_a * 1.15, R_a * 1.15)
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

    # Merge L=64 and L=90 if present.
    for big_name in ("double_L64.npz", "double_L90.npz"):
        big_path = DATA / big_name
        if not big_path.exists():
            continue
        big = np.load(big_path, allow_pickle=True)
        Ls = np.concatenate([Ls, [float(big["L"])]])
        # The L90 file carries 5 modes (vicsek_gauss + 4 ht); the
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
    ax.set_title(r"(a)", fontsize=9, loc="left")

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
    ax.set_title(r"(b)", fontsize=9, loc="left")

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
    ax.set_title(r"(c)", fontsize=9, loc="left")

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
    ax.set_title(r"(d)", fontsize=9, loc="left")

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
    ax.set_title(r"(e)", fontsize=9, loc="left")

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
    ax.set_title(r"(f)", fontsize=9, loc="left")

    # Shared legend at the top of the figure, no stack on panel a.
    handles = [plt.Line2D([0], [0], color=PALETTE[m], marker="o",
                            ms=4, label=LABELS[m])
                for m in modes]
    fig.legend(handles=handles, loc="upper center",
                 bbox_to_anchor=(0.5, 0.99),
                 ncol=4, fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.07,
                        wspace=0.22, hspace=0.28)
    _save(fig, "fig_double_pilot.pdf")


def fig_double_snapshot(npz_path: Path):
    """Two-row snapshot grid: v3-limit (top) vs full (bottom), three
    eta values (ordered / near-critical / disordered). v1-style
    monochrome quiver on cream: each panel reads as a clean
    early-Vicsek snapshot, and the mode/eta contrast is carried by
    the geometry of the dense phase rather than by per-particle
    colour."""
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
    arrow_len = 0.45
    last_q = None
    for im in range(nm):
        for ic in range(nc):
            ax = axes[im, ic]
            _cream_panel(ax)
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
                        speeds=v[im, ic])
            nice_c = _disp(case_labels[ic])
            nice_m = _disp(mode_labels[im])
            ax.set_title(
                fr"{nice_m} | {nice_c}, $\eta={eta[ic]:g}$"
                "\n"
                fr"$\langle\varphi\rangle = {phi[im, ic]:.2f}$",
                fontsize=7,
            )
            last_q = q
    fig.tight_layout(rect=(0.0, 0.0, 0.94, 1.0))
    cbar = fig.colorbar(last_q, ax=axes, orientation="vertical",
                        fraction=0.020, pad=0.02)
    cbar.set_label(r"local speed $v_i$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    _cream_save(fig, "fig_double_snapshot.pdf")


def fig_double_plane(npz_path: Path):
    """Heat maps of chi_peak and sep_peak in the (n_star, s) plane
    for the FULL two-feedback model."""
    z = np.load(npz_path, allow_pickle=True)
    n_stars = z["n_stars"]
    slopes = z["slopes"]
    chi_peak = z["chi_peak"]
    sep_peak = z["sep_peak"]
    L = float(z["L"])

    fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE_COL[0], 2.6))
    panels = (
        (axes[0], chi_peak, fr"(a) $\chi_{{\max}}$ (full mode, $L = {L:g}$)",
         "viridis", r"$\chi_{\max}$"),
        (axes[1], sep_peak, fr"(b) $s_{{\rm sep}}$",
         "magma", r"$s_{\rm sep}$"),
    )
    for ax, data, title, cmap, vlabel in panels:
        im = ax.imshow(data, origin="lower", aspect="auto",
                       cmap=cmap,
                       extent=[slopes.min() - 0.5,
                               slopes.max() + 0.5,
                               n_stars.min() - 0.5,
                               n_stars.max() + 0.5])
        ax.set_xticks(slopes)
        ax.set_yticks(n_stars)
        ax.set_xlabel(r"sigmoid slope $s$")
        ax.set_ylabel(r"threshold $n_\star$")
        ax.set_title(title, fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(vlabel, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        for i, n_star in enumerate(n_stars):
            for j, sl in enumerate(slopes):
                ax.text(sl, n_star, f"{data[i, j]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white")
    fig.tight_layout()
    _save(fig, "fig_double_plane.pdf")


def fig_double_orderpdf(npz_path: Path):
    """P(<phi>) for v3 limit and full mode at L = 30."""
    z = np.load(npz_path, allow_pickle=True)
    labels = [str(s) if isinstance(s, str) else s.decode()
              for s in z["labels"]]
    phi_traj = z["phi_traj"]
    eta_arr = z["eta"]

    from scipy.stats import skew, kurtosis, gaussian_kde

    fig, axes = plt.subplots(1, len(labels),
                             figsize=(style.DOUBLE_COL[0], 2.6),
                             sharey=True)
    if len(labels) == 1:
        axes = [axes]
    bins = np.linspace(0, 1, 60)
    grid = np.linspace(0, 1, 200)
    # Determine a common y-limit from the maximum density across
    # both panels so the visual comparison is honest.
    y_max = 0.0
    for ic, lbl in enumerate(labels):
        h, _ = np.histogram(phi_traj[ic], bins=bins, density=True)
        y_max = max(y_max, h.max())
    for ic, lbl in enumerate(labels):
        ax = axes[ic]
        c = PALETTE.get(lbl, "#1f4ea1")
        ax.hist(phi_traj[ic], bins=bins, density=True, color=c,
                alpha=0.55, edgecolor="black", linewidth=0.3)
        kde = gaussian_kde(phi_traj[ic], bw_method="scott")
        ax.plot(grid, kde(grid), "-", color=c, lw=1.4)
        mu = float(phi_traj[ic].mean())
        sd = float(phi_traj[ic].std())
        sk = float(skew(phi_traj[ic]))
        ku = float(kurtosis(phi_traj[ic]))
        u4 = 1.0 - (phi_traj[ic] ** 4).mean() / (
            3.0 * (phi_traj[ic] ** 2).mean() ** 2)
        ax.axvline(mu, color="black", lw=0.6, ls="--", alpha=0.7)
        ax.text(0.97, 0.95,
                fr"$\mu = {mu:.2f}$" "\n"
                fr"$\sigma = {sd:.2f}$" "\n"
                fr"skew $= {sk:+.2f}$" "\n"
                fr"kurt $= {ku:+.2f}$" "\n"
                fr"$U_4 = {u4:.2f}$",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="none", pad=2))
        ax.set_xlabel(r"$\langle\varphi\rangle$")
        ax.set_xlim(0, 1)
        nice = _disp(lbl)
        ax.set_title(fr"{nice}, $\eta = {eta_arr[ic]:g}$",
                     fontsize=8)
        if ic == 0:
            ax.set_ylabel(r"$P(\langle\varphi\rangle)$")
        ax.text(-0.18, 1.04, f"({chr(97 + ic)})",
                transform=ax.transAxes,
                fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.1 * y_max)
    fig.tight_layout()
    _save(fig, "fig_double_orderpdf.pdf")


def fig_double_profile(npz_path: Path):
    """Polar-axis density profile for the four modes."""
    z = np.load(npz_path, allow_pickle=True)
    labels = [str(s) if isinstance(s, str) else s.decode()
              for s in z["labels"]]
    centers = z["centers"]
    profiles = z["profiles"]
    v_profiles = z["v_profiles"]
    eta_arr = z["eta"]
    band_idx = z["band_idx"]
    L = float(z["params"][1])

    n = len(labels)
    fig, axes = plt.subplots(2, n,
                             figsize=(style.DOUBLE_COL[0], 4.0),
                             sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)
    for ic in range(n):
        col = PALETTE.get(labels[ic], "#1f4ea1")
        ax_d = axes[0, ic]
        norm = profiles[ic] / profiles[ic].mean()
        ax_d.plot(centers, norm, "-", color=col, lw=1.4)
        ax_d.axhline(1.0, ls=":", c="grey", lw=0.6)
        ax_d.set_ylim(0, max(2.0, 1.1 * norm.max()))
        ax_d.set_xlim(0, L)
        ax_d.set_title(
            fr"{_disp(labels[ic])}, "
            fr"$\eta={eta_arr[ic]:g}$, "
            fr"$b={band_idx[ic]:.2f}$",
            fontsize=7,
        )
        if ic == 0:
            ax_d.set_ylabel(
                r"$\sigma(x_\parallel)/\langle\sigma\rangle$"
            )

        ax_v = axes[1, ic]
        ax_v.plot(centers, v_profiles[ic], "-", color=col, lw=1.4)
        ax_v.axhline(0.05, ls=":", c="grey", lw=0.5)
        ax_v.axhline(0.005, ls=":", c="grey", lw=0.5)
        ax_v.set_xlabel(r"$x_\parallel$")
        ax_v.set_ylim(0, 0.06)
        ax_v.set_xlim(0, L)
        if ic == 0:
            ax_v.set_ylabel(r"$\langle v_i\rangle(x_\parallel)$")
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


def fig_double_gr(npz_path: Path):
    """Heading correlation $g(r)$ in dense vs dilute regions, with
    seed-level standard-error bands when per-seed data is
    available (HS files), or the bare curve otherwise."""
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

    fig, axes = plt.subplots(1, 2,
                              figsize=(style.DOUBLE_COL[0], 2.8),
                              sharey=True)
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
    for ax in axes:
        ax.axhline(0.0, ls=":", color="grey", lw=0.6)
        ax.axvline(0.7, ls=":", color="grey", lw=0.6)
        ax.set_xlabel(r"distance $r$")
        ax.set_xlim(0.5, r_centers[-1])
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
            xytext=(r_centers[0] + 0.6, gr_dense[i_f, 0] + 0.05),
            fontsize=7, color=PALETTE["full"],
            arrowprops=dict(arrowstyle="-", lw=0.6,
                              color=PALETTE["full"]))
    axes[0].legend(fontsize=6, loc="upper right", frameon=False)
    axes[0].text(-0.18, 1.04, "(a)", transform=axes[0].transAxes,
                  fontsize=10, fontweight="bold")
    axes[1].text(-0.18, 1.04, "(b)", transform=axes[1].transAxes,
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

    # Pilot has v3_limit (motility) and full at index 2 and 3.
    pilot_modes = [str(s) if isinstance(s, str) else s.decode()
                   for s in pilot["modes"]]
    i_mot_pilot = pilot_modes.index("v3_limit")
    i_full_pilot = pilot_modes.index("full")

    pts: list[tuple] = []   # (L, mode, s_sep_max)

    # Pilot covers L = 15, 22, 30, 45.
    for iL, L in enumerate(pilot["Ls"]):
        pts.append((float(L), "motility",
                     float(pilot["s_sep"][i_mot_pilot, iL].max())))
        pts.append((float(L), "full",
                     float(pilot["s_sep"][i_full_pilot, iL].max())))

    # Big files: L = 64, 90, 128.
    for big in (big64, big90, big128):
        L = float(big["L"])
        i_mot = big_idx(big, "v3_limit")
        i_full = big_idx(big, "full")
        if i_mot is not None:
            pts.append((L, "motility",
                         float(big["s_sep"][i_mot].max())))
        if i_full is not None:
            pts.append((L, "full",
                         float(big["s_sep"][i_full].max())))

    # Fine scan: L = 38, 50, 60, 75, 105.
    fine_modes = [str(s) if isinstance(s, str) else s.decode()
                  for s in fine["modes"]]
    i_mot_f = fine_modes.index("motility")
    i_full_f = fine_modes.index("full")
    for iL, L in enumerate(fine["Ls"]):
        pts.append((float(L), "motility",
                     float(fine["s_sep"][i_mot_f, iL].max())))
        pts.append((float(L), "full",
                     float(fine["s_sep"][i_full_f, iL].max())))

    # Sort and split.
    pts.sort()
    L_full = sorted({p[0] for p in pts if p[1] == "full"})
    s_full = [next(p[2] for p in pts if p[0] == L and p[1] == "full")
              for L in L_full]
    L_mot = sorted({p[0] for p in pts if p[1] == "motility"})
    s_mot = [next(p[2] for p in pts if p[0] == L and p[1] == "motility")
              for L in L_mot]
    L_arr = np.array(L_full)
    s_full_arr = np.array(s_full)
    s_mot_arr = np.array(s_mot)
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
    ax.plot(L_arr, s_mot_arr, "o", color=PALETTE["v3_limit"],
             label=DISPLAY["v3_limit"], ms=4)
    ax.plot(L_arr, s_full_arr, "s", color=PALETTE["full"],
             label=DISPLAY["full"], ms=4)
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
    style of the early-Vicsek snapshot figures: uniform-color
    particles, white-on-cream quiver, $\\eta$ and $\\langle\\varphi
    \\rangle$ in the panel titles."""
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

    fig, axes = plt.subplots(1, 3,
                              figsize=(style.DOUBLE_COL[0], 3.2))
    arrow_len = 0.45
    last_q = None
    for ic, ax in enumerate(axes):
        _cream_panel(ax)
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
                    speeds=vs[im_full, ic])
        ax.set_title(
            f"{nice_cases[ic]}\n"
            fr"$\eta={eta[ic]:g}$, "
            fr"$\langle\varphi\rangle={phi[im_full, ic]:.2f}$",
            fontsize=8,
        )
        last_q = q
    fig.tight_layout(rect=(0.0, 0.0, 0.93, 1.0))
    cbar = fig.colorbar(last_q, ax=axes, orientation="vertical",
                        fraction=0.030, pad=0.02)
    cbar.set_label(r"local speed $v_i$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    _cream_save(fig, "fig_double_3regimes.pdf")


def fig_double_cluster_map(npz_path: Path):
    """Real-space cluster identification at the near-critical
    point: motility-only versus double-adaptive. Particles in
    dense bins (above 1.5x median bin occupancy) are coloured
    by their connected-component cluster ID; particles in
    dilute bins are grey. The 10x10 dense grid is overlaid as a
    light alpha shading."""
    from scipy.ndimage import label as nd_label
    import matplotlib.colors as mcolors

    z = np.load(npz_path, allow_pickle=True)
    mode_labels = [str(s) if isinstance(s, str) else s.decode()
                   for s in z["mode_labels"]]
    case_labels = [str(s) if isinstance(s, str) else s.decode()
                   for s in z["case_labels"]]
    x = z["x"]; y = z["y"]; theta = z["theta"]
    eta = z["eta"]; phi = z["phi"]
    L = float(z["params"][1])

    n_bin = 10
    factor = 1.5
    arrow_len = 0.45

    im_mot = mode_labels.index("v3_limit") if "v3_limit" in mode_labels else 0
    im_full = mode_labels.index("full")
    ic = case_labels.index("near_critical")

    fig, axes = plt.subplots(1, 2,
                              figsize=(style.DOUBLE_COL[0], 3.4))
    panel_data = [
        (axes[0], im_mot, "motility adaptive"),
        (axes[1], im_full, "double-adaptive"),
    ]

    rng_palette = np.random.default_rng(0)

    for ax, im, mode_title in panel_data:
        xs = x[im, ic]
        ys = y[im, ic]
        ths = theta[im, ic]

        # Identify dense bins on a 10x10 grid with periodic-aware
        # connected components.
        H, _, _ = np.histogram2d(
            xs, ys, bins=[n_bin, n_bin], range=[[0, L], [0, L]])
        nz = H[H > 0]
        threshold = factor * np.median(nz) if len(nz) else 0.0
        mask = H > threshold
        tiled = np.tile(mask, (3, 3))
        labelled, _ = nd_label(tiled)
        central = labelled[n_bin:2 * n_bin, n_bin:2 * n_bin]
        cluster_ids = np.unique(central[central > 0])

        # Assign each particle to the cluster ID of its bin.
        bin_ix = np.clip((xs / L * n_bin).astype(int), 0, n_bin - 1)
        bin_iy = np.clip((ys / L * n_bin).astype(int), 0, n_bin - 1)
        particle_cid = central[bin_ix, bin_iy]   # 0 if dilute

        # Build a palette of distinct colours for clusters; grey
        # for dilute particles.
        n_clusters = len(cluster_ids)
        cmap = plt.get_cmap("tab20")
        colours = np.zeros((len(xs), 4))
        colours[:] = mcolors.to_rgba("#bbbbbb", alpha=0.5)
        for k, cid in enumerate(cluster_ids):
            sel = particle_cid == cid
            base = cmap(k % 20)
            colours[sel] = base

        # Light shading of dense bins as a background.
        ax.imshow(mask.T, extent=[0, L, 0, L], origin="lower",
                  cmap=mcolors.ListedColormap([(0, 0, 0, 0),
                                                (0.96, 0.86, 0.55,
                                                 0.18)]),
                  aspect="auto", interpolation="nearest")

        ax.scatter(xs, ys, c=colours, s=6, edgecolor="none",
                    zorder=2)
        ax.quiver(xs, ys, np.cos(ths), np.sin(ths),
                  color="white", alpha=0.55,
                  scale=1.0 / arrow_len, scale_units="xy",
                  angles="xy", width=0.0025,
                  headwidth=2.6, headlength=3.0,
                  zorder=3)
        ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"{mode_title}\n"
            fr"$\eta={eta[ic]:g}$, $\langle\varphi\rangle = "
            fr"{phi[im, ic]:.2f}$, "
            fr"$n_{{\rm cl}}={n_clusters}$",
            fontsize=8,
        )
    fig.tight_layout()
    _save(fig, "fig_double_cluster_map.pdf")


def fig_double_gr_decay(npz_path: Path):
    """Profile of the dense-quartile gap $\\Delta g(r)$ between
    the double-adaptive model and the motility-only ablation,
    over the full alignment range $r \\in [0.5, 6]$, for the
    four sizes at which the high-statistics measurement was
    run."""
    z = np.load(npz_path, allow_pickle=True)
    Ls = z["Ls"]
    r = z["r_centers"]
    gap = z["gap"]
    se = z["se"]

    fig, ax = plt.subplots(figsize=(style.SINGLE_COL[0] * 1.4,
                                      style.SINGLE_COL[1] * 1.3))
    cmap = plt.get_cmap("plasma")
    for iL, L in enumerate(Ls):
        c = cmap(0.15 + 0.7 * iL / max(len(Ls) - 1, 1))
        ax.fill_between(r, gap[iL] - se[iL], gap[iL] + se[iL],
                         color=c, alpha=0.20)
        ax.plot(r, gap[iL], "-", lw=1.4, color=c,
                 label=fr"$L={int(L)}$")
    # plateau reference line and metric alignment cutoff
    plateau = float(np.nanmean(gap))
    ax.axhline(plateau, color="black", ls="--", lw=0.5, alpha=0.5)
    ax.text(r[-1] * 0.98, plateau + 0.005,
             fr"$\Delta g \simeq {plateau:.2f}$",
             fontsize=7, ha="right", va="bottom", color="black")
    ax.axhline(0.0, color="grey", ls="-", lw=0.4)
    ax.axvline(0.7, color="grey", ls=":", lw=0.5)
    ax.text(0.72, ax.get_ylim()[0] + 0.01, r"$R_a$",
             fontsize=7, ha="left", va="bottom", color="grey")
    ax.set_xlabel(r"distance $r$")
    ax.set_ylabel(r"$\Delta g(r) = g_{\rm full}(r) - g_{\rm motility}(r)$"
                  r" in dense quartile")
    ax.set_title(r"Dense-quartile correlation gap profile",
                  fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig_double_gr_decay.pdf")


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


def fig_double_orderpdf_largeL(npz_path: Path,
                                npz_L128_path: Path | None = None):
    r"""Order-parameter PDF $P(\langle\varphi\rangle)$ for
    motility-only and double-adaptive at $L = 64$, $L = 90$, and
    optionally $L = 128$, confirming that the macroscopic mean
    shift is a scale-invariant signature of the double feedback
    and probing for late multimodality on a 10x longer
    trajectory."""
    from scipy.stats import skew, kurtosis, gaussian_kde
    z = np.load(npz_path, allow_pickle=True)
    labels = [str(s) if isinstance(s, str) else s.decode()
              for s in z["labels"]]
    Ls = list(z["Ls"])
    eta_per_case = z["eta_per_case"]
    phi_traj = z["phi_traj"]
    panel_data = []
    for iL, L in enumerate(Ls):
        panel_data.append((float(L),
                           float(eta_per_case[labels.index("motility")]),
                           float(eta_per_case[labels.index("full")]),
                           phi_traj[labels.index("motility"), iL],
                           phi_traj[labels.index("full"), iL]))
    if npz_L128_path is not None and npz_L128_path.exists():
        z2 = np.load(npz_L128_path, allow_pickle=True)
        labs2 = [str(s) if isinstance(s, str) else s.decode()
                 for s in z2["labels"]]
        eta2 = z2["eta_per_case"]
        traj2 = z2["phi_traj"]
        panel_data.append((float(z2["L"]),
                           float(eta2[labs2.index("motility")]),
                           float(eta2[labs2.index("full")]),
                           traj2[labs2.index("motility")],
                           traj2[labs2.index("full")]))

    fig, axes = plt.subplots(1, len(panel_data),
                              figsize=(style.DOUBLE_COL[0], 2.8),
                              sharey=False)
    if len(panel_data) == 1:
        axes = [axes]
    bins = np.linspace(0, 1, 60)
    grid = np.linspace(0, 1, 200)

    u4 = lambda a: 1.0 - (a ** 4).mean() / (3.0 * (a ** 2).mean() ** 2)
    for ip, (L, eta_mot, eta_full, m_mot, m_full) in enumerate(panel_data):
        ax = axes[ip]
        y_max = 0.0
        for lbl, data in (("motility", m_mot), ("full", m_full)):
            c = PALETTE.get(lbl, "#1f4ea1")
            ax.hist(data, bins=bins, density=True, color=c,
                    alpha=0.45, edgecolor="black", linewidth=0.3,
                    label=_disp(lbl))
            kde = gaussian_kde(data, bw_method="scott")
            curve = kde(grid)
            ax.plot(grid, curve, "-", color=c, lw=1.4)
            y_max = max(y_max, curve.max())
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
        ax.set_ylim(0, 1.15 * y_max)
        ax.set_title(fr"$L = {int(L)}$ "
                     fr"($\eta_{{\rm mot}}\!=\!{eta_mot:g}$, "
                     fr"$\eta_{{\rm full}}\!=\!{eta_full:g}$)",
                     fontsize=8)
        if ip == 0:
            ax.set_ylabel(r"$P(\langle\varphi\rangle)$")
            ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.text(-0.18, 1.04, f"({chr(97 + ip)})",
                transform=ax.transAxes,
                fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig_double_orderpdf_largeL.pdf")


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
    fig_double_schematic()
    npz = _pick("double_pilot")
    if npz.exists():
        fig_double_pilot(npz)
    else:
        print(f"[warn] {npz} not found -- run run_double_pilot.py first")
    npz_snap = _pick("double_snapshot")
    if npz_snap.exists():
        fig_double_snapshot(npz_snap)
    else:
        print(f"[warn] {npz_snap} not found -- "
              "run run_double_snapshot.py first")
    npz_plane = _pick("double_plane")
    if npz_plane.exists():
        fig_double_plane(npz_plane)
    else:
        print(f"[warn] {npz_plane} not found -- "
              "run run_double_plane.py first")
    npz_plane_L30 = _pick("double_plane_L30")
    if npz_plane_L30.exists():
        # Reuse the same figure layout for the L=30 refinement.
        z = npz_plane_L30
        z = np.load(z, allow_pickle=True)
        # Render via the existing fig_double_plane structure with
        # an alternate output name.
        n_stars = z["n_stars"]
        slopes_s = z["slopes"]
        chi_peak = z["chi_peak"]
        sep_peak = z["sep_peak"]
        L_ = float(z["L"])
        fig, axes = plt.subplots(1, 2,
                                  figsize=(style.DOUBLE_COL[0], 2.6))
        for ax, data, title, cmap, vlabel in (
            (axes[0], chi_peak, fr"(a) $\chi_{{\max}}$ (full mode, $L = {L_:g}$, 5 seeds)",
             "viridis", r"$\chi_{\max}$"),
            (axes[1], sep_peak, fr"(b) $s_{{\rm sep}}$",
             "magma", r"$s_{\rm sep}$"),
        ):
            im = ax.imshow(data, origin="lower", aspect="auto",
                            cmap=cmap,
                            extent=[slopes_s.min() - 0.5,
                                    slopes_s.max() + 0.5,
                                    n_stars.min() - 0.5,
                                    n_stars.max() + 0.5])
            ax.set_xticks(slopes_s)
            ax.set_yticks(n_stars)
            ax.set_xlabel(r"sigmoid slope $s$")
            ax.set_ylabel(r"threshold $n_\star$")
            ax.set_title(title, fontsize=8)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(vlabel, fontsize=8)
            cb.ax.tick_params(labelsize=7)
            for i, n_star in enumerate(n_stars):
                for j, sl in enumerate(slopes_s):
                    ax.text(sl, n_star, f"{data[i, j]:.2f}",
                            ha="center", va="center", fontsize=6,
                            color="white")
        fig.tight_layout()
        _save(fig, "fig_double_plane_L30.pdf")
    npz_op = _pick("double_orderpdf")
    if npz_op.exists():
        fig_double_orderpdf(npz_op)
    npz_pr = _pick("double_profile")
    if npz_pr.exists():
        fig_double_profile(npz_pr)
    # Cluster figure: the no-cone pipeline only stores the
    # high-statistics summary (n_cl, max, mean per seed). When the
    # _nocone hs file is available, draw a summary bar plot;
    # otherwise fall back to the legacy CCDF.
    npz_cl_hs = DATA / "double_clusters_hs_nocone.npz"
    npz_cl_legacy = DATA / "double_clusters.npz"
    if npz_cl_hs.exists():
        fig_double_clusters_hs(npz_cl_hs)
    elif npz_cl_legacy.exists():
        fig_double_clusters(npz_cl_legacy)
    # g(r): _pick prefers the hs file (per-seed bands).
    npz_gr_hs = DATA / "double_gr_hs_nocone.npz"
    npz_gr_legacy = DATA / "double_gr.npz"
    if npz_gr_hs.exists():
        fig_double_gr(npz_gr_hs)
    elif npz_gr_legacy.exists():
        fig_double_gr(npz_gr_legacy)
    npz_decay = _pick("double_gr_decay")
    if npz_decay.exists():
        fig_double_gr_decay(npz_decay)
    npz_snap = _pick("double_snapshot")
    if npz_snap.exists():
        fig_double_3regimes(npz_snap)
        fig_double_cluster_map(npz_snap)
    npz_lfine = _pick("double_Lfine")
    if npz_lfine.exists():
        fig_double_Lfine_gap(
            npz_lfine,
            _pick("double_pilot"),
            _pick("double_L64"),
            _pick("double_L90"),
            _pick("double_L128"),
        )
    npz_ac = _pick("double_autocorr")
    if npz_ac.exists():
        fig_double_autocorr(npz_ac)
    npz_oplargeL = _pick("double_orderpdf_largeL")
    if npz_oplargeL.exists():
        fig_double_orderpdf_largeL(npz_oplargeL,
                                   _pick("double_orderpdf_L128"))


if __name__ == "__main__":
    main()
