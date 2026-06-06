"""Figure 1 for version 4: Vicsek--Couzin setup and update rules,
adapted from the version-1 schematic and extended with the shared
sigmoid that defines the two-feedback model.

Single composite PDF (overwrites fig_double_schematic.pdf so the
manuscript include keeps working):
  (a) two-zone geometry (repulsion disc, alignment annulus, 360 vision);
  (b) symmetric alpha-stable noise pdfs (tails grow as alpha falls);
  (c) the shared sigmoid mapping the neighbour count n_i onto the speed
      v_i and the noise stability index alpha_i;
  (d)-(f) worked examples of the inertia, repulsion and alignment rules,
      each shown at t and t+dt.
The repulsion>alignment>inertia geometry and the alpha-stable kick are
identical to version 1; the cartoon zone radii are the v4 values
(R_r=0.5, R_a=0.7) inflated x1.2 for legibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import style as st  # noqa: E402
from flock_simulator.core.noise import stable_rvs_vector  # noqa: E402

FIG = ROOT / "figures"

BLUE = "#1f4ea1"          # neighbour / focal heading (v1 navy)
FOCAL_RED = "#c83a3a"
REP_COLOR = "#e07b7b"
ALI_COLOR = "#9bb8de"
V_PURPLE = "#7a3aa0"
A_GREEN = "#3aa040"

R_R = 0.5 * 1.2           # cartoon zone radii (v4 R_r=0.5, R_a=0.7, x1.2)
R_A = 0.7 * 1.2
V_VIS = 0.34              # visualisation step for the worked examples


# --------------------------------------------------------------------
# geometry helpers (ported from version1, sized for a composite panel)
# --------------------------------------------------------------------
def _draw_zones(ax):
    ax.add_patch(Wedge((0, 0), R_A, 0, 360, width=R_A - R_R,
                       facecolor=ALI_COLOR, alpha=0.55,
                       edgecolor="#3a4a78", lw=0.7, zorder=1))
    ax.add_patch(Wedge((0, 0), R_R, 0, 360,
                       facecolor=REP_COLOR, alpha=0.55,
                       edgecolor="#9c3a3a", lw=0.7, zorder=1))


def _particle(ax, pos, heading, *, color, ss, al, lw, label=None,
              loff=(0.05, 0.07), fs=8, alpha=1.0):
    x, y = float(pos[0]), float(pos[1])
    ax.annotate("", xy=(x + al * np.cos(heading), y + al * np.sin(heading)),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                alpha=alpha), zorder=6)
    ax.scatter([x], [y], s=ss, color=color, alpha=alpha,
               edgecolor="white", lw=0.7, zorder=7)
    if label:
        ax.text(x + loff[0], y + loff[1], label, fontsize=fs,
                color=color, fontweight="bold", zorder=7)


def _apply_rule(pos_i, theta_i, pos_j, theta_j, R_r, R_a):
    dx = pos_j[:, 0] - pos_i[0]
    dy = pos_j[:, 1] - pos_i[1]
    d = np.hypot(dx, dy)
    rep = (d > 0) & (d < R_r)
    ali = (d >= R_r) & (d < R_a)
    if rep.any():
        return float(np.arctan2(-np.sum(dy[rep] / d[rep]),
                                -np.sum(dx[rep] / d[rep])))
    if ali.any():
        return float(np.arctan2(np.sum(np.sin(theta_j[ali])),
                                np.sum(np.cos(theta_j[ali]))))
    return float(theta_i)


def _evolve(pos_i, th_i, pos_j, th_j):
    new_th_i = _apply_rule(pos_i, th_i, pos_j, th_j, R_R, R_A)
    new_pos_i = pos_i + V_VIS * np.array([np.cos(new_th_i),
                                          np.sin(new_th_i)])
    new_pos_j = pos_j + V_VIS * np.column_stack([np.cos(th_j),
                                                 np.sin(th_j)])
    return new_pos_i, new_th_i, new_pos_j, th_j.copy()


def _render_frame(ax, pos_i, th_i, pos_j, th_j, labels):
    _draw_zones(ax)
    _particle(ax, pos_i, th_i, color=FOCAL_RED, ss=70, al=0.40, lw=2.8,
              label=r"$i$", loff=(0.06, -0.26), fs=9)
    for pos, th, lab in zip(pos_j, th_j, labels):
        _particle(ax, pos, th, color=BLUE, ss=34, al=0.28, lw=1.9,
                  label=lab, loff=(0.05, 0.07), fs=7)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.40, 1.50)


# --------------------------------------------------------------------
# panels
# --------------------------------------------------------------------
def panel_geometry(ax):
    _draw_zones(ax)
    hl = 0.34
    ax.annotate("", xy=(hl, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.6),
                zorder=6)
    ax.scatter([0], [0], s=80, color=BLUE, edgecolor="white", lw=0.8,
               zorder=7)
    ax.text(0.05, -0.18, r"$i$", fontsize=10, fontweight="bold", zorder=7)
    ax.text(hl + 0.02, 0.07, r"$\vec e_i(t)$", fontsize=9, color=BLUE,
            zorder=7)

    def nb(x, y, ang, lab, color=BLUE, alpha=1.0):
        th = np.deg2rad(ang)
        ax.annotate("", xy=(x + 0.24 * np.cos(th), y + 0.24 * np.sin(th)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.7,
                                    alpha=alpha), zorder=6)
        ax.scatter([x], [y], s=34, color=color, alpha=alpha,
                   edgecolor="white", lw=0.6, zorder=7)
        ax.text(x + 0.05, y + 0.08, lab, fontsize=8, alpha=alpha, zorder=7)

    j1 = (-0.18 * 1.2, 0.22 * 1.2)
    nb(*j1, 90, r"$j_1$")
    nrm = np.hypot(*j1)
    away = (-j1[0] / nrm * 0.34, -j1[1] / nrm * 0.34)
    ax.annotate("", xy=away, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#9c3a3a", lw=1.5,
                                ls=(0, (3, 2))), zorder=6)
    ax.text(away[0] + 0.02, away[1] - 0.08, "repulse", fontsize=7.5,
            color="#9c3a3a", style="italic", zorder=7)
    nb(0.46 * 1.2, 0.42 * 1.2, 20, r"$j_2$")
    nb(-0.50 * 1.2, -0.34 * 1.2, 200, r"$j_3$")
    nb(0.85 * 1.2, 0.55 * 1.2, 60, r"$j_\infty$", color="#666", alpha=0.55)
    ax.annotate(r"$R_r$", xy=(R_R * np.cos(np.deg2rad(-50)),
                              R_R * np.sin(np.deg2rad(-50))),
                xytext=(0.55, -1.15), fontsize=9,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#444"))
    ax.annotate(r"$R_a$", xy=(R_A * np.cos(np.deg2rad(-30)),
                              R_A * np.sin(np.deg2rad(-30))),
                xytext=(1.05, -0.70), fontsize=9,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#444"))
    ax.legend(handles=[
        Patch(facecolor=REP_COLOR, alpha=0.55, edgecolor="#9c3a3a",
              label=r"repulsion $d<R_r$"),
        Patch(facecolor=ALI_COLOR, alpha=0.55, edgecolor="#3a4a78",
              label=r"alignment $R_r\!\leq\! d\!<\! R_a$"),
    ], loc="upper right", fontsize=6.5, framealpha=0.9)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.35, 1.35)
    ax.set_title(r"(a) two-zone update", fontsize=8.5, loc="left")


def panel_noise(ax):
    rng = np.random.default_rng(0)
    bins = np.linspace(-6, 6, 200)
    centers = 0.5 * (bins[1:] + bins[:-1])
    cols = [st.WONG["black"], st.WONG["blue"], st.WONG["green"],
            st.WONG["vermil"]]
    for a, ls, c in zip([2.0, 1.5, 1.0, 0.7],
                        ["-", "--", "-.", ":"], cols):
        s = stable_rvs_vector(np.full(200_000, a), 1.0, rng)
        s = s[(s > -6) & (s < 6)]
        h, _ = np.histogram(s, bins=bins, density=True)
        ax.semilogy(centers, h + 1e-6, ls, lw=1.3, color=c,
                    label=fr"$\alpha={a}$")
    ax.set_xlabel(r"angular kick $\xi$", fontsize=8)
    ax.set_ylabel(r"pdf $p_\alpha(\xi)$", fontsize=8)
    ax.set_ylim(1e-4, 1.0)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title(r"(b) $\alpha$-stable noise pdfs", fontsize=8.5, loc="left")


def panel_sigmoid(ax):
    n = np.linspace(0, 8, 200)
    n_star, slope = 3.0, 2.0
    sig = 1.0 / (1.0 + np.exp(-(n - n_star) * slope))
    v_max, v_min, a_min, a_max = 0.05, 0.005, 1.0, 2.0
    v = v_max - (v_max - v_min) * sig
    a = a_min + (a_max - a_min) * sig
    ax2 = ax.twinx()
    ax.plot(n, v, "-", color=V_PURPLE, lw=1.7, label=r"$v_i(n_i)$")
    ax2.plot(n, a, "-", color=A_GREEN, lw=1.7, label=r"$\alpha_i(n_i)$")
    ax.axvline(n_star, ls=":", c="grey", lw=0.8)
    ax.set_xlabel(r"local neighbour count $n_i$", fontsize=8)
    ax.set_ylabel(r"speed $v_i$", color=V_PURPLE, fontsize=8)
    ax2.set_ylabel(r"stability $\alpha_i$", color=A_GREEN, fontsize=8)
    ax.tick_params(axis="y", colors=V_PURPLE, labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax2.tick_params(axis="y", colors=A_GREEN, labelsize=7)
    ax.text(n_star, v_max * 1.03, r"$n_\star$", fontsize=8, ha="center",
            color="grey")
    ax.set_title(r"(c) shared sigmoid (this work)", fontsize=8.5, loc="left")


def _rule(ax_t, ax_tp, title, pos_i, th_i, pos_j, th_j, labels):
    _render_frame(ax_t, pos_i, th_i, pos_j, th_j, labels)
    npi, nti, npj, ntj = _evolve(pos_i, th_i, pos_j, th_j)
    _render_frame(ax_tp, npi, nti, npj, ntj, labels)
    if title.startswith("(d)"):
        ax_tp.plot([pos_i[0], npi[0]], [pos_i[1], npi[1]], ":",
                   color="#444", lw=0.9, zorder=2)
    ax_t.text(0.5, -0.03, r"$t$", transform=ax_t.transAxes, ha="center",
              va="top", fontsize=8, fontweight="bold")
    ax_tp.text(0.5, -0.03, r"$t+\delta t$", transform=ax_tp.transAxes,
               ha="center", va="top", fontsize=8, fontweight="bold")
    ax_t.set_title(title, fontsize=8, loc="left", pad=4)


def main() -> None:
    fig = plt.figure(figsize=(st.DOUBLE_COL[0], 5.0))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0],
                          hspace=0.32, wspace=0.45)
    panel_geometry(fig.add_subplot(gs[0, 0:2]))
    panel_noise(fig.add_subplot(gs[0, 2:4]))
    panel_sigmoid(fig.add_subplot(gs[0, 4:6]))

    _rule(fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
          r"(d) inertia",
          np.array([-0.55, -0.10]), np.deg2rad(20.0),
          np.array([[1.05, 1.05], [-1.20, 0.90]]),
          np.array([np.deg2rad(160.0), np.deg2rad(-20.0)]),
          [r"$j_1$", r"$j_2$"])
    _rule(fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3]),
          r"(e) repulsion",
          np.array([0.0, 0.0]), np.deg2rad(45.0),
          np.array([[-0.22, 0.27], [0.55, -0.25], [1.10, 0.90]]),
          np.array([np.deg2rad(110.0), np.deg2rad(0.0), np.deg2rad(-90.0)]),
          [r"$j_1$", r"$j_2$", r"$j_3$"])
    _rule(fig.add_subplot(gs[1, 4]), fig.add_subplot(gs[1, 5]),
          r"(f) alignment",
          np.array([0.0, 0.0]), np.deg2rad(-30.0),
          np.array([[0.55, 0.40], [-0.58, 0.28], [0.10, -0.62],
                    [1.20, 0.95]]),
          np.array([np.deg2rad(30.0), np.deg2rad(50.0), np.deg2rad(40.0),
                    np.deg2rad(180.0)]),
          [r"$j_1$", r"$j_2$", r"$j_3$", r"$j_4$"])

    fig.savefig(FIG / "fig_double_schematic.pdf", dpi=200,
                bbox_inches="tight")
    print("saved fig_double_schematic.pdf")


if __name__ == "__main__":
    main()
