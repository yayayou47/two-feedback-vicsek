"""Figure 1 for version 4: Vicsek--Couzin setup and update rules.

Ported verbatim from the version-1 schematic (same per-panel figsize
5.4 x 4.4 and the same element sizes), adapted to v4 (R_r = 0.5,
R_a = 0.7; v4 alpha-stable sampler) and extended with the shared
sigmoid that defines the two-feedback model. Each component is a
SEPARATE PDF so the manuscript can assemble them at the exact
version-1 \\includegraphics widths:

  fig_setup_geometry.pdf   (a) two-zone geometry
  fig_setup_noise.pdf      (b) symmetric alpha-stable noise pdfs
  fig_setup_sigmoid.pdf    (c) the shared sigmoid (this work)
  fig_setup_inertia.pdf    (d) inertia rule, t and t+dt
  fig_setup_repulsion.pdf  (e) repulsion rule, t and t+dt
  fig_setup_alignment.pdf  (f) alignment rule, t and t+dt
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

PARTICLE_BLUE = "#1f4ea1"          # v1 navy
FOCAL_RED = "#c83a3a"
REP_COLOR = "#e07b7b"
ALI_COLOR = "#9bb8de"
V_PURPLE = "#7a3aa0"
A_GREEN = "#3aa040"

# Cartoon zone radii: v4 values (R_r=0.5, R_a=0.7) inflated x1.2 for
# legibility, exactly as version 1 inflated its 0.45/0.7 radii.
R_R = 0.5 * 1.2
R_A = 0.7 * 1.2
V_VIS = 0.40                        # visualisation step (v1 value)

FIGSIZE = (5.4, 4.4)               # exact version-1 per-panel figsize


def _save(fig, name):
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


# ====================================================================
# (a) two-zone geometry  (= version1 fig_model_schematic)
# ====================================================================
def fig_geometry():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rep = Wedge((0, 0), R_R, 0, 360, facecolor=REP_COLOR, alpha=0.55,
                edgecolor="#9c3a3a", lw=0.9, zorder=1)
    ali = Wedge((0, 0), R_A, 0, 360, width=R_A - R_R,
                facecolor=ALI_COLOR, alpha=0.55,
                edgecolor="#3a4a78", lw=0.9, zorder=1)
    ax.add_patch(rep); ax.add_patch(ali)

    head_len = 0.20 * 1.2 * 1.2
    ax.annotate("", xy=(head_len, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=PARTICLE_BLUE,
                                lw=3.1), zorder=6)
    ax.scatter([0], [0], s=108, color=PARTICLE_BLUE, edgecolor="white",
               lw=0.96, zorder=7)
    ax.text(0.05, -0.13, r"$i$", fontsize=16, fontweight="bold", zorder=7)
    ax.text(head_len + 0.02, 0.05, r"$\vec e_i(t)$", fontsize=14,
            color=PARTICLE_BLUE, zorder=7)

    arrow_len = 0.14 * 1.2 * 1.2

    def neighbour(x, y, theta_deg, label, color=PARTICLE_BLUE, alpha=1.0):
        th = np.deg2rad(theta_deg)
        ax.annotate("", xy=(x + arrow_len * np.cos(th),
                            y + arrow_len * np.sin(th)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.92,
                                    alpha=alpha), zorder=6)
        ax.scatter([x], [y], s=50, color=color, alpha=alpha,
                   edgecolor="white", lw=0.72, zorder=7)
        ax.text(x + 0.05, y + 0.08, label, fontsize=13, alpha=alpha,
                zorder=7)

    j1 = (-0.18 * 1.2, 0.22 * 1.2)
    neighbour(*j1, theta_deg=90, label=r"$j_1$")
    nrm = np.hypot(*j1)
    away = (-j1[0] / nrm * 0.30 * 1.2 * 1.2, -j1[1] / nrm * 0.30 * 1.2 * 1.2)
    ax.annotate("", xy=away, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#9c3a3a", lw=1.8,
                                ls=(0, (3, 2))), zorder=6)
    ax.text(away[0] + 0.02, away[1] - 0.06, "repulse", fontsize=12,
            color="#9c3a3a", style="italic", zorder=7)

    neighbour(0.46 * 1.2, 0.42 * 1.2, theta_deg=20, label=r"$j_2$")
    neighbour(-0.50 * 1.2, -0.34 * 1.2, theta_deg=200, label=r"$j_3$")
    neighbour(-0.55 * 1.2, 0.06 * 1.2, theta_deg=0, label=r"$j_4$")
    neighbour(0.85 * 1.2, 0.55 * 1.2, theta_deg=60, label=r"$j_\infty$",
              color="#666", alpha=0.55)

    ax.annotate(r"$R_r$",
                xy=(R_R * np.cos(np.deg2rad(-50)),
                    R_R * np.sin(np.deg2rad(-50))),
                xytext=(0.66, -1.14), fontsize=14,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444"))
    ax.annotate(r"$R_a$",
                xy=(R_A * np.cos(np.deg2rad(-30)),
                    R_A * np.sin(np.deg2rad(-30))),
                xytext=(1.26, -0.74), fontsize=14,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444"))

    ax.legend(handles=[
        Patch(facecolor=REP_COLOR, alpha=0.55, edgecolor="#9c3a3a",
              label=r"Repulsion ($d<R_r$)"),
        Patch(facecolor=ALI_COLOR, alpha=0.55, edgecolor="#3a4a78",
              label=r"Alignment ($R_r \leq d < R_a$)"),
    ], loc="upper right", fontsize=12, framealpha=0.92)

    ax.set_xlim(-1.86, 1.86); ax.set_ylim(-1.44, 1.44)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r"(a) Two-zone Vicsek--Couzin update "
                 r"(full $360^\circ$ vision)", fontsize=14)
    fig.tight_layout()
    _save(fig, "fig_setup_geometry.pdf")


# ====================================================================
# (b) symmetric alpha-stable noise pdfs  (= version1 fig_noise_pdf)
# ====================================================================
def fig_noise():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(-6, 6, 200)
    centers = 0.5 * (bins[1:] + bins[:-1])
    for a, ls in zip([2.0, 1.5, 1.0, 0.7], ["-", "--", "-.", ":"]):
        s = stable_rvs_vector(np.full(200_000, a), 1.0, rng)
        s = s[(s > -6) & (s < 6)]
        h, _ = np.histogram(s, bins=bins, density=True)
        ax.semilogy(centers, h + 1e-6, ls, lw=1.3, label=fr"$\alpha={a}$")
    ax.set_xlabel(r"angular kick $\xi$", fontsize=13)
    ax.set_ylabel(r"pdf $p_\alpha(\xi)$", fontsize=13)
    ax.set_ylim(1e-4, 1.0)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=12)
    fig.suptitle(r"(b) Symmetric $\alpha$-stable noise pdfs",
                 fontsize=13.5, y=0.97)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save(fig, "fig_setup_noise.pdf")


# ====================================================================
# (c) the shared sigmoid (this work)
# ====================================================================
def fig_sigmoid():
    n = np.linspace(0, 8, 200)
    n_star, slope = 3.0, 2.0
    sig = 1.0 / (1.0 + np.exp(-(n - n_star) * slope))
    v_max, v_min, a_min, a_max = 0.05, 0.005, 1.0, 2.0
    v = v_max - (v_max - v_min) * sig
    a = a_min + (a_max - a_min) * sig

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax2 = ax.twinx()
    ax.plot(n, v, "-", color=V_PURPLE, lw=2.2, label=r"$v_i(n_i)$")
    ax2.plot(n, a, "-", color=A_GREEN, lw=2.2, label=r"$\alpha_i(n_i)$")
    ax.axvline(n_star, ls=":", c="grey", lw=1.0)
    ax.set_xlabel(r"local neighbour count $n_i$", fontsize=13)
    ax.set_ylabel(r"speed $v_i$", color=V_PURPLE, fontsize=13)
    ax2.set_ylabel(r"stability index $\alpha_i$", color=A_GREEN, fontsize=13)
    ax.tick_params(axis="y", colors=V_PURPLE, labelsize=11)
    ax.tick_params(axis="x", labelsize=11)
    ax2.tick_params(axis="y", colors=A_GREEN, labelsize=11)
    ax.text(n_star, v_max * 1.02, r"$n_\star$", fontsize=13, ha="center",
            color="grey")
    l1, = ax.plot([], [], "-", color=V_PURPLE, lw=2.2, label=r"$v_i$")
    l2, = ax.plot([], [], "-", color=A_GREEN, lw=2.2, label=r"$\alpha_i$")
    ax.legend(handles=[l1, l2], loc="center right", fontsize=12,
              framealpha=0.92)
    fig.suptitle(r"(c) Shared sigmoid (this work)", fontsize=13.5, y=0.97)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save(fig, "fig_setup_sigmoid.pdf")


# ====================================================================
# (d)-(f) worked rule examples  (= version1 fig_rule_*)
# ====================================================================
def _draw_zones(ax):
    ax.add_patch(Wedge((0, 0), R_R, 0, 360, facecolor=REP_COLOR, alpha=0.55,
                       edgecolor="#9c3a3a", lw=0.9, zorder=1))
    ax.add_patch(Wedge((0, 0), R_A, 0, 360, width=R_A - R_R,
                       facecolor=ALI_COLOR, alpha=0.55,
                       edgecolor="#3a4a78", lw=0.9, zorder=1))


def _particle(ax, pos, heading, *, color, ss, al, lw, label=None,
              loff=(0.07, 0.08), fs=13, alpha=1.0):
    x, y = float(pos[0]), float(pos[1])
    ax.annotate("", xy=(x + al * np.cos(heading), y + al * np.sin(heading)),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                alpha=alpha), zorder=6)
    ax.scatter([x], [y], s=ss, color=color, alpha=alpha,
               edgecolor="white", lw=0.9, zorder=7)
    if label:
        ax.text(x + loff[0], y + loff[1], label, fontsize=fs, color=color,
                fontweight="bold", zorder=7)


def _apply_rule(pos_i, theta_i, pos_j, theta_j):
    dx = pos_j[:, 0] - pos_i[0]; dy = pos_j[:, 1] - pos_i[1]
    d = np.hypot(dx, dy)
    rep = (d > 0) & (d < R_R); ali = (d >= R_R) & (d < R_A)
    if rep.any():
        return float(np.arctan2(-np.sum(dy[rep] / d[rep]),
                                -np.sum(dx[rep] / d[rep])))
    if ali.any():
        return float(np.arctan2(np.sum(np.sin(theta_j[ali])),
                                np.sum(np.cos(theta_j[ali]))))
    return float(theta_i)


def _evolve(pos_i, th_i, pos_j, th_j):
    nti = _apply_rule(pos_i, th_i, pos_j, th_j)
    npi = pos_i + V_VIS * np.array([np.cos(nti), np.sin(nti)])
    npj = pos_j + V_VIS * np.column_stack([np.cos(th_j), np.sin(th_j)])
    return npi, nti, npj, th_j.copy()


def _render_frame(ax, pos_i, th_i, pos_j, th_j, labels):
    _draw_zones(ax)
    _particle(ax, pos_i, th_i, color=FOCAL_RED, ss=108, al=0.45, lw=4.0,
              label=r"$i$", loff=(0.07, -0.30), fs=15)
    for pos, th, lab in zip(pos_j, th_j, labels):
        _particle(ax, pos, th, color=PARTICLE_BLUE, ss=50, al=0.32, lw=2.6,
                  label=lab, loff=(0.06, 0.09), fs=12)


def _rule_pair(title, pos_i, th_i, pos_j, th_j, labels, name, drift=False):
    fig, (ax_t, ax_tp) = plt.subplots(1, 2, figsize=FIGSIZE,
                                      sharex=True, sharey=True)
    for ax, lab in zip((ax_t, ax_tp), (r"$t$", r"$t + \delta t$")):
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.45, 1.55)
        ax.text(0.5, 1.02, lab, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=14, fontweight="bold")
    fig.suptitle(title, fontsize=13.5, y=0.97)
    _render_frame(ax_t, pos_i, th_i, pos_j, th_j, labels)
    npi, nti, npj, ntj = _evolve(pos_i, th_i, pos_j, th_j)
    _render_frame(ax_tp, npi, nti, npj, ntj, labels)
    if drift:
        ax_tp.plot([pos_i[0], npi[0]], [pos_i[1], npi[1]], ":",
                   color="#444", lw=0.9, zorder=2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save(fig, name)


def fig_rules():
    _rule_pair(r"(d) Inertia: no neighbour in $R_a$"
               r" $\Rightarrow \vec e_i$ unchanged",
               np.array([-0.55, -0.10]), np.deg2rad(20.0),
               np.array([[1.05, 1.05], [-1.20, 0.90]]),
               np.array([np.deg2rad(160.0), np.deg2rad(-20.0)]),
               [r"$j_1$", r"$j_2$"], "fig_setup_inertia.pdf", drift=True)
    _rule_pair(r"(e) Repulsion: $d_{ij_1}<R_r$"
               r" $\Rightarrow \vec e_i^{\,\star} = -\widehat{x_{j_1}-x_i}$",
               np.array([0.0, 0.0]), np.deg2rad(45.0),
               np.array([[-0.22, 0.27], [0.55, -0.25], [1.10, 0.90]]),
               np.array([np.deg2rad(110.0), np.deg2rad(0.0),
                         np.deg2rad(-90.0)]),
               [r"$j_1$", r"$j_2$", r"$j_3$"], "fig_setup_repulsion.pdf")
    _rule_pair(r"(f) Alignment: $R_r\!\leq\! d_{ij}\!<\!R_a$"
               r" $\Rightarrow \vec e_i^{\,\star} ="
               r" \mathrm{atan2}(\!\sum\!\sin\theta_j,\!\sum\!\cos\theta_j)$",
               np.array([0.0, 0.0]), np.deg2rad(-30.0),
               np.array([[0.55, 0.40], [-0.58, 0.28], [0.10, -0.62],
                         [1.20, 0.95]]),
               np.array([np.deg2rad(30.0), np.deg2rad(50.0),
                         np.deg2rad(40.0), np.deg2rad(180.0)]),
               [r"$j_1$", r"$j_2$", r"$j_3$", r"$j_4$"],
               "fig_setup_alignment.pdf")


def main() -> None:
    fig_geometry()
    fig_noise()
    fig_sigmoid()
    fig_rules()
    print("saved fig_setup_{geometry,noise,sigmoid,inertia,"
          "repulsion,alignment}.pdf")


if __name__ == "__main__":
    main()
