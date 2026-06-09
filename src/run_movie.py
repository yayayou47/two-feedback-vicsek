r"""
Movie generator for the double-adaptive Vicsek-Couzin model.
Drives the DoubleAdaptiveVicsek simulator and writes an MP4 whose
particles are coloured by instantaneous local speed $v_i$ (yellow
= slow/crowded, purple = fast/isolated), with white heading
arrows and two insets tracking the polar order
$\langle\varphi\rangle(t)$ and the density-separation index
$s_{\rm sep}(t)$. The mode, eta, L, N, seed, and step count are
CLI flags; --all renders the five modes and --regimes renders
three contrasted eta values. Output: ../videos/<mode>_eta<eta>.mp4.

The same script covers the five study modes through CLI flags:

  vicsek          original Vicsek (Gaussian noise, fixed v)
  cauchy_ref      heavy-tailed reference (alpha = 1, fixed v)
  motility        motility adaptive (alpha = 1, adaptive v)
  noise_shape     noise-shape adaptive (alpha in [1,2], fixed v)
  double          full two-feedback model (this work)

Examples:

  python3 run_movie.py --mode double --eta 0.10 --steps 600 \
      --L 30 --N 2000 --seed 11

  python3 run_movie.py --mode vicsek --eta 0.20 --steps 600 \
      --L 30 --N 2000

By default, output is written to ../videos/<mode>_eta<eta>.mp4.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import style
from vicsek_double_adaptive import (DoubleAdaptiveParams,
                                     DoubleAdaptiveVicsek)


HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent / "videos"
VIDEOS.mkdir(exist_ok=True)

style.apply()


# Mode -> (v_min, v_max, alpha_min, alpha_max, pretty title)
MODES = {
    "vicsek":      (0.05, 0.05, 2.0, 2.0,
                     "Vicsek (Gaussian, fixed v)"),
    "cauchy_ref":  (0.05, 0.05, 1.0, 1.0,
                     "Cauchy reference"),
    "motility":    (0.005, 0.05, 1.0, 1.0,
                     "motility adaptive"),
    "noise_shape": (0.05, 0.05, 1.0, 2.0,
                     "noise-shape adaptive"),
    "double":      (0.005, 0.05, 1.0, 2.0,
                     "double-adaptive"),
}


def render_frame(sim, phi_hist, sep_hist, arrow_len, mode_label,
                  v_min, v_max):
    """Render a single frame: scatter coloured by v_i + heading
    arrows + two insets tracking phi(t) and s_sep(t)."""
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=110)
    p = sim.p
    u = np.cos(sim.theta)
    v = np.sin(sim.theta)

    # Scatter coloured by v_i. Use the global v_min/v_max of the
    # mode for the colorbar limits (so fixed-v modes are uniform).
    if v_max > v_min:
        vmin_cm, vmax_cm = v_min, v_max
    else:
        # Fixed-speed mode -- give a tiny range so colours are
        # uniform but the colorbar still renders.
        vmin_cm, vmax_cm = v_min - 1e-6, v_max + 1e-6
    sc = ax.scatter(sim.x, sim.y, c=sim.v_i, cmap="viridis_r",
                     s=8, vmin=vmin_cm, vmax=vmax_cm,
                     alpha=0.85, edgecolor="none")
    ax.quiver(sim.x, sim.y, u, v,
              color="white",
              scale=1.0 / arrow_len, scale_units="xy",
              angles="xy", width=0.0028,
              headwidth=3.0, headlength=3.5,
              alpha=0.65)
    ax.set_xlim(0, p.L)
    ax.set_ylim(0, p.L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    title = (
        f"{mode_label}   "
        fr"$N={p.N}$  $L={p.L:.0f}$  $\eta={p.eta:.3f}$  $t={sim.t}$"
    )
    ax.set_title(title, fontsize=10)

    # Inset 1: polar order phi(t).
    ax2 = fig.add_axes([0.13, 0.13, 0.30, 0.16],
                        facecolor=style.CREAM)
    ax2.plot(phi_hist, color="black", lw=1.0)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, max(50, len(phi_hist)))
    ax2.set_title(r"$\langle\varphi\rangle(t)$", fontsize=8)
    ax2.tick_params(labelsize=6)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.5)

    # Inset 2: density-separation index s_sep(t).
    ax3 = fig.add_axes([0.60, 0.13, 0.30, 0.16],
                        facecolor=style.CREAM)
    ax3.plot(sep_hist, color="#7a3aa0", lw=1.0)
    ax3.set_ylim(1.0, max(2.0, 1.1 * (max(sep_hist)
                                        if sep_hist else 1.5)))
    ax3.set_xlim(0, max(50, len(sep_hist)))
    ax3.set_title(r"$s_{\rm sep}(t)$", fontsize=8)
    ax3.tick_params(labelsize=6)
    for spine in ax3.spines.values():
        spine.set_linewidth(0.5)

    # Compact colorbar for v_i (only meaningful in adaptive modes).
    if v_max > v_min:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label(r"$v_i$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def build_sim(mode: str, args) -> DoubleAdaptiveVicsek:
    if mode not in MODES:
        raise ValueError(
            f"unknown mode {mode!r}; choose from {sorted(MODES)}"
        )
    v_min, v_max, alpha_min, alpha_max, _ = MODES[mode]
    p = DoubleAdaptiveParams(
        N=args.N, L=args.L,
        v_max=v_max, v_min=v_min,
        alpha_min=alpha_min, alpha_max=alpha_max,
        R_r=args.R_r, R_a=args.R_a, beta=args.beta,
        eta=args.eta,
        n_star=args.n_star, slope=args.slope,
        seed=args.seed,
    )
    return DoubleAdaptiveVicsek(p)


def render_movie(mode: str, args, out_path: Path):
    sim = build_sim(mode, args)
    sim.theta[:] = 0.0
    v_min, v_max, _, _, mode_label = MODES[mode]

    # Optional warm-up before recording.
    for _ in tqdm(range(args.warmup), desc=f"{mode} warmup",
                  leave=False):
        sim.step()

    phi_hist: list[float] = []
    sep_hist: list[float] = []
    writer = imageio.get_writer(str(out_path), fps=args.fps,
                                 codec="libx264", quality=7)
    try:
        for k in tqdm(range(args.steps), desc=f"{mode} record"):
            sim.step()
            phi_hist.append(sim.polarisation())
            sep_hist.append(sim.density_separation_index())
            if k % args.every == 0:
                frame = render_frame(sim, phi_hist, sep_hist,
                                      args.arrow_len, mode_label,
                                      v_min, v_max)
                writer.append_data(frame)
    finally:
        writer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=sorted(MODES), default="double",
                     help="model variant to simulate")
    ap.add_argument("--all", action="store_true",
                     help="render the five modes back to back at the "
                          "same eta")
    ap.add_argument("--regimes", action="store_true",
                     help="render the chosen mode at three contrasted "
                          "eta values (ordered/critical/disordered)")
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--L", type=float, default=30.0)
    ap.add_argument("--R_r", type=float, default=0.5)
    ap.add_argument("--R_a", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=30.0)
    ap.add_argument("--eta", type=float, default=0.10)
    ap.add_argument("--n_star", type=float, default=3.0)
    ap.add_argument("--slope", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--arrow-len", type=float, default=0.55)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    modes_to_run = sorted(MODES) if args.all else [args.mode]
    etas_to_run = ([0.020, 0.100, 0.300]
                   if args.regimes else [args.eta])

    for mode in modes_to_run:
        for eta in etas_to_run:
            args.eta = float(eta)
            if args.out and not args.all and not args.regimes:
                out = Path(args.out)
            else:
                out = VIDEOS / f"{mode}_eta{eta:.3f}.mp4"
            render_movie(mode, args, out)
            print(f"saved: {out}")


if __name__ == "__main__":
    main()
