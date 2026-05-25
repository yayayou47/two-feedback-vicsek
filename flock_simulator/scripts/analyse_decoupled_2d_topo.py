"""
Compare the metric and topological 2D decoupled-sigmoid maps.

Reads both ``data/double_decoupled_2d_nocone.npz`` (metric) and
``data/double_decoupled_2d_topological.npz`` (k=4 nearest-
neighbour alignment). For each map computes:

  * paired Delta_g(r ~= 0.62) per cell, mean +/- SE over the 10
    seeds;
  * per-variable linear fits Delta_g ~ a * n_v + b and
    Delta_g ~ a * n_a + b with their R^2;
  * the multivariate fit Delta_g ~ a_v * n_v + a_a * n_a + c.

Writes ``data/decoupled_2d_topo_summary.npz`` and a two-panel
side-by-side comparison figure
``figures/fig_double_decoupled_2d_metric_vs_topo.pdf``.
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


def per_cell_diff(z):
    rc = z["r_centers"]
    j = int(np.argmin(np.abs(rc - 0.62)))
    gr_full = z["gr_dense"][:, :, :, j]
    gr_mot = z["gr_dense_motility"][:, j]
    diff = gr_full - gr_mot[None, None, :]
    mean = np.nanmean(diff, axis=2)
    se = (np.nanstd(diff, axis=2, ddof=1)
          / np.sqrt(np.sum(np.isfinite(diff), axis=2)))
    return mean, se, z["n_v_grid"], z["n_a_grid"]


def fits(mean_g, se_g, n_v, n_a):
    xv = np.array([nv for nv in n_v for na in n_a])
    xa = np.array([na for nv in n_v for na in n_a])
    y = mean_g.flatten().astype(float)
    sigma = se_g.flatten()
    w = 1.0 / np.maximum(sigma, 1e-6) ** 2

    def r2(yhat):
        ybar = (w * y).sum() / w.sum()
        return 1.0 - (w * (y - yhat) ** 2).sum() / (w * (y - ybar) ** 2).sum()

    cv = np.polyfit(xv.astype(float), y, 1, w=w)
    yh_v = cv[0] * xv + cv[1]
    ca = np.polyfit(xa.astype(float), y, 1, w=w)
    yh_a = ca[0] * xa + ca[1]
    A = np.column_stack([xv.astype(float), xa.astype(float), np.ones_like(xv)])
    cb, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh_b = A @ cb
    return {
        "fit_v": (cv[0], cv[1], r2(yh_v)),
        "fit_a": (ca[0], ca[1], r2(yh_a)),
        "fit_bivar": (cb[0], cb[1], cb[2], r2(yh_b)),
    }


def main() -> None:
    metric = np.load(DATA / "double_decoupled_2d_nocone.npz",
                     allow_pickle=True)
    topo = np.load(DATA / "double_decoupled_2d_topological.npz",
                   allow_pickle=True)

    m_mean, m_se, n_v, n_a = per_cell_diff(metric)
    t_mean, t_se, n_v2, n_a2 = per_cell_diff(topo)
    assert np.array_equal(n_v, n_v2)
    assert np.array_equal(n_a, n_a2)

    m_fits = fits(m_mean, m_se, n_v, n_a)
    t_fits = fits(t_mean, t_se, n_v, n_a)

    print("--- Metric alignment ---")
    print(f"{'n_v\\n_a':>10s} " + " ".join(f"{na:>8.0f}" for na in n_a))
    for iv, nsv in enumerate(n_v):
        print(f"{int(nsv):>10d} " + " ".join(
            f"{m_mean[iv, ia]:+8.3f}" for ia in range(len(n_a))))
    a, b, r2 = m_fits["fit_v"]
    print(f"  fit on n_v: Delta_g = {a:+.4f}*n_v + {b:+.3f}   R^2 = {r2:.3f}")
    a, b, r2 = m_fits["fit_a"]
    print(f"  fit on n_a: Delta_g = {a:+.4f}*n_a + {b:+.3f}   R^2 = {r2:.3f}")
    av, aa, c, r2 = m_fits["fit_bivar"]
    print(f"  bivariate:  Delta_g = {av:+.4f}*n_v + {aa:+.4f}*n_a + {c:+.3f}   R^2 = {r2:.3f}")

    print("\n--- Topological alignment (k_NN = 4) ---")
    print(f"{'n_v\\n_a':>10s} " + " ".join(f"{na:>8.0f}" for na in n_a))
    for iv, nsv in enumerate(n_v):
        print(f"{int(nsv):>10d} " + " ".join(
            f"{t_mean[iv, ia]:+8.3f}" for ia in range(len(n_a))))
    a, b, r2 = t_fits["fit_v"]
    print(f"  fit on n_v: Delta_g = {a:+.4f}*n_v + {b:+.3f}   R^2 = {r2:.3f}")
    a, b, r2 = t_fits["fit_a"]
    print(f"  fit on n_a: Delta_g = {a:+.4f}*n_a + {b:+.3f}   R^2 = {r2:.3f}")
    av, aa, c, r2 = t_fits["fit_bivar"]
    print(f"  bivariate:  Delta_g = {av:+.4f}*n_v + {aa:+.4f}*n_a + {c:+.3f}   R^2 = {r2:.3f}")

    np.savez_compressed(
        DATA / "decoupled_2d_topo_summary.npz",
        n_v=n_v, n_a=n_a,
        metric_mean=m_mean, metric_se=m_se,
        topo_mean=t_mean, topo_se=t_se,
        metric_fit_v=np.array(m_fits["fit_v"]),
        metric_fit_a=np.array(m_fits["fit_a"]),
        topo_fit_v=np.array(t_fits["fit_v"]),
        topo_fit_a=np.array(t_fits["fit_a"]),
    )

    # Side-by-side heatmaps with matching color scale.
    fig, axes = plt.subplots(1, 2, figsize=(st.DOUBLE_COL[0], 3.2))
    vmax = max(float(np.nanmax(np.abs(m_mean))),
               float(np.nanmax(np.abs(t_mean))))
    for ax, data, title in (
        (axes[0], m_mean, "(a) metric alignment"),
        (axes[1], t_mean, "(b) topological alignment ($k = 4$)"),
    ):
        im = ax.imshow(
            data, origin="lower", aspect="auto",
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            extent=[n_a.min() - 0.5, n_a.max() + 0.5,
                    n_v.min() - 0.5, n_v.max() + 0.5],
        )
        ax.set_xticks(n_a)
        ax.set_yticks(n_v)
        ax.set_xlabel(r"$n_\star^\alpha$")
        ax.set_ylabel(r"$n_\star^v$")
        ax.set_title(title, fontsize=8, loc="left")
        for iv in range(len(n_v)):
            for ia in range(len(n_a)):
                col = "white" if abs(data[iv, ia]) > 0.6 * vmax else "black"
                ax.text(n_a[ia], n_v[iv], f"{data[iv, ia]:+.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color=col)
        diag = np.linspace(n_a.min(), n_a.max(), 100)
        ax.plot(diag, diag, ":", color="grey", lw=0.6, alpha=0.8)
    cb = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cb.set_label(r"$\Delta g$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    out_pdf = FIG / "fig_double_decoupled_2d_metric_vs_topo.pdf"
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    print(f"\nsaved figure: {out_pdf.name}")


if __name__ == "__main__":
    main()
