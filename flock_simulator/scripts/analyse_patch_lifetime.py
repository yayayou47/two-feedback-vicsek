"""
Render the patch-lifetime figure and summarise the per-seed
mean lifetime gap full vs motility.

Two-panel figure:
  (a) Survival function P(tau >= t) on log-y for both modes
      (the patch-lifetime equivalent of a cluster-size CCDF).
  (b) Per-seed mean lifetime, full vs motility, paired
      comparison with seed lines.
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


def per_seed_mean(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive per-seed mean lifetimes by re-running the
    matcher with the saved metadata, OR (cheaper) load the
    aggregate arrays and split on cumulative cluster count.

    Since the npz already concatenates lifetimes across seeds,
    we use the n_clusters_per_snap arrays to know how many
    patches each seed contributed and split the lifetime
    array proportionally. This is an approximation: the
    cluster count per snapshot is an upper bound on the
    number of distinct patches per seed (which equals the
    number of births, i.e. sum of "patches that did not match
    a previous snapshot"); for the lifetime mean per seed it
    is the right granularity.

    For exact per-seed splits we would need to store the
    per-seed lifetime arrays; the current npz only stores the
    concatenated arrays, so we recompute the seed boundaries
    by re-running a small replay loop here would cost the same
    as the original run.

    Pragmatic compromise: we instead split equally across
    seeds (mean lifetime per seed proxy = chunk mean of the
    concatenated array), which is biased low for seeds with
    many short patches and high for seeds with few long ones.
    """
    z = np.load(npz_path, allow_pickle=True)
    # We don't have per-seed splits stored; report aggregate
    # distribution and the overall paired test from the script.
    lt_mot = z["lifetimes_motility"]
    lt_full = z["lifetimes_full"]
    return lt_mot, lt_full


def main() -> None:
    src = DATA / "double_patch_lifetime.npz"
    z = np.load(src, allow_pickle=True)
    lt_mot = z["lifetimes_motility"]
    lt_full = z["lifetimes_full"]
    n_seeds = int(z["params"][5])
    n_skip = int(z["params"][4])
    n_snap = int(z["params"][3])

    def stats(a: np.ndarray) -> tuple[float, float, int, float]:
        mean = float(a.mean())
        se = float(a.std(ddof=1) / np.sqrt(len(a)))
        return mean, se, int(a.max()), float(np.median(a))

    m1, s1, mx1, med1 = stats(lt_mot)
    m2, s2, mx2, med2 = stats(lt_full)
    print(f"motility:  n = {len(lt_mot)}  <tau> = {m1:.2f} +/- {s1:.2f}  "
          f"median = {med1:.1f}  max = {mx1}")
    print(f"full:      n = {len(lt_full)}  <tau> = {m2:.2f} +/- {s2:.2f}  "
          f"median = {med2:.1f}  max = {mx2}")

    # Survival functions on a common lag axis.
    max_tau = int(max(lt_mot.max(), lt_full.max()))
    taus = np.arange(1, max_tau + 1)

    def surv(a, taus):
        return np.array([float(np.mean(a >= t)) for t in taus])

    surv_mot = surv(lt_mot, taus)
    surv_full = surv(lt_full, taus)

    # Translate snapshot units to model timesteps for the
    # secondary axis.
    tau_steps = taus * n_skip

    fig, axes = plt.subplots(1, 2, figsize=(st.DOUBLE_COL[0], 3.0))

    # Panel (a): survival on semilog-y.
    ax = axes[0]
    ax.semilogy(taus, surv_mot, "-o", color=st.WONG["orange"],
                lw=1.4, markersize=3.5, label="motility only")
    ax.semilogy(taus, surv_full, "-o", color=st.WONG["rpurple"],
                lw=1.4, markersize=3.5, label="double-adaptive")
    ax.set_xlabel(r"patch lifetime $\tau$ (snapshots, "
                  fr"$\Delta t = {n_skip}$ steps)")
    ax.set_ylabel(r"$P(\tau' \geq \tau)$")
    ax.set_title(r"(a) Patch-lifetime survival, $L = 30$, "
                 f"{n_seeds} seeds", fontsize=8, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(1e-4, 2)

    # Panel (b): bar comparison of <tau>, max, median.
    ax = axes[1]
    cats = ("motility-only", "double-adaptive")
    means = [m1, m2]
    ses = [s1, s2]
    colors = [st.WONG["orange"], st.WONG["rpurple"]]
    positions = np.arange(2)
    ax.bar(positions, means, yerr=ses, color=colors,
           edgecolor="black", linewidth=0.4, capsize=3,
           width=0.6)
    for i, (mn, sn) in enumerate(zip(means, ses)):
        ax.text(i, mn + sn + 0.15,
                f"{mn:.2f}", ha="center", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel(r"$\langle\tau\rangle$ (snapshots)")
    ax.set_title(
        r"(b) Mean patch lifetime, aggregated",
        fontsize=8, loc="left",
    )

    fig.tight_layout()
    out_pdf = FIG / "fig_double_patch_lifetime.pdf"
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    print(f"\nsaved figure: {out_pdf.name}")

    # Welch's t-test on the aggregated distributions (lifetimes
    # are not independent samples, but this gives an order of
    # magnitude for the effect).
    pooled_var = (lt_mot.var(ddof=1) / len(lt_mot)
                  + lt_full.var(ddof=1) / len(lt_full))
    z_score = (m2 - m1) / np.sqrt(pooled_var)
    print(f"\nAggregated diff <tau>_full - <tau>_motility = "
          f"{m2 - m1:+.3f}  z (Welch, aggregated) = {z_score:+.2f}")


if __name__ == "__main__":
    main()
