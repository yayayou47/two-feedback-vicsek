"""
Test the prediction that the critical motility threshold
n_star_v at which Delta_g changes sign scales linearly with
the global density sigma.

Combines the sigma-sweep output (sigma in {1.0, 1.5, 3.0})
with the n_star_alpha = 3 column of the 2D decoupled map
(sigma = 2.22) and extracts, per sigma, the linear fit
Delta_g(n_star_v) and the zero-crossing n_star_v_crit.

Output:
  * data/sigma_sweep_summary.npz: per-sigma slope, intercept,
    R^2, n_star_v_crit, dense-phase density ratio
    n_star_v_crit / (A * sigma).
  * figures/fig_double_sigma_sweep.pdf: two-panel figure
    (Delta_g vs n_v at each sigma; n_v_crit vs sigma with
    linear fit).
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

A = np.pi * (0.7 ** 2 - 0.5 ** 2)  # alignment-annulus area


def main() -> None:
    swp = np.load(DATA / "double_sigma_sweep_nocone.npz",
                  allow_pickle=True)
    twod = np.load(DATA / "double_decoupled_2d_nocone.npz",
                   allow_pickle=True)

    sigmas_swp = swp["sigmas"]                 # (3,)
    n_v_grid = swp["n_v_grid"]                 # (7,)
    r_centers = swp["r_centers"]
    j = int(np.argmin(np.abs(r_centers - 0.62)))

    # sigma-sweep paired diff at r ~ 0.62.
    gr_full_swp = swp["gr_dense_full"]         # (n_s, n_v, n_seed, n_bins)
    gr_mot_swp = swp["gr_dense_motility"]      # (n_s, n_seed, n_bins)
    diff_swp = gr_full_swp[..., j] - gr_mot_swp[:, None, :, j]
    mean_swp = np.nanmean(diff_swp, axis=2)    # (n_s, n_v)
    se_swp = (np.nanstd(diff_swp, axis=2, ddof=1)
              / np.sqrt(np.sum(np.isfinite(diff_swp), axis=2)))

    # 2D map at sigma = 2.22, n_star_alpha = 3 (third column,
    # index 2). Map n_star_v matches 2D grid {1,2,3,5,8}, so we
    # have 5 of the 7 points; we keep what we have.
    twod_nv = twod["n_v_grid"]                 # {1,2,3,5,8}
    twod_na = twod["n_a_grid"]
    ia_three = int(np.where(twod_na == 3.0)[0][0])
    gr_full_2d = twod["gr_dense"][:, ia_three, :, j]    # (5, n_seed)
    gr_mot_2d = twod["gr_dense_motility"][:, j]         # (n_seed,)
    diff_2d = gr_full_2d - gr_mot_2d[None, :]
    mean_2d = np.nanmean(diff_2d, axis=1)               # (5,)
    se_2d = (np.nanstd(diff_2d, axis=1, ddof=1)
             / np.sqrt(np.sum(np.isfinite(diff_2d), axis=1)))

    # Assemble (sigma, n_v) -> (mean, se).
    sigma_2_22 = 2.22
    sigmas_all = np.concatenate([sigmas_swp, [sigma_2_22]])
    sort_idx = np.argsort(sigmas_all)
    sigmas_all = sigmas_all[sort_idx]
    # mean/se per sigma; for sigma=2.22 we have 5 n_v values, others have 7.
    series = []   # list of (sigma, np.array of n_v, np.array of mean, np.array of se)
    for isg, sigma in enumerate(sigmas_swp):
        series.append((sigma, n_v_grid, mean_swp[isg], se_swp[isg]))
    series.append((sigma_2_22, twod_nv, mean_2d, se_2d))
    series.sort(key=lambda t: t[0])

    # Per-sigma linear fit and zero crossing.
    n_v_crit = []
    n_v_crit_se = []
    slopes_a = []
    intercepts_b = []
    r2_per = []
    for sigma, nvs, ms, ses in series:
        w = 1.0 / np.maximum(ses, 1e-6) ** 2
        a, b = np.polyfit(nvs.astype(float), ms, 1, w=w)
        yh = a * nvs + b
        ybar = (w * ms).sum() / w.sum()
        r2 = 1.0 - (w * (ms - yh) ** 2).sum() / (w * (ms - ybar) ** 2).sum()
        crit = -b / a
        # Crude propagation of SE through the zero-crossing.
        var_b = np.cov(np.vstack([nvs, np.ones_like(nvs)]), aweights=w)
        # Use a simple bootstrap on the per-cell points for a CI on crit.
        rng = np.random.default_rng(11)
        B = 5000
        bs_crit = []
        for _ in range(B):
            # resample by perturbing each cell within its SE
            ms_b = ms + ses * rng.standard_normal(len(ms))
            ab, bb = np.polyfit(nvs.astype(float), ms_b, 1, w=w)
            bs_crit.append(-bb / ab)
        bs_crit = np.array(bs_crit)
        crit_se = float(np.std(bs_crit, ddof=1))
        n_v_crit.append(crit)
        n_v_crit_se.append(crit_se)
        slopes_a.append(a)
        intercepts_b.append(b)
        r2_per.append(r2)
        print(f"sigma = {sigma:>5.2f}: Delta_g = {a:+.4f}*n_v + {b:+.3f}  "
              f"(R^2 = {r2:.3f})  n_v_crit = {crit:+.2f} +/- {crit_se:.2f}  "
              f"n_v_crit / (A*sigma) = {crit / (A * sigma):.2f}")

    # Linear fit n_v_crit(sigma) = c * sigma + d
    sigmas_arr = np.array([s for s, *_ in series])
    crit_arr = np.array(n_v_crit)
    crit_se_arr = np.array(n_v_crit_se)
    w_sigma = 1.0 / np.maximum(crit_se_arr, 1e-6) ** 2
    c, d = np.polyfit(sigmas_arr, crit_arr, 1, w=w_sigma)
    yh = c * sigmas_arr + d
    ybar_w = (w_sigma * crit_arr).sum() / w_sigma.sum()
    R2_scaling = 1.0 - (w_sigma * (crit_arr - yh) ** 2).sum() \
        / (w_sigma * (crit_arr - ybar_w) ** 2).sum()
    print()
    print(f"Scaling fit n_v_crit(sigma) = {c:+.3f} * sigma + {d:+.3f}  "
          f"(R^2 = {R2_scaling:.3f})")
    print(f"A = pi(R_a^2 - R_r^2) = {A:.3f}")
    print(f"slope c / A = {c / A:.2f}   "
          f"(interpretable as the dense-to-mean density ratio)")

    # Save summary.
    out = DATA / "sigma_sweep_summary.npz"
    np.savez_compressed(
        out,
        sigmas=sigmas_arr,
        slope_a=np.array(slopes_a),
        intercept_b=np.array(intercepts_b),
        r2_per=np.array(r2_per),
        n_v_crit=crit_arr, n_v_crit_se=crit_se_arr,
        scaling_c=float(c), scaling_d=float(d),
        scaling_R2=float(R2_scaling),
        A=float(A),
    )
    print(f"saved {out.name}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(st.DOUBLE_COL[0], 3.0))
    palette = {
        1.0: st.WONG["blue"],
        1.5: st.WONG["sky"],
        2.22: st.WONG["green"],
        3.0: st.WONG["vermil"],
    }

    # Panel (a): Delta_g vs n_v at each sigma.
    ax = axes[0]
    for sigma, nvs, ms, ses in series:
        c_col = palette.get(round(sigma, 2), st.WONG["black"])
        ax.errorbar(
            nvs, ms, yerr=ses,
            fmt="o", markersize=4.5,
            color=c_col, ecolor="grey", elinewidth=0.6, capsize=2,
            markeredgecolor="black", markeredgewidth=0.3,
            label=fr"$\sigma = {sigma:g}$",
        )
        a, b = np.polyfit(nvs.astype(float), ms, 1,
                          w=1.0 / np.maximum(ses, 1e-6) ** 2)
        xf = np.linspace(0.5, 8.5, 100)
        ax.plot(xf, a * xf + b, "-", color=c_col, lw=0.8, alpha=0.7)
    ax.axhline(0, color="black", lw=0.4, ls="--", alpha=0.4)
    ax.set_xlabel(r"$n_\star^v$")
    ax.set_ylabel(r"$\Delta g(r\!\simeq\!0.6)$")
    ax.set_title(r"(a) $\Delta g(n_\star^v)$ at four densities",
                 fontsize=8, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper right")

    # Panel (b): n_v_crit vs sigma with linear fit.
    ax = axes[1]
    for sigma, crit, crit_se in zip(sigmas_arr, crit_arr, crit_se_arr):
        c_col = palette.get(round(sigma, 2), st.WONG["black"])
        ax.errorbar(sigma, crit, yerr=crit_se,
                    fmt="o", markersize=6,
                    color=c_col, ecolor="grey", elinewidth=0.7, capsize=3,
                    markeredgecolor="black", markeredgewidth=0.4)
    xf = np.linspace(0.7, 3.3, 100)
    ax.plot(xf, c * xf + d, "-", color="grey", lw=1.0,
            label=(fr"$n^{{v,{{\rm crit}}}}_\star = "
                   fr"{c:+.2f}\,\sigma {d:+.2f}$  "
                   fr"($R^2 = {R2_scaling:.2f}$)"))
    # Prediction line: c = A * (density_ratio); plot the unweighted
    # density-ratio interpretation slope at the fitted value.
    ax.plot(xf, A * xf, ":", color="black", lw=0.8, alpha=0.6,
            label=fr"$A\,\sigma = {A:.2f}\,\sigma$ (mean density)")
    ax.set_xlabel(r"$\sigma$ (mean density)")
    ax.set_ylabel(r"$n^{v,\,{\rm crit}}_\star$")
    ax.set_title(r"(b) Linear scaling test, weighted fit",
                 fontsize=8, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    fig.tight_layout()
    out_pdf = FIG / "fig_double_sigma_sweep.pdf"
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    print(f"saved figure: {out_pdf.name}")


if __name__ == "__main__":
    main()
