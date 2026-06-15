"""Analyse the revision FSS runs A1 (9-size slopes) and D1 (multi-density).

A1: merge the published seven-size homogeneous ten-seed series with the
two new sizes L=180,256 into a nine-size chi_max(L) series, refit the
slopes with a seed-bootstrap CI, recompute the synergy Delta, and report
the large-L local slope so we can say whether the excess survives or
saturates.

D1: chi_max(L) slopes at rho0 = 1.0 and 3.0 (sizes 22..128) to test
whether the motility-active vs fixed-noise split reproduces away from
rho0 = 2.22.

All numbers print to stdout; nothing is written. Seeds are the unit of
replication; bootstrap resamples seeds.
"""
from __future__ import annotations

import numpy as np

DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data"
RNG = np.random.default_rng(20260615)
B = 10000
MODES = ["vicsek_gauss", "baseline", "v2_limit", "v3_limit", "full"]
DISP = {"vicsek_gauss": "Vicsek", "baseline": "Cauchy",
        "v2_limit": "noise-shape", "v3_limit": "motility", "full": "full"}


def chi_max_of_L(chi_mode_L_eta_seed, seed_idx):
    """chi_max(L) for a seed sub-sample: max over eta of seed-mean chi."""
    sub = chi_mode_L_eta_seed[:, :, seed_idx]          # (L, eta, nsub)
    return np.nanmax(np.nanmean(sub, axis=2), axis=1)  # (L,)


def slope(Ls, chimax):
    return np.polyfit(np.log(Ls), np.log(chimax), 1)[0]


def fit_with_ci(Ls, chi_modes):
    """chi_modes: dict mode -> (L,eta,seed). Returns slope + 95% CI per mode."""
    nseed = next(iter(chi_modes.values())).shape[2]
    out = {}
    for m, chi in chi_modes.items():
        a0 = slope(Ls, chi_max_of_L(chi, np.arange(nseed)))
        bs = np.array([slope(Ls, chi_max_of_L(
            chi, RNG.integers(0, nseed, nseed))) for _ in range(B)])
        out[m] = (a0, np.percentile(bs, 2.5), np.percentile(bs, 97.5), bs)
    return out


def main():
    # ---- A1: nine-size merge ----
    d7 = np.load(DATA / "double_fss_homog10_nocone.npz", allow_pickle=True)
    dL = np.load(DATA / "double_fss_homog10_largeL_nocone.npz",
                 allow_pickle=True)
    assert list(d7["modes"]) == MODES and list(dL["modes"]) == MODES
    Ls9 = np.concatenate([d7["Ls"], dL["Ls"]])               # 9 sizes
    chi9 = np.concatenate([d7["chi"], dL["chi"]], axis=1)     # (5,9,10,10)
    order = np.argsort(Ls9)
    Ls9 = Ls9[order]
    chi9 = chi9[:, order]
    print("=== A1: nine-size FSS, L =", [int(x) for x in Ls9], "===")
    chi_modes = {m: chi9[i] for i, m in enumerate(MODES)}

    # chi_max(L) table
    print("\nchi_max(L) (pooled seed-mean):")
    for i, m in enumerate(MODES):
        cm = chi_max_of_L(chi9[i], np.arange(10))
        print(f"  {DISP[m]:>11}: " + "  ".join(f"{v:7.2f}" for v in cm))

    res9 = fit_with_ci(Ls9, chi_modes)
    print("\nslopes (9 sizes), [95% bootstrap CI]:")
    for m in MODES:
        a, lo, hi, _ = res9[m]
        print(f"  {DISP[m]:>11}: {a:+.2f}  [{lo:+.2f}, {hi:+.2f}]")

    # slope on the original 7 sizes for comparison
    res7 = fit_with_ci(Ls9[:7], {m: chi9[i][:7] for i, m in enumerate(MODES)})
    print("\nslopes (7 sizes, for comparison):")
    for m in MODES:
        print(f"  {DISP[m]:>11}: {res7[m][0]:+.2f}")

    # synergy Delta on 9 sizes, paired seed-bootstrap
    iC, iN, iM, iF = 1, 2, 3, 4   # Cauchy, noise, motility, full
    def delta_from(seed_idx):
        a = {i: slope(Ls9, chi_max_of_L(chi9[i], seed_idx))
             for i in (iC, iN, iM, iF)}
        return (a[iF] - a[iC]) - (a[iN] - a[iC]) - (a[iM] - a[iC])
    d9_0 = delta_from(np.arange(10))
    d9_bs = np.array([delta_from(RNG.integers(0, 10, 10)) for _ in range(B)])
    print(f"\nDelta_9 = {d9_0:+.2f}  "
          f"[{np.percentile(d9_bs,2.5):+.2f}, {np.percentile(d9_bs,97.5):+.2f}]"
          f"  P(Delta>0) = {(d9_bs>0).mean():.3f}")

    # large-L local slope: does the excess survive or saturate?
    print("\nlocal slope a_eff = dln chi_max / dln L over the top intervals:")
    for m in (iM, iF):
        cm = chi_max_of_L(chi9[m], np.arange(10))
        for j in range(6, len(Ls9) - 1):   # 90->128, 128->180, 180->256
            aeff = (np.log(cm[j + 1]) - np.log(cm[j])) / \
                   (np.log(Ls9[j + 1]) - np.log(Ls9[j]))
            print(f"  {DISP[MODES[m]]:>11} {int(Ls9[j])}->{int(Ls9[j+1])}: "
                  f"a_eff = {aeff:+.2f}")

    # ---- D1: multi-density split ----
    for rho, fn in [(1.0, "double_fss_density_rho1p0_nocone.npz"),
                    (3.0, "double_fss_density_rho3p0_nocone.npz")]:
        dd = np.load(DATA / fn, allow_pickle=True)
        Ls = dd["Ls"]; chi = dd["chi"]
        print(f"\n=== D1: rho0 = {rho}, L =", [int(x) for x in Ls], "===")
        res = fit_with_ci(Ls, {m: chi[i] for i, m in enumerate(MODES)})
        for m in MODES:
            a, lo, hi, _ = res[m]
            cm = chi_max_of_L({mm: chi[k] for k, mm in enumerate(MODES)}[m],
                              np.arange(chi.shape[3]))
            print(f"  {DISP[m]:>11}: a = {a:+.2f}  [{lo:+.2f}, {hi:+.2f}]"
                  f"   chi_max(L_max) = {cm[-1]:.1f}")


if __name__ == "__main__":
    main()
