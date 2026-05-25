# `flock_simulator` package report

## Overview

`flock_simulator/` is the canonical simulator and analysis
toolkit for the v4 manuscript. The package implements the
two-feedback Vicsek--Couzin model with omnidirectional sensing
on a periodic linked-cell list, JIT-compiled with Numba, and
ships with a pytest suite, a reproducibility validator, a
typing configuration, and one driver script per manuscript
figure.

The legacy ``version4/src/vicsek_double_adaptive.py`` simulator
is preserved alongside as the historical reference; all
manuscript numbers and figures are produced by the package.

## Module map

```
flock_simulator/
├── __init__.py
├── simulator.py          # FlockSimulator + FlockParams + FlockState
├── core/
│   ├── geometry.py       # build_cell_list, normalize_angle (numba)
│   ├── interactions.py   # zonal_update (numba parallel)
│   ├── noise.py          # stable_rvs_vector (Chambers-Mallows-Stuck)
│   └── adaptivity.py     # shared_sigmoid_fields -> (v_i, alpha_i)
├── observables/
│   ├── order.py          # phi, chi, U_4
│   ├── spatial.py        # density_separation_index, pair-corr
│   │                     # separation index, DBSCAN cluster sizes
│   └── temporal.py       # heading_autocorrelation
├── tests/                # 48 tests, 87% reported coverage
└── scripts/              # one driver per manuscript run
```

## Tests, types, reproducibility

```bash
make tests         # 48 passed in ~18 s
make typecheck     # mypy --strict, 0 errors on the core package
make coverage      # 87 % coverage (numba @njit functions are
                   # exercised by every simulator test but cannot
                   # be instrumented by coverage.py)
make validate      # bit-for-bit reproducibility against the
                   # reference at data/no_cone_reference.npz
```

The reproducibility validator runs a fixed grid (L=30, eta=0.10,
N=2000, 500 warm + 1000 measurement steps, 10 seeds) and
compares each per-seed `<phi>` and `<s_sep>` to the saved
reference at `rtol = 1e-12`. With ``SeedSequence`` and PCG64 the
trajectories are bit-identical on a given architecture.

The strict mypy run covers the core package
(``__init__.py``, ``simulator.py``, ``core/``, ``observables/``,
11 source files) and reports no issues. The mypy configuration
at ``version4/mypy.ini`` applies relaxed rules to the
tests/ and scripts/ subpackages where return-type annotations
on test methods would be overhead.

## Stage 3 production summary

Stage 3 is the manuscript-data production phase. Each driver in
``scripts/`` runs one protocol and writes its output to
``data/*.npz``.

### FSS extension (~14 h)

  * ``run_pilot_nocone.py`` (53 min) -- four-mode pilot at
    L in {15, 22, 30, 45}, three seeds per cell, 10-point eta
    grid. Output: ``double_pilot_nocone.npz``.
  * ``run_fss_large_nocone.py`` (14 h total) -- five-mode
    extension to L in {64, 90, 128}, three seeds per cell.
    Outputs: ``double_L{64,90,128}_nocone.npz``.

### Cheap-to-moderate batch (~2 h 16, sequential)

  ``run_stage3_batch.sh`` runs the eleven cheap-to-moderate
  protocols in order:
  snapshot (1 min) -> plane (18 min) -> orderpdf (2 min) ->
  finegrid (9 min) -> vicsek_gauss_ref (13 min) ->
  profile (1 min) -> plane_L30 (24 min) -> clusters_hs (3 min)
  -> gr_hs (2 min) -> decoupled (2 min) -> autocorr (56 min).

### Heavy batch (~108 h, sequential)

  ``run_stage3_heavy.sh`` runs the six heavy protocols:
  Lfine (146 min) -> orderpdf_largeL (62 min) ->
  L_highstat (702 min) -> micro_hs_largeL (1304 min) ->
  micro_hs_L128 (261 min) -> orderpdf_L128 (4008 min).
  Total ~ 4.5 days wall-clock. Several jobs include rare
  Cauchy-burst seeds where a single iteration takes 6-8 h on
  the cell-list; these slow the run substantially.

## Headline findings

  * Susceptibility FSS at L in {15, 22, 30, 45, 64, 90, 128}:
    only the two motility-active modes develop a critical
    chi_max(L) scaling. The Vicsek--Gaussian, Cauchy reference
    and noise-shape-only ablation all sit at slope ~ 0
    (bootstrap CI brackets zero); a_motility = +2.34 and
    a_full = +2.95 with the full mode super-additive.
  * Synergy Delta_n = a_full + a_Cauchy - a_noise - a_motility
    is positive across the full size budget (+0.69, +1.45,
    +0.46, +0.48), with bootstrap CIs that exclude zero at
    n = 4 and n = 5.
  * Dense-quartile g(r) gap persists across four sizes:
    +0.153 (L=30), +0.150 (L=64), +0.127 (L=90), +0.106
    (L=128); plateau in [+0.11, +0.15], z >= 7 at every size.
  * Heading-autocorrelation gap holds the same plateau over
    four decades in lag at L=30 with bootstrap CIs excluding
    zero throughout.
  * Decoupled-sigmoid timing test: shared sigmoid +0.248
    (z=+27) over the motility ablation, motility-first
    variant +0.229 (z=+18), alpha-first variant -0.270
    (z=-32, sign flip). The timing of the two sigmoids is
    qualitatively important: the motility response must engage
    first for the rectification to operate.
  * Cluster diagnostics: cluster count is not a clean
    discriminator at L=30 (z = -1.2 between full and
    motility-only), but the maximum-cluster size is larger in
    the double-adaptive model (+121 particles, z=+2.7). The
    rectification fuses dense patches into a few large droplets
    rather than producing many small ones.
  * Order-parameter distribution: unimodal at every size we
    have probed, with U_4 plateauing at ~ 0.56-0.57 at L=128
    and mild platykurtosis (kappa = -0.2 to -0.3). No bimodal
    coexistence; the typical polar order shifts upward with
    size in the motility-active modes.

## Cross-checks

  * **Reproducibility** -- ``make validate`` enforces bit-for-bit
    agreement against the reference snapshot.
  * **Cone-vs-no-cone diff** --
    ``scripts/compare_cone_vs_nocone.py`` runs the legacy
    simulator and the new simulator on the same 10-seed
    validation grid; output:
    ``data/cone_vs_nocone_validation.npz``. This is preserved
    as documentation; the manuscript reports the new
    simulator only.
  * **Grid-free diagnostics** --
    ``scripts/compare_diagnostics.py`` cross-checks the
    grid-based s_sep against a pair-correlation separation
    index and the grid-threshold cluster finder against
    DBSCAN. The grid-free implementations live in
    ``observables/spatial.py``.

## Compute and software

  * Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.8,
    Numba 0.65.1 with LLVM 0.47.0, Matplotlib 3.10.9, on
    Linux x86_64.
  * The interaction kernel is ``@njit(cache=True, parallel=True)``
    with ``prange`` over particles; alpha-stable kicks are drawn
    in a single vectorised NumPy expression outside the JIT
    loop. Per-step cost at N=500, L=15 after JIT warm-up is
    ~ 1 ms.
  * ``Makefile`` targets: ``tests``, ``typecheck``, ``coverage``,
    ``validate``, ``figures``, ``manuscript``, ``pilot``,
    ``stage3``, ``heavy``, ``all``, ``clean-pdf``,
    ``clean-cache``.

## Install and run

```sh
cd version4
../.venv/bin/python -m pip install -r requirements.txt   # if needed
make all   # tests + typecheck + figures + manuscript
```
