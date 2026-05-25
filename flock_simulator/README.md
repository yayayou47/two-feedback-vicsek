# `flock_simulator` — omnidirectional two-feedback Vicsek--Couzin

Modular Python + Numba simulator for the two-feedback
Vicsek--Couzin model used in the ``version4`` paper. The
package replaces the legacy ``version4/src/vicsek_double_adaptive.py``
simulator with three substantive changes:

* the **rear blind cone is removed** (omnidirectional sensing);
* the simulator is **modular** (core / observables / tests /
  scripts), so each piece can be tested in isolation;
* the random number generator is a NumPy ``Generator`` (PCG64)
  seeded via ``SeedSequence``, so trajectories are bit-for-bit
  reproducible on a given architecture.

## Quickstart

```python
from flock_simulator import FlockParams, FlockSimulator

p = FlockParams(N=2000, L=30.0, eta=0.10, seed=0)
sim = FlockSimulator(p)
for _ in range(1000):
    sim.step()

print(sim.polarisation(), sim.density_separation_index())
```

The constructor validates the parameters (radii ordering,
$v_{\min} \le v_{\max}$, $\alpha \in (0, 2]$, $L > 0$, etc.).
Use ``FlockParams(n_star_v=..., n_star_alpha=..., slope_v=...,
slope_alpha=...)`` to run the **decoupled-sigmoid** variant where
the motility and noise-shape channels have independent
thresholds.

## Module map

```
flock_simulator/
├── simulator.py          FlockParams + FlockState + FlockSimulator
├── core/
│   ├── geometry.py       cell-list, angle wrap (numba)
│   ├── interactions.py   Couzin priority kernel (numba, no cone)
│   ├── noise.py          alpha-stable kicks (Chambers--Mallows--Stuck)
│   └── adaptivity.py     shared sigmoid -> (v_i, alpha_i)
├── observables/
│   ├── order.py          phi, chi, Binder cumulant
│   ├── spatial.py        density-separation index s_sep
│   └── temporal.py       heading autocorrelation C(tau)
├── tests/                pytest suite (42 tests, 87% coverage)
└── scripts/              run-driver scripts for each manuscript figure
```

## Scripts

The ``scripts/`` directory contains one driver per re-run
protocol. Each script mirrors a legacy ``version4/src/run_*.py``
file but uses ``flock_simulator`` and the omnidirectional kernel.

* ``validate_reproducibility.py`` — short validation grid
  (``L=30``, $\eta=0.10$, 10 seeds, 500 + 1000 steps). Writes
  ``data/no_cone_reference.npz`` with ``--save``; without the
  flag, compares the current output to that reference at
  ``rtol=1e-12`` and exits non-zero on any divergence.
* ``compare_cone_vs_nocone.py`` — runs the legacy with-cone
  simulator and the new omnidirectional simulator on the same
  10-seed validation grid and writes the cone-vs-no-cone diff
  to ``data/cone_vs_nocone_validation.npz``.
* ``run_stage3_batch.sh`` — sequential wrapper that runs every
  Stage-3 protocol (snapshot, plane, orderpdf, finegrid,
  vicsek_gauss_ref, profile, plane_L30, clusters_hs, gr_hs,
  decoupled, autocorr) in order.
* ``run_stage3_heavy.sh`` — same for the heavy batch (Lfine,
  L_highstat, orderpdf_largeL, micro_hs_largeL, micro_hs_L128,
  orderpdf_L128).
* ``run_double_*_nocone.py`` — individual protocol drivers
  (see the legacy ``version4/src/run_double_*.py`` files for
  their original protocols).

## Tests

```bash
../.venv/bin/python -m pytest flock_simulator/tests/ -q
# 42 passed in 18s
```

To check coverage:

```bash
../.venv/bin/python -m pytest flock_simulator/tests/ \
    --cov=flock_simulator -q
# 87% reported coverage. The remaining uncovered code is in
# @njit-compiled functions that coverage.py cannot instrument
# (numba bypasses Python tracing); they are exercised by every
# simulator test.
```

To type-check (strict):

```bash
../.venv/bin/python -m mypy --config-file ../mypy.ini \
    flock_simulator/__init__.py flock_simulator/simulator.py \
    flock_simulator/core/ flock_simulator/observables/
# Success: no issues found in 11 source files
```

The ``mypy.ini`` at the repo root applies relaxed rules to the
test and script directories.

## Reproducibility

Each ``(seed, mode, L)`` triple gives a bit-for-bit identical
trajectory on a given architecture. The
``validate_reproducibility.py`` script enforces this against a
saved reference (``data/no_cone_reference.npz``). The Numba JIT
cache is local; the alpha-stable kick is drawn from a single
NumPy ``Generator`` outside the JIT loop and is therefore not
affected by the choice of parallel threads.

## Provenance

This package was built during the response to a biological-
relevance review of the v4 manuscript that flagged the rear
blind cone as biologically unwarranted for fish, starlings, and
swarming bacteria. See ``REFACTOR_REPORT.md`` at the repo root
for the full Stage 1 + Stage 3 history (pilot, FSS extension,
g(r) persistence, decoupled sigmoid, etc.).
