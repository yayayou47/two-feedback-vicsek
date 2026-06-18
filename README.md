# A two-feedback Vicsek-Couzin flock

Simulation code, analysis pipeline, figure generation, manuscript
source, and movies for the paper

> **A two-feedback Vicsek–Couzin flock: dense-phase noise
> rectification as the unifying action of density-gated
> motility–noise coupling.**
> Yaya Youssouf Yaya (ORCID
> [0000-0003-0781-4923](https://orcid.org/0000-0003-0781-4923))
> and Mamadou Sy.

The model couples two density-dependent feedbacks of a
Vicsek–Couzin flock — the self-propulsion speed $v_i$ and the
stability index $\alpha_i$ of an $\alpha$-stable angular kick —
through one shared sigmoid in the local neighbour count $n_i$.
A crowded particle is simultaneously slow and Gaussian-like;
an isolated particle is simultaneously fast and Cauchy-like.
The paper benchmarks the resulting dynamics against the
original Vicsek model on a controlled nine-size finite-size
scaling ($L$ up to $256$) across three densities, and shows that
the noise-shape channel acts through a single route: it lowers
the effective angular noise inside the motility-built dense
phase. A matched-noise control makes the static $g(r)$, dynamic
$C(\tau)$ and finite-size susceptibility signatures collapse
together, so the two feedbacks are additive rather than an
emergent thermodynamic synergy.

## Repository layout

```
version4/
├── src/                     simulation, analysis, figure code
│   ├── vicsek.py                cell-list metric Vicsek kernel
│   ├── noise.py                 alpha-stable noise generator
│   ├── style.py                 matplotlib style
│   ├── vicsek_double_adaptive.py   two-feedback simulator (shared sigmoid)
│   ├── vicsek_decoupled.py      two-feedback simulator (independent sigmoids)
│   ├── run_double_pilot.py      4-mode FSS pilot at L = {15, 22, 30, 45}
│   ├── run_double_L64.py        controlled L = 64 row
│   ├── run_double_L90.py        controlled L = 90 row
│   ├── run_double_L128.py       controlled L = 128 row
│   ├── run_double_finegrid.py   small-eta robustness check
│   ├── run_vicsek_gauss_ref.py  original Vicsek (Gaussian) reference
│   ├── run_double_snapshot.py   real-space snapshot data
│   ├── run_double_plane.py      (n_star, s) plane at L = 22
│   ├── run_double_plane_L30.py  refined plane at L = 30, 5 seeds
│   ├── run_double_orderpdf.py   P(<phi>) at L = 30
│   ├── run_double_profile.py    polar-axis density profile
│   ├── run_double_Lfine.py      fine-L scan {38, 50, 60, 75, 105}
│   ├── run_double_L_highstat.py ten-seed gap test at L = {50, 64, 80}
│   ├── run_double_clusters_hs.py   ten-seed cluster-size at L = 30
│   ├── run_double_gr_hs.py         ten-seed g(r) density-stratified at L = 30
│   ├── run_double_micro_hs_largeL.py  microscopic HS at L = 64, 90
│   ├── run_double_micro_hs_L128.py   microscopic HS at L = 128
│   ├── run_double_decoupled.py     decoupled-sigmoid test
│   ├── analyse_gr_decay.py         g(r) gap profile across L (analysis only)
│   ├── refit_slopes.py             7-size FSS slope refit
│   ├── make_figures.py             all figure renderers
│   ├── run_movie.py                MP4 video generator
│   └── legacy/                     superseded scripts (3-seed budgets)
├── data/                    .npz simulation outputs and .log run logs
├── figures/                 PDF + PNG renders consumed by the manuscript
├── manuscript/              LaTeX source, refs.bib, compiled PDF
├── videos/                  MP4 supplementary movies
├── notes/                   internal working notes
├── PLAN.md                  project roadmap and bookkeeping
├── requirements.txt         pinned Python dependencies
├── LICENSE                  MIT
└── README.md
```

## Installation

```bash
git clone <repo-url> two-feedback-vicsek
cd two-feedback-vicsek
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pipeline relies on **numba** for the cell-list Vicsek kernel.
Without numba the simulator falls back to pure Python and runs
roughly 60× slower. A working `numba` is therefore strongly
recommended.

## Reproduce the results

The full controlled finite-size scaling and the high-statistics
microscopic tests take a few hours on a modern laptop. The
sequence below regenerates every diagnostic from scratch.

```bash
cd src

# 1. Foundational seven-size FSS  ( ~40 min )
python run_double_pilot.py            # L = 15, 22, 30, 45
python run_double_L64.py              # L = 64
python run_double_L90.py              # L = 90
python run_double_L128.py             # L = 128
python run_double_finegrid.py         # small-eta robustness
python run_vicsek_gauss_ref.py        # Vicsek-Gaussian reference
python refit_slopes.py                # 7-size slope refit + Delta_n

# 2. Real-space and parameter robustness  ( ~50 min )
python run_double_snapshot.py
python run_double_plane.py
python run_double_plane_L30.py
python run_double_Lfine.py

# 3. Order parameter and spatial probes  ( ~10 min )
python run_double_orderpdf.py
python run_double_profile.py

# 4. Macroscopic gap test  ( ~30 min )
python run_double_L_highstat.py

# 5. Microscopic ten-seed tests  ( ~60 min )
python run_double_clusters_hs.py
python run_double_gr_hs.py
python run_double_micro_hs_largeL.py
python run_double_micro_hs_L128.py
python analyse_gr_decay.py            # analysis only, no simulation

# 6. Decoupled-sigmoid extension  ( ~5 min )
python run_double_decoupled.py

# 7. Figures and supplementary movies
python make_figures.py
python run_movie.py --all --regimes   # ~25 min for the full batch
```

Every script writes its `.npz` to `../data/` and prints a short
summary on standard output. Figure rendering reads from
`../data/` and writes to `../figures/`. The manuscript is then
compiled from `manuscript/` with `latexmk -pdf manuscript.tex`.

## Key results

The paper's robust findings are:

1. **A clean two-regime split.** Heavy-tailed noise on its own
   grows no size-dependent susceptibility, whereas the motility
   channel builds a phase-separated dense phase (`s_sep` rising
   from ≈ 1.2 for the fixed-noise references to ≈ 1.7). The split
   reproduces across three densities.
2. **The two channels are additive, not synergistic.** Inside the
   dense phase the noise-shape channel rectifies Cauchy kicks into
   Gaussian ones, lowering the effective angular noise. A
   matched-noise control makes the static `g(r)`, dynamic `C(τ)`
   and finite-size susceptibility signatures collapse together
   (the susceptibility excess `Δ₉` from `+0.45` to `+0.02`, the
   `g(r)` and `C(τ)` gaps reversing sign), so the apparent
   super-additive synergy is one dense-noise effect rather than an
   emergent thermodynamic cooperation.

Claims rest on nine-size, ten-seed finite-size scans with
seed-bootstrap confidence intervals and non-parametric (Wilcoxon
signed-rank, Mann–Whitney) tests; the finite-size slopes are
reported as finite-size crossover rates, not asymptotic exponents.

## Citing

Please cite the manuscript as follows once published, and the
software via `CITATION.cff` (to be added on release).

## License

MIT — see `LICENSE`.
