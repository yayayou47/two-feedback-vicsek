# JSTAT submission package

Target: **Journal of Statistical Mechanics: Theory and Experiment**
(JSTAT, IOP / SISSA), original research article.

Self-contained snapshot of the submission (frozen copy; the live
source lives in `version4/manuscript/`).

**Status (2026-06-17):** refreshed from the revised manuscript after the
three referee rounds (nine-size FSS, multi-density control, matched-noise
reframing: the noise-shape channel acts through a single dense-noise
rectification route, additive at matched noise) and condensed ~10%
(8383 to 7580 words; 21 pp main + 3 pp SI).

## Contents

| File | Role |
|------|------|
| `cover_letter.{tex,pdf}` | Cover letter to the JSTAT editorial board |
| `manuscript.tex` (+ `.pdf`) | Main article, generic `article` class — **compiles here**, use for quick local builds / arXiv |
| `manuscript_iopart.tex`  | Main article in the **IOP `iopart` class** (JSTAT house style) — see note below |
| `supplement.{tex,pdf}`   | Supplementary material (figures S1–S2) |
| `refs.bib`               | Bibliography |
| `figures/`               | The 18 figure PDFs included by the `.tex` files |

### `manuscript_iopart.tex` — IOP house style

This is the JSTAT-formatted version: `\documentclass[12pt]{iopart}`,
`\title`/`\author`/`\address`/`\ead` front matter, `\maketitle` after the
abstract, a Keywords line, the Elsevier-style Highlights dropped, and
`\bibliographystyle{iopart-num}` (the `.bst` ships with TeX Live).

**It is not compiled in this folder**: `iopart.cls` is distributed by IOP
Publishing (not on CTAN) and is absent from this TeX install. Build it on
**Overleaf** (start from the "iopart" template, which provides the class)
or with IOP's downloadable author template; the body, figures and
bibliography are otherwise identical to `manuscript.tex`. The supplement
is kept in the generic class — its S-numbered labels still feed the
iopart manuscript's `xr` cross-references (Figs.~S1/S2) via
`supplement.aux`; convert it to `iopart` too if a uniform SI is wanted.

## Build

The main file and the supplement cross-reference each other with
`xr-hyper` (the main text cites Figs. S1/S2; the supplement cites a
main-text equation), so compile **both, twice**, for the
cross-document references to resolve:

```
latexmk -pdf manuscript.tex supplement.tex
latexmk -pdf manuscript.tex supplement.tex
latexmk -pdf cover_letter.tex
```

Figures are referenced via `\graphicspath{{figures/}}` (this folder
is self-contained; it does not depend on `../figures/`).

## Code & data

All simulation code, run drivers, figure scripts and precompiled
data: <https://github.com/yayayou47/two-feedback-vicsek> (MIT
license; Zenodo DOI on acceptance).
