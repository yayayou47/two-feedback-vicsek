# JSTAT submission package

Target: **Journal of Statistical Mechanics: Theory and Experiment**
(JSTAT, IOP / SISSA), original research article.

Self-contained snapshot of the submission (frozen copy; the live
source lives in `version4/manuscript/`).

## Contents

| File | Role |
|------|------|
| `cover_letter.{tex,pdf}` | Cover letter to the JSTAT editorial board |
| `manuscript.{tex,pdf}`   | Main article (figures 1–10) |
| `supplement.{tex,pdf}`   | Supplementary material (figures S1–S2) |
| `refs.bib`               | Bibliography |
| `figures/`               | The 17 figure PDFs included by the two `.tex` files |

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
