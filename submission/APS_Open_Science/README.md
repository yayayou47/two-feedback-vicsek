# APS Open Science submission package

Target: **APS Open Science**, original research article.

Self-contained snapshot of the submission (frozen copy; the live
source lives in `version4/manuscript/`).

**Status (2026-06-18):** built from the revised manuscript after the three
referee rounds (nine-size FSS, multi-density control, matched-noise
reframing: the noise-shape channel acts through a single dense-noise
rectification route, additive at matched noise) and condensed ~10%
(8383 to 7580 words). Retargeted from the earlier JSTAT package to APS
Open Science house style.

## Contents

| File | Role |
|------|------|
| `cover_letter.{tex,pdf}` | Cover letter to the APS Open Science editorial board |
| `manuscript.tex` (+ `.pdf`) | Main article, **REVTeX 4.2** (`aps`) — **compiles here** with `apsrev4-2` |
| `supplement.{tex,pdf}`   | Supplementary material (figures S1–S2), generic `article` class |
| `refs.bib`               | Bibliography |
| `figures/`               | The 18 figure PDFs included by the `.tex` files |

## House style

The manuscript uses `\documentclass[aps,preprint,amsmath,amssymb,nofootinbib,superscriptaddress]{revtex4-2}`
with `\bibliographystyle{apsrev4-2}`. Both `revtex4-2.cls` and
`apsrev4-2.bst` ship with TeX Live, so this builds locally (unlike the
previous IOP `iopart` package). `preprint` gives the single-column APS
submission format; switch to `reprint` for the compact two-column look
(then make the wide Table I and the multi-panel figures `table*`/`figure*`).
REVTeX ships no dedicated APS Open Science option, so the generic `aps`
journal option is used; add `pre` (sibling PR format) for a closer match,
only minor formatting changes.

The supplement stays in the generic `article` class; its S-numbered labels
feed the main text's `xr-hyper` cross-references (Figs.~S1/S2) via
`supplement.aux`.

## Build

The main file and the supplement cross-reference each other with
`xr-hyper`, so build the supplement first (for `supplement.aux`), then the
manuscript twice:

```
latexmk -pdf supplement.tex
latexmk -pdf -bibtex manuscript.tex
latexmk -pdf manuscript.tex
latexmk -pdf cover_letter.tex
```

Figures are referenced via `\graphicspath{{figures/}}` (this folder is
self-contained; it does not depend on `../figures/`).

## Code & data

All simulation code, run drivers, figure scripts and precompiled
data: <https://github.com/yayayou47/two-feedback-vicsek> (MIT
license; Zenodo DOI on acceptance).
