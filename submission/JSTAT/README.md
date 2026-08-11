# JSTAT submission package

Target: **Journal of Statistical Mechanics: Theory and Experiment**
(SISSA / IOP Publishing), original research article.

Suggested section: **Classical statistical mechanics, equilibrium and
non-equilibrium**. Secondary fit: *Interdisciplinary statistical
mechanics*.

Submission portal: <https://jstat.sissa.it/>

## Why JSTAT rather than a generalist venue

The paper's headline is deflationary — an apparent super-additive
synergy between two adaptive channels turns out to be an artefact of
comparing noise shapes at matched nominal amplitude instead of matched
effective decorrelation — and its transferable content is a control
protocol plus a mechanism, not a new universality class. That profile
suits a journal whose acceptance criteria are scientific quality,
originality, relevance and comprehension, and which explicitly welcomes
careful non-equilibrium statistical-mechanics work. It is the wrong
profile for a generalist venue triaging on breadth of impact.

## Contents

Self-contained snapshot; the live source lives in
`version4/manuscript/`.

| File | Role |
|------|------|
| `cover_letter.{tex,pdf}` | Cover letter to the JSTAT editorial board (1 page) |
| `manuscript.tex` (+ `.pdf`) | Main article, generic `article` class, 24 pp. |
| `supplement.tex` (+ `.pdf`) | Supplementary material, figures S1–S7, tables S1–S2, 7 pp. |
| `refs.bib`               | Bibliography, 73 entries, all cited |
| `figures/`               | The 18 figure PDFs included by the `.tex` files |
| `suggested_referees.txt` | Referee suggestions carried over from the earlier package |
| `yaya_signature.png`     | Signature image used by the cover letter |

The manuscript uses the generic `article` class, which IOP accepts for
initial submission; `iopart` conversion is only required at acceptance.

## Title

Retitled to match the repositioning:

> Density-gated noise rectification in a Vicsek–Couzin flock: one
> dense-phase mechanism, additive at matched decorrelation

The title now names the mechanism, localises it, and states the condition
under which the two channels turn out to be additive, so it tracks the
cover letter instead of leading on the older "unifying action" framing.
It is carried through `manuscript.tex`, `supplement.tex`, both cover
letters, `version4/CITATION.cff` and `version4/README.md`.

## Still to do before submitting

The full list, with the sources checked, is in
`submission_form/07_checklist.txt`. The blocking items:

1. **AI-usage declaration** — JSTAT requires it explicitly and the
   manuscript has none. Three wordings are ready, commented out at the
   end of `manuscript.tex`; uncomment the one that matches reality.
2. **Tag `jstat-submission`** — the data-availability statement now
   promises it. Create and push it from `version4/`.
3. **Repository** — `two-feedback-vicsek` is public but its description
   still carries the pre-July title, and its last push predates the
   retitle.
4. **ORCID for Mamadou Sy** — absent from the manuscript.
5. **Referee emails** — five of six verified and sourced in
   `submission_form/04_suggested_referees.txt`; Hugues Chaté's is still
   to be found.

Zenodo is no longer referenced anywhere: the data-availability
statement cites the repository plus a git tag, which pins the exact
version without depending on an external archive.

## Submission form

`submission_form/` holds one text file per field of the JSTAT portal,
numbered in the order they are asked for, plus `check_citations.py`,
which fails if a cited key is missing from `refs.bib` or if a reference
listed as mandatory is absent from the printed `.bbl`.

## Build

The manuscript and the supplement cross-reference each other through
`xr-hyper`, so each needs the other's `.aux`; run the pair twice, then
the manuscript once more:

```
latexmk -pdf -bibtex manuscript.tex
latexmk -pdf -bibtex supplement.tex
latexmk -pdf -bibtex manuscript.tex
latexmk -pdf -bibtex supplement.tex
latexmk -pdf manuscript.tex
latexmk -pdf cover_letter.tex
```

Verified clean: `grep -c undefined manuscript.log` and the same for
`supplement.log` both return 0.
