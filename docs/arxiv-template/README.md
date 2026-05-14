# arXiv submission template (deferred until v1.0)

This directory holds `fuzuli.tex` — a LaTeX version of the v0.1 release writeup
formatted as an arXiv preprint.

**It is intentionally not being submitted right now.** The release strategy
(see `/RELEASE-PLAN.md` at the project root) is to publish each Fuzuli version
(v0.1, v0.5, v1.0, encoder track, embeddings) on HuggingFace as model-card
releases, and to write the comprehensive arXiv paper only when the project
reaches v1.0 maturity. That paper will be a multi-version technical report
covering the full autoresearch methodology, cross-version scaling
observations, and head-to-head comparisons against BERTurk + the bucolin
HisTR / OTA-BOUN / OTC baselines from Özateş et al. (2025) — material that
does not yet exist for v0.1 alone.

`fuzuli.tex` is preserved here as the structural template for that future
paper. Sections, citation style, table layouts, and figure references can
all be re-used. The current content is roughly 60% of what the v1.0 paper
will contain; the remaining 40% (cross-version results, baseline
comparisons, the matured methodology section) will be added then.

To compile the current template (for inspection only — do not submit):

```bash
cd docs/arxiv-template
cp -r ../article-fuzuli-v0.1/assets ./assets    # diagrams live in the article dir
pdflatex fuzuli.tex
pdflatex fuzuli.tex    # second pass for references
```

When v1.0 ships, this directory becomes the active submission. Until then,
leave it as-is.
