# Fuzuli release plan

This document records the publication strategy for the Fuzuli Ottoman
Turkish language model project. It is the canonical answer to "where and
when do we publish what." Last updated 2026-05-14.

## Strategic principle

**HuggingFace is the version registry. arXiv is the summative paper at
maturity.**

Each Fuzuli version (v0.1 → v0.5 → v1.0, plus the parallel encoder and
embeddings tracks) ships as an HF model release with a long-form model card
that doubles as a release blog post. The arXiv paper is deferred until v1.0
so that the citation lands on the mature artifact, not on early
proof-of-concept numbers.

### Why this order

1. **arXiv citations are forever.** A 2026 paper at v0.1 (84M params, 8M
   unique training tokens, 200× under Chinchilla) would anchor every
   future Google Scholar / reviewer impression to those numbers, even
   after v1.0 ships a stronger model.
2. **HF is inherently versioned.** Every release on HF gets its own
   citable URL with a SHA-pinned snapshot. We don't need three separate
   arXiv papers to publish three versions.
3. **The research narrative is stronger written backwards.** A single
   arXiv paper written alongside v1.0 can frame v0.1 as the proof of
   concept, v0.5 as the scaling experiment, and v1.0 as the contribution
   — much cleaner than three split releases.
4. **The v0.1 release writeup is structurally a launch blog, not a
   peer-reviewable paper.** First-person voice, project-history
   narrative, roadmap section — that reads well as an HF model card and
   poorly as an arXiv submission. The arXiv version requires
   substantial rewriting (more rigorous baselines, N>5 seed
   reproducibility, tighter scope, comparison to BERTurk + bucolin
   benchmarks).
5. **SIGTURK 2024 and arXiv 2501.04828 already cover the
   "Ottoman corpus + BERTurk continual pretraining" line.** Putting
   another Ottoman paper on arXiv right now would split visibility within
   the small Turkic-NLP community. The right next arXiv contribution
   from this project is the summative one.

## Release pipeline

| Phase | Artifact | Where | When |
|---|---|---|---|
| **v0.1** | 84M decoder model + tokenizer + release blog | HuggingFace (`fatihburakkaragoz/fuzuli-base`) | First |
| **v0.5** | ~120M decoder model + tokenizer + release blog (updated corpus) | Same HF repo, new version branch | After corpus reaches ~30M unique BPE tokens |
| **v1.0** | ~150M decoder model + tokenizer + release blog | Same HF repo, new version branch | After corpus reaches ~50M unique BPE tokens |
| **encoder v0.1** | BERT-style MLM model on current corpus | HuggingFace (`fatihburakkaragoz/fuzuli-encoder`) | Parallel to decoder v0.5 timing |
| **embeddings v0.1** | Sentence-encoder derived from encoder | HuggingFace (`fatihburakkaragoz/fuzuli-embed`) | After encoder v0.1 |
| **arXiv paper** | Comprehensive multi-version technical report | arXiv cs.CL | After v1.0 ships |

## What each HF release includes

- **Weights** (`model.pt` or `pytorch_model.bin`)
- **Tokenizer** (`tokenizer.json` — frozen since v0.1)
- **Model card** (`README.md`) — the full release blog with diagrams
- **Diagrams** (`article/assets/*.png`) — embedded in the model card
- **License files** — Apache-2.0 for code/weights, CC-BY-4.0 for the article

Pushed via `cli.py publish-baseline --repo-id fatihburakkaragoz/fuzuli-base
--version vX.Y --public --include-article`.

## What the eventual arXiv paper covers

Targeted for submission alongside the v1.0 HF release. Content scope:

1. **Corpus construction lineage.** Builds on the methodology of Karagöz
   et al. (SIGTURK 2024) and the infrastructure of Özateş et al. (arXiv
   2501.04828). Reports final corpus stats and the cleaning pipeline
   evolution across versions.
2. **The autoresearch methodology.** Tier-based mutability, strict-no-trades
   policy, SQLite ledger as agent memory, forbidden-pattern scanning —
   reported with full design rationale and prior-art comparison
   (FunSearch, AlphaEvolve, Voyager, Reflexion).
3. **Cross-version scaling observations.** v0.1 → v0.5 → v1.0 perplexity,
   lexicon, and downstream-task numbers. Where in the Chinchilla
   landscape each version sits. The data-ceiling diagnosis with N>30
   seed-variance evidence (not the N=5 anecdote in the v0.1 blog).
4. **Head-to-head comparisons.** Against BERTurk continual-pretrained
   (Karagöz 2024) and the bucolin task models (Özateş 2025) on shared
   benchmarks: HisTR NER, OTA-BOUN dependency parsing, OTC perplexity.
5. **Limitations and negative results.** What the autoresearch loop
   discovered does NOT work (the rejected experiments — Lion optimizer,
   width changes, etc.) — useful field knowledge.

Template: `docs/arxiv-template/fuzuli.tex` (currently a v0.1-only draft;
will be extended at v1.0 time).

## Decision triggers — when does v0.5 / v1.0 happen?

The corpus is the binding constraint, not training compute. Triggers for
moving to the next version are corpus-driven:

- **v0.5 trigger:** corpus expanded to ~30M unique BPE tokens (≈4× current).
  Sources: remaining Sebilürreşad volumes, late-Ottoman periodicals
  (*İkdam*, *Tasvir-i Efkâr*, *Mizan*, *Tercümân-ı Ahvâl*), Turkish-studies
  dissertations with Ottoman primary sources.
- **v1.0 trigger:** corpus at ~50M unique BPE tokens. Adds systematic
  collection of academic encyclopedia entries (TDV İslâm Ansiklopedisi),
  TÜBA-OTL alignment, possibly an annotated subset of Ottoman archival
  documents.

Until those triggers fire, additional autoresearch iteration on the
current corpus is bounded by the data-ceiling diagnosis from v0.1 and
should not produce a new public release.

## What is NOT on this plan

- **v0.1 → arXiv now.** Explicitly deferred. See "Strategic principle" above.
- **Mid-version blog posts on Substack / Medium.** HF model cards serve
  this role.
- **Translation models.** A from-scratch Ottoman → Modern Turkish
  translator is interesting but blocked by paired-data scarcity; not in
  scope for v1.0 or earlier.
- **Multi-language Turkic extension** (Crimean Tatar, Nogay, Chagatai).
  Out of scope as established in the original spec.
