---
title: ottoman-autoresearch — Phase 1 Design
date: 2026-05-12
status: Draft (pending user review)
author: zk + Claude (brainstorming pair)
phase: 1 of N (Phase 2+ deferrals documented below)
---

# ottoman-autoresearch — Phase 1 Design

## 1. Overview

### 1.1 Goal

Produce **`cdli/asena-base`**: a ~80M-parameter decoder-only language model trained **from scratch** on cleaned, Latinized post-1500 Ottoman Turkish, with a custom 24k BPE tokenizer, via an evaluator-driven autoresearch loop where **kimi** acts as the researcher and a deterministic factory holds final authority over accept/reject decisions.

> **Sizing note (2026-05-12 amendments):** The initial draft targeted 150–300M params on 1.5B tokens. After the actual corpus was imported and the BPE was trained, the real budget is **7.97M unique training tokens** (`anadolu-ocr-corpus` + `evliya-celebi-seyahatname-ocr` after Stages 1–4) — the custom 24k BPE compresses Ottoman much harder than cl100k (3.6 chars/tok), so the on-disk char count translated to a far smaller token budget than the dataset cards' cl100k counts suggested. The promotion target was therefore amended twice in one session: first to ~125M params / 300M tokens, then to **~80M params / ~60M tokens (~7 epochs)** once the actual BPE-token budget was known. Larger sizes are deferred to a Phase-1.5 amendment after the corpus grows past ~30M unique BPE tokens.

The published artifact is the model + tokenizer + training code + dataset, all under Apache-2.0 (code, weights) and CC-BY-4.0 (dataset). No fine-tune lineage. No foreign weights. The tokenizer, weights, and code are entirely owned by the project.

### 1.2 Non-goals (v1)

- Not a chatbot. `asena-base` is a next-token predictor in the GPT-2 sense.
- Not instruction-following. SFT laps are deferred to Phase 2 specialist heads.
- Not multi-variety. Crimean Tatar, Nogay, and Chagatai are explicitly out of scope for v1 (different languages, not just dialects — would produce interlingua slop at this scale).
- Not a fine-tune of any pretrained model.
- No MoE, no Mamba, no encoder-decoder. Decoder-only dense transformer only.

### 1.3 Project vocabulary

| Term | Meaning |
|---|---|
| **Factory** | The deterministic pipeline that turns *(proposal, base config, base data)* into *(checkpoint, eval scores, accept/reject, git commit/revert)*. Tier-1 locked. |
| **Researcher** | The kimi CLI subprocess that generates patches. Tier-2 surface only. |
| **Sprint** | A 5-minute training run on a small (30-60M) model used for autoresearch iteration. Discovers recipes. |
| **Promotion** | A ~1-2-hour training run on the publishable (~80M) model using the current accepted-baseline recipe. Produces shippable checkpoints. |
| **Freeze** | The one-time event locking the tokenizer + held-out eval set with SHA-256 hashes. Unfreeze is destructive on purpose. |
| **Eval** | The four-evaluator harness + strict-no-trades policy combiner that decides accept/reject. The authority. |
| **Baseline** | The current-best accepted model + scores + git SHA. Every new experiment compares against this, not against the previous experiment. |

## 2. Repository layout

```
ottoman-autoresearch/
├── data/
│   ├── extract/             # PDF/OCR → raw .txt (upstream, user-owned)
│   ├── raw/                 # *.parquet matching schema in §3.1
│   ├── clean/               # cleaned, post-Stage-4 output
│   ├── cleaning_rules.yaml  # AGENT-EDITABLE (Tier 2)
│   └── modern_loanwords.txt # human-curated blacklist (Tier 1)
├── tokenizer/
│   ├── train_bpe.py
│   └── asena-bpe-24k.json   # FROZEN after §3.4 (Tier 0)
├── train/                   # forked from karpathy/autoresearch
│   ├── train.py             # AGENT-EDITABLE (Tier 2)
│   ├── arch.py              # AGENT-EDITABLE (Tier 2)
│   └── configs/             # AGENT-EDITABLE (Tier 2)
│       ├── sprint.yaml
│       └── promotion.yaml
├── eval/                    # IMMUTABLE (Tier 0 + 1)
│   ├── heldout/             # frozen text + smoke prompts + lexicon (Tier 0)
│   │   ├── text/*.parquet
│   │   ├── smoke_prompts.yaml
│   │   ├── ottoman_lexicon.txt
│   │   └── FROZEN.lock
│   ├── heldout_ppl.py
│   ├── lexicon_score.py
│   ├── flatness.py
│   ├── smoke.py
│   └── policy.py
├── factory/                 # IMMUTABLE (Tier 1)
│   ├── orchestrator.py      # invoked indirectly via cli.py train-sprint
│   ├── db.py                # SQLite ledger
│   ├── git_ops.py
│   └── janitor.py
├── agent/                   # the researcher
│   └── prompts/
│       └── run-autoresearch.md
├── checkpoints/             # gitignored
├── experiments.sqlite       # gitignored
├── cli.py
├── README.md
├── SAFETY.md
└── pyproject.toml
```

### 2.1 Mutability tiers

| Tier | Paths | Who edits |
|---|---|---|
| **Tier 0 — Frozen forever** | `eval/heldout/`, `tokenizer/asena-bpe-24k.json` (after freeze) | Nobody. Editing invalidates all prior experiments. Unfreeze requires `cli.py unfreeze --i-know-what-im-doing --clear-ledger`. |
| **Tier 1 — Locked (human-only)** | `eval/*.py`, `factory/`, `cli.py`, `SAFETY.md`, `README.md`, `data/modern_loanwords.txt`, `agent/prompts/` | You. Rarely. With explicit ledger annotation. |
| **Tier 2 — Agent-editable** | `train/`, `data/cleaning_rules.yaml` | kimi inside experiment branches. |

The factory's pre-run guard refuses any operation whose diff modifies a Tier-0 or Tier-1 path.

## 3. Data pipeline

### 3.1 Input schema (`data/raw/*.parquet`)

```
text: str                     # cleaned, Latinized Ottoman text
source: str                   # document ID
era: str                      # 'classical' | 'late_ottoman' | 'tanzimat'
genre: str                    # 'newspaper' | 'literary' | 'legal' | 'religious' | 'official' | 'poetry' | 'other'
language_variant: str         # 'ottoman_istanbul' (v1 only; column reserved for v2)
source_pdf: str               # original filename
extraction_method: str
extraction_confidence: float  # nullable
length_chars: int
```

### 3.2 Cleaning stages (`cli.py prepare-data`)

| Stage | Mutability | Operation |
|---|---|---|
| 1. Normalize | Locked | Unicode NFC; strip ASCII control chars; collapse whitespace; normalize line endings |
| 2. Clean | Agent-editable via `data/cleaning_rules.yaml` | Ordered regex substitutions, length filters, modern-Turkish loanword ratio filter, era routing |
| 3. Dedup | Locked | MinHash + Jaccard ≥ 0.85 |
| 4. Split | Locked | `sha256(source_pdf) % 100 < 2` → heldout (~2%), else train. Split is per-document, not per-row. |

Agent edits to `cleaning_rules.yaml` after freeze only affect the **train** split. The cleaned heldout is byte-frozen at freeze-time.

### 3.3 `cleaning_rules.yaml` schema

```yaml
version: 1
substitutions:               # ordered regex passes
  - {pattern: '^\s*\d+\s*$', replace: ''}
  - {pattern: '-\n', replace: ''}
length_filters:
  min_chars: 40
  max_chars: 4000
modern_turkish_filter:
  blacklist_file: data/modern_loanwords.txt
  max_ratio: 0.04
era_routing:
  classical:    {weight: 0.20}
  late_ottoman: {weight: 0.55}
  tanzimat:     {weight: 0.25}
```

### 3.4 Tokenizer

- **Library**: HuggingFace `tokenizers` (BPE).
- **Vocab**: 24,000.
- **Byte-fallback**: enabled.
- **Trained on**: `data/clean/train/*.parquet` only. Heldout never seen.
- **Special tokens**: `<|bos|>`, `<|eos|>`, `<|pad|>`, plus 8 reserved slots `<|reserved_0..7|>`.
- **Artifact**: `tokenizer/asena-bpe-24k.json`.

### 3.5 Freeze event (`cli.py freeze`)

1. Compute SHA-256 of `tokenizer/asena-bpe-24k.json` → write `tokenizer/FROZEN.lock`.
2. Compute SHA-256 of each `eval/heldout/text/*.parquet` and the smoke prompts/lexicon files → write `eval/heldout/FROZEN.lock`.
3. Commit lock files to `main`.
4. From this moment, every `cli.py train-sprint` invocation calls `verify_freeze_invariants()` before doing anything — hash mismatch → abort with clear error.

### 3.6 What the user provides vs what the project builds

| User | Project |
|---|---|
| `data/raw/*.parquet` (matching §3.1) | Stages 1-4 implementation |
| Initial `data/modern_loanwords.txt` seed (~200-500 words) — drafted by Claude, user reviews | Tokenizer trainer |
| Era/genre tags if available; else fallback heuristics by source-PDF name | Freeze tooling |

## 4. Training core

### 4.1 What we fork from `karpathy/autoresearch`

We start from Karpathy's `train.py` (~630 lines, the nanochat training core stripped to single-file form) and adapt it:

- **Removed**: Karpathy's `prepare.py` (its responsibilities split across our `data/`, `tokenizer/`, `eval/` modules, each independently frozen).
- **Kept**: `train.py` as the agent's primary editable surface, the single-file legibility virtue, the 5-min sprint contract.
- **Modified**: data loader points at our parquet + frozen tokenizer; arch module mildly split into `train/arch.py` for Phase-2 SFT reuse; hparams externalized to `train/configs/*.yaml`.

### 4.2 Two profiles

**Sprint** (autoresearch iteration unit):

```yaml
profile: sprint
model:
  n_layers: 6
  n_embd: 384
  n_head: 6
  n_kv_heads: 2          # GQA
  mlp_ratio: 2.67        # SwiGLU
  rope_theta: 10000
  tie_embeddings: true
  init_std: 0.02
training:
  seq_len: 1024
  batch_size: 32
  grad_accum: 1
  total_tokens: 25_000_000   # ~5 min on 4090
  lr_peak: 3e-3
  lr_schedule: cosine
  warmup_steps: 200
  weight_decay: 0.1
  betas: [0.9, 0.95]
  grad_clip: 1.0
  precision: bf16
  optimizer: adamw
data:
  mix: {classical: 0.20, late_ottoman: 0.55, tanzimat: 0.25}
eval:
  every_steps: 200
  smoke_at_end: true
```

≈ 30M params. Runs in ~5 minutes on a 4090.

**Promotion** (publishable run):

```yaml
profile: promotion
model:
  n_layers: 12
  n_embd: 640
  n_head: 10
  n_kv_heads: 4
  mlp_ratio: 2.67
  rope_theta: 10000
  tie_embeddings: false
  init_std: 0.015
training:
  seq_len: 2048
  batch_size: 16
  grad_accum: 4              # effective 64
  total_tokens: 60_000_000   # ~7 epochs over 7.97M unique tokens; ~1-2h on 4090
  lr_peak: 6e-4
  lr_schedule: cosine
  warmup_steps: 500
  weight_decay: 0.1
  betas: [0.9, 0.95]
  grad_clip: 1.0
  precision: bf16
  optimizer: adamw
data:
  mix: {classical: 0.59, late_ottoman: 0.40, tanzimat: 0.01}  # observed corpus mass after import
eval:
  every_steps: 200
  smoke_every_steps: 1000
  checkpoint_every_steps: 500
  keep_last_n_checkpoints: 5
  keep_best_n_checkpoints: 3
```

≈ 84M params. Trained ~60M tokens (~7 epochs). Severely data-constrained — the model card will be transparent about this. Heavy regularization (dropout, weight decay) and early stopping on heldout PPL are expected to dominate the recipe space the agent explores; bigger models are rejected by `factory/bounds.py` until the corpus grows past ~30M unique BPE tokens.

### 4.3 Architecture

- Decoder-only transformer
- RoPE positional embeddings
- RMSNorm
- SwiGLU MLP
- GQA (grouped-query attention)
- No bias terms

### 4.4 Sprint-to-promotion verification

Sprint accepts don't auto-promote. Before `cli.py train-promotion`:

1. Top-K (default 5) accepted recipes from sprint ledger.
2. Each runs as **mini-promotion** (~2h, depth-10, 100M tokens) for verification.
3. Lowest combined eval at mini-promotion → graduates to full 24-36h promotion.

Catches sprint-to-promotion transfer failures (recipes that win at 30M/25Mt but lose at 80M/60Mt).

## 5. Evaluation harness — the immutable contract

### 5.1 Four evaluators

Each is a pure function `(model, frozen_heldout) → float`. Lower is better for all four.

| Evaluator | Metric | What it catches |
|---|---|---|
| `heldout_ppl.py` | Bits-per-byte on frozen heldout (Karpathy's `val_bpb`) | General language modeling quality |
| `lexicon_score.py` | `-log(fraction of generated tokens in Ottoman lexicon)` | Lacks-Ottoman-flavor failures |
| `flatness.py` | Modern-Turkish loanword occurrence rate in generations | Modern-Turkish contamination |
| `smoke.py` | Fraction of fixed prompts failing deterministic pass/fail rules | Catastrophic regressions on specific outputs |

`smoke.py` uses ~50 hand-curated prompts; per-prompt rules are pure regex + token-count checks. No LLM-as-judge.

### 5.2 Policy combiner — strict no-trades

```python
# eval/policy.py — IMMUTABLE
def decide(baseline, new) -> Decision:
    deltas = {k: new[k] - baseline[k] for k in metrics}
    
    if any(deltas[k] > REGRESSION_TOLERANCE[k] for k in metrics):
        return Reject(reason=f"regression in {regressed_metric}")
    
    if all(deltas[k] >= -NOISE_FLOOR[k] for k in metrics):
        if any(deltas[k] < -IMPROVEMENT_THRESHOLD[k] for k in metrics):
            return Accept(deltas=deltas)
        return Reject(reason="no real improvement (all within noise)")
```

```python
REGRESSION_TOLERANCE = {
    "heldout_ppl_bpb": 0.005, "lexicon_score": 0.02,
    "flatness": 0.002,        "smoke_fail_rate": 0.0,
}
IMPROVEMENT_THRESHOLD = {
    "heldout_ppl_bpb": 0.015, "lexicon_score": 0.05,
    "flatness": 0.005,        "smoke_fail_rate": 0.02,
}
NOISE_FLOOR = {
    "heldout_ppl_bpb": 0.003, "lexicon_score": 0.01,
    "flatness": 0.001,        "smoke_fail_rate": 0.0,
}
```

These constants are placeholder. **Calibrated by running 5-10 identical baselines and measuring run-to-run variance during initial setup.** Noise floor must exceed observed variance.

**Key property: no trading.** A patch that improves PPL 5% but regresses lexicon 3% is **rejected**. Weighted-sum scores are how models silently degrade. Strict per-metric gates prevent this.

### 5.3 Baseline pointer

`factory/db.py` maintains `current_baseline = (experiment_id, scores, git_sha)`. Every comparison is vs `current_baseline`, never vs prior-experiment. Prevents micro-improvement drift from accumulating into total regression.

### 5.4 Determinism

- Eval generation: `temperature=0, top_k=1` for smoke (pure greedy); `temperature=0.8, top_p=0.9` with fixed seed for lexicon/flatness.
- Model in `eval()` mode, no dropout.
- `torch.use_deterministic_algorithms(True)` where supported.
- Residual bf16 variance accepted as long as it stays below noise floor.

## 6. Factory orchestration

### 6.1 The loop is one CLI command

`cli.py train-sprint` performs the entire experiment cycle end-to-end:

```
1. Refuse if working tree has uncommitted changes in Tier-0 or Tier-1 paths.
2. Refuse if free VRAM < 20 GB.
3. Verify FROZEN.lock hashes (tokenizer + heldout).
4. Auto-create `exp/<timestamp>` branch from main with kimi's current edits committed.
5. Run 30-sec smoke; abort on NaN/Inf or import error.
6. Run 5-min sprint per train/configs/sprint.yaml.
7. Run all four evaluators on the resulting checkpoint.
8. Apply strict-no-trades policy.
9. Accept: ff-only merge to main; update baseline; insert SQLite row; delete branch.
   Reject: delete branch; insert SQLite row with reason.
10. Print outcome JSON to stdout.
```

Kimi calls this command. Nothing else drives the loop.

### 6.2 SQLite schema

```sql
CREATE TABLE experiments (
    id              INTEGER PRIMARY KEY,
    started_utc     TEXT NOT NULL,
    finished_utc    TEXT,
    git_sha_before  TEXT NOT NULL,
    git_sha_after   TEXT,
    branch_name     TEXT NOT NULL,
    scope           TEXT,                    -- inferred from diff if kimi didn't tag
    hypothesis      TEXT,                    -- from commit message
    diff            TEXT NOT NULL,           -- the patch itself, for resurrection
    outcome         TEXT NOT NULL,           -- 'accept' | 'reject_validation' | 'reject_smoke' | 'reject_eval' | 'error'
    reject_reason   TEXT,
    delta_ppl_bpb   REAL, delta_lexicon  REAL, delta_flatness REAL, delta_smoke REAL,
    score_ppl_bpb   REAL, score_lexicon  REAL, score_flatness REAL, score_smoke REAL,
    train_tokens    INTEGER,
    train_steps     INTEGER,
    train_seconds   REAL,
    peak_vram_mb    INTEGER
);
CREATE INDEX idx_outcome ON experiments(outcome);
CREATE INDEX idx_scope   ON experiments(scope);

CREATE TABLE baselines (
    id INTEGER PRIMARY KEY, experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    git_sha TEXT NOT NULL, set_utc TEXT NOT NULL,
    score_ppl_bpb REAL NOT NULL, score_lexicon REAL NOT NULL,
    score_flatness REAL NOT NULL, score_smoke REAL NOT NULL
);
-- Most recent row is the current baseline.

CREATE TABLE freeze_locks (
    component TEXT PRIMARY KEY,            -- 'tokenizer' | 'heldout'
    sha256 TEXT NOT NULL, frozen_utc TEXT NOT NULL, frozen_by TEXT NOT NULL
);
```

### 6.3 Git workflow

- `main` = published baseline. Linear history.
- `exp/<timestamp>` = per-experiment branch. Created with current edits committed.
- Accept → ff-only merge to main → branch deleted.
- Reject → branch deleted immediately; full diff preserved in SQLite `experiments.diff` column for resurrection.
- Promotion runs tag `main` with `asena-base-v0.1` (etc.) and push the tag.

No worktrees in v1 (single-threaded loop). Worktrees are Phase-2 (parallel experiments).

### 6.4 Janitor (`factory/janitor.py`)

| Artifact | Retention |
|---|---|
| Sprint checkpoints | Last 1; delete the rest immediately |
| Promotion checkpoints | Last 5 + best 3 by eval + final + tagged-baseline; rest deleted |
| Tokenizer artifacts | Permanent |
| Cleaned corpus | Permanent (heldout half is frozen-by-contract) |
| Raw corpus | Permanent |
| SQLite ledger | Permanent |
| Old experiment branches | Deleted on accept (merged) or reject |
| Per-run logs | Compressed; retained 30 days |

Hard floor: if `df` reports < 20 GB free, orchestrator pauses with a warning. Better to halt than OOM mid-training.

### 6.5 Failure modes

| Failure | Behavior |
|---|---|
| Freeze hash mismatch | Halt session, alert |
| VRAM busy (VLLM running) | Refuse to start; clear error with PID to kill |
| Patch apply fails | Log as `error`, continue |
| Smoke NaN/Inf | Log as `reject_smoke`, continue |
| Sprint OOM at runtime | Log as `reject_smoke` (pre-check should have prevented), continue |
| Eval crash | Halt session, alert |
| Git merge fails on accept | Halt session, alert (indicates state corruption) |
| Disk fills mid-run | Janitor floor should prevent; if happens, halt + alert |
| Kimi subprocess timeout | Log as `error`, continue |

Principle: **routine failures → log + continue; state-corruption failures → halt + alert.**

## 7. Agent (kimi) integration

### 7.1 The mental model

Kimi is the loop. The factory is one CLI command kimi calls (`cli.py train-sprint`). No Python orchestrator drives kimi; kimi drives itself.

### 7.2 Invocation

User launches one long-lived session:

```bash
cli.py autoresearch-run --duration 8h
# under the hood:
kimi --yolo -p "$(cat agent/prompts/run-autoresearch.md)"
```

`agent/prompts/run-autoresearch.md` (~30 lines, paraphrased): *Read README.md and SAFETY.md. Goal: improve the current baseline. Each cycle: read ledger (`cli.py ledger tail 20`) and baseline (`cli.py baseline show`), decide what to try, edit files in `train/`, run `cli.py train-sprint`, read outcome, iterate. Aim for 50+ accepts over this session. You may search the web. You may consult `claude` or `codex` when you judge it'd help. Never touch Tier-0 or Tier-1 paths.*

### 7.3 Kimi's native toolset (no wrappers)

Kimi-CLI ships with: file r/w/edit, shell execution, web search/fetch, MCP. We provide ~5 CLI subcommands on top: `train-sprint`, `ledger`, `baseline`, optional `request-advisor`. Kimi figures out the rest.

### 7.4 What's enforced — the only hard rules

1. **Protected-paths guard.** Pre-run scan: refuse if diff touches `eval/`, `tokenizer/`, `factory/`, `cli.py`, `SAFETY.md`, `README.md`, `data/modern_loanwords.txt`, `agent/prompts/`, or any `*FROZEN.lock`. ~5 lines of code.
2. **Smoke test.** 30-second pre-sprint check; rejects NaN/Inf/import-error patches before burning 5 minutes.
3. **Forbidden imports/patterns.** Refuse patches that import or define: `torch.distributed`, `MoE`/`MixtureOfExperts`, `mamba`, `s4`, `hyena`, encoder modules.
4. **Bounds.** Param count: sprint 20-80M, promotion 60-130M (post-second-amendment; tightened upper bound after the BPE token budget was known). Wall clock: sprint ≤ 6 min, promotion ≤ 4h. Peak VRAM: ≤ 22 GB. Estimated pre-flight, hard rejected if exceeded.
5. **Deterministic eval policy.** Section 5.

### 7.5 What's not enforced (deferred — see §10)

No proposal schema, no forced context summary, no mandatory memos, no calibration tracking, no anti-fixation rule, no ideas.yaml whitelist, no auto-triggered advisor reviews. The eval is the entire safety net. If kimi proposes 100 bad ideas overnight, we burn ~8h GPU and produce zero broken models — the eval rejects them all.

### 7.6 Honest trade-off statement

We trust: (a) the eval is correct (we own — §5), (b) kimi has decent baseline judgment (Moonshot owns), (c) the protected-paths guard holds (~5 lines, we own). We do not trust kimi to never hallucinate; we let the eval catch hallucinations cheaply. If we observe specific failure modes in practice, we add specific mitigations — not before.

## 8. CLI surface

| Command | Audience | What it does |
|---|---|---|
| `cli.py prepare-data` | User | Raw → cleaned parquet via Stages 1-4 |
| `cli.py train-tokenizer` | User | Trains BPE-24k on train split |
| `cli.py freeze` | User | Writes FROZEN.lock files for tokenizer + heldout; one-time |
| `cli.py train-sprint` | Kimi | The autoresearch one-shot (§6.1) |
| `cli.py train-promotion` | User | 24-36h real training run |
| `cli.py eval [--checkpoint PATH]` | User | Run 4 evaluators against any checkpoint |
| `cli.py autoresearch-run [--duration 8h]` | User | Launches kimi session with the autoresearch prompt |
| `cli.py ledger {tail N \| query --scope X --outcome Y}` | Kimi + user | SQLite ledger access |
| `cli.py baseline show` | Kimi + user | Current baseline summary |
| `cli.py export-gguf --checkpoint PATH [--quant q8_0]` | User | safetensors → GGUF for ollama/llama.cpp |
| `cli.py unfreeze --i-know-what-im-doing --clear-ledger` | User | Destructive; resets Phase 1 |

## 9. Phase 1 Definition of Done

Phase 1 is complete when **all** are true:

1. `cli.py prepare-data && cli.py train-tokenizer && cli.py freeze` runs clean. `FROZEN.lock` files committed to git.
2. `cli.py eval --checkpoint <randomly-initialized model>` produces all four metrics. (Eval works before any training.)
3. `cli.py train-sprint` runs end-to-end, accepts or rejects deterministically, updates ledger.
4. `cli.py autoresearch-run --duration 8h` ran ≥1 overnight session unattended → SQLite has ≥50 rows → ≥3 accepts and ≥3 rejects → no unexplained `error` outcomes.
5. `cli.py train-promotion` ran ~24-36h → `checkpoints/asena-base-v0.1/` exists with `model.safetensors`, `tokenizer.json`, `config.json`, `generation_samples.txt`, `eval_report.md`.
6. `cli.py export-gguf` produces a working GGUF → `ollama run <path>` generates plausible Ottoman text on a smoke prompt.
7. `README.md` and `SAFETY.md` exist and reflect current behavior.
8. `pytest tests/` passes: freeze-invariant check, policy combiner unit tests, smoke training step on dummy model, eval determinism check.

## 10. Phase 2 deferrals (named, so we don't forget)

| Deferred | Trigger to revisit |
|---|---|
| SFT lap and specialist heads (`asena-translator`, `-normalizer`, `-completion`) | After `asena-base-v0.1` ships |
| Kimi proposes new ideas / reads arxiv | After 1 month of Phase-1 loop operation |
| Calibration tracking | When kimi hit-rate visibly drifts |
| Anti-fixation 3-strikes rule | When ledger shows kimi looping on one scope |
| `ideas.yaml` whitelist | If hallucinated proposals waste >2h GPU/night |
| Mandatory memos | If `git log` proves insufficient as audit trail |
| Auto-triggered advisor council | If structural changes keep regressing |
| Parallel experiments via worktrees | When single-threaded throughput is the bottleneck |
| Multi-variety models (Crimean / Nogay / Chagatai) | After `asena-base` stable; each gets own repo + tokenizer |
| Bigger model sizes (>130M) | After corpus expands beyond ~30M unique BPE tokens (Phase-1.5 amendment) |
| DPO / RLHF | Out of scope indefinitely; SFT first |

## 11. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | Corpus quality dominates outcome | Strong Stage-2 cleaning + modern-loanword filter + dedup; user curates eligibility |
| 2 | Eval gameability | Multi-metric strict-no-trades; frozen heldout + tokenizer with hash checks |
| 3 | VLLM holds 22GB VRAM | Pre-flight VRAM check; clear error with PID to kill |
| 4 | Disk pressure (80GB free) | Janitor retention + 20GB hard floor + pause-and-warn |
| 5 | Sprint-to-promotion transfer failure | Mini-promotion verification before full promotion (§4.4) |
| 6 | Tokenizer drift invalidates prior experiments | Freeze with hash checks; unfreeze is destructive on purpose |
| 7 | Kimi hallucinates papers/techniques | Eval rejects bad ideas cheaply; protected-paths guard limits blast radius |
| 8 | Register mixing (classical/late/Tanzimat) | Eval targets late-Tanzimat; data mix is agent-tunable; smoke prompts surface confusion |
| 9 | Capability misalignment ("does it chat?") | Model card is explicit: ~80M base LM, not chatbot |
| 10 | Corpus assembly delay | Decoupled: Phase 1 can be implemented against the first 100MB of corpus; remaining ingest in parallel |
| 11 | Git state corruption from concurrent edits | Single-threaded loop in v1; parallelism is Phase 2 |
| 12 | bf16 nondeterminism | `torch.use_deterministic_algorithms(True)`; noise-floor in policy accepts residual variance |

## 12. `SAFETY.md` content (terse rules)

```
# SAFETY — ottoman-autoresearch

## Protected paths (Tier 0 + 1)
The factory's pre-run guard refuses operations modifying:
  eval/**, tokenizer/asena-bpe-24k.json, factory/**, **/FROZEN.lock,
  SAFETY.md, cli.py, README.md, data/modern_loanwords.txt, agent/prompts/**

Unfreezing is destructive — invalidates every prior experiment.

## Forbidden architectural changes (v1)
Patches importing/defining are rejected:
  torch.distributed, MoE/MixtureOfExperts, mamba, s4, hyena, encoder modules.

## Bounds enforced pre-GPU
  param count: sprint 20-80M, promotion 60-130M
  wall clock:  sprint ≤ 6 min, promotion ≤ 48h
  peak VRAM:   ≤ 22 GB

## Hard halts (require human)
  FROZEN.lock hash mismatch, eval crash, disk < 20 GB free,
  git merge failure on accept.

## Routine failures (log + continue)
  smoke NaN/Inf, sprint OOM, patch validation failure, kimi timeout.

## What kimi may NEVER do
  Edit any protected path, bypass `cli.py train-sprint`, modify SAFETY.md,
  disable freeze invariant, use --no-verify on commits, force-push to main,
  delete branches not its own.
```

## 13. Roadmap (rough)

| Week | Deliverable |
|---|---|
| 1 | Data pipeline + tokenizer trained + freeze; factory skeleton; first baseline (depth-6) trains; eval works. |
| 2 | `cli.py autoresearch-run` operational; first overnight session; ledger populated; mid-week recipe analysis. |
| 3 | Promotion run (24-36h) → `asena-base-v0.1` produced; eval report; model card draft. |
| 4 | GGUF export; HF publication; spec for Phase-2 SFT lap begins. |

Corpus assembly is decoupled — user expects extraction completion to align with Phase-1 implementation completion (~week 1-2).

## 14. License (Phase 1 default — reviewable)

- **Code**: Apache-2.0
- **Model weights**: Apache-2.0 (matching code; no fine-tune origin to worry about)
- **Dataset**: CC-BY-4.0
- User retains right to override before publication.

## 15. Open questions (for user review)

1. **License defaults** above — confirm or override.
2. **Noise-floor thresholds** in §5.2 are placeholders; calibrated empirically during Week 1 by running 5-10 identical baselines and measuring variance. Spec records the policy; specific numbers finalize at calibration time.
3. **Initial seed list for `data/modern_loanwords.txt`** — Claude drafts ~300 entries; user reviews.
4. **Phase-1 DoD #4** (≥50 ledger rows, ≥3 accepts and ≥3 rejects) — confirm thresholds.
5. **Era weight defaults** (20/55/25 for classical/late_ottoman/tanzimat in §3.3) — placeholder; revisit when corpus stats are known.

## 16. Appendix A — initial idea seed (Phase 1 reference, non-binding)

Claude will draft an `agent/ideas_seed.md` at implementation time with ~20-25 well-known, small-LM, single-GPU, well-validated improvements drawn from established literature, for kimi to reference when picking what to try. Examples (non-exhaustive):

- Optimizer variants: Muon, Lion, schedule-free AdamW
- LR schedules: cosine, WSD (warmup-stable-decay), trapezoidal
- Architecture knobs: GQA group size sweep, MLP ratio (2.67 / 3.5 / 4), RoPE θ (10k / 50k / 500k), pre-norm vs sandwich-norm, tied vs untied embeddings
- Regularization: z-loss, attention dropout schedules
- Data-mix sweeps: era ratios, length-bucket weighting
- Cleaning-rule improvements: stricter modern-loanword cutoff, dedup threshold tuning

Phase-1 contract: kimi is **not** required to pick from this list — it's reference material. Kimi may research and propose freely. The eval is the authority on what's accepted.

## 17. References

- Karpathy, A. (2026). [autoresearch](https://github.com/karpathy/autoresearch). The 630-line single-GPU autoresearch repo we fork.
- Zhang, A., Kraska, T., Khattab, O. (2025). [Recursive Language Models](https://arxiv.org/abs/2512.24601). Cited for the lesson — operate on structured inputs, not text dumps — but we do not adopt the full REPL machinery in Phase 1.
- MoonshotAI. [kimi-cli](https://github.com/MoonshotAI/kimi-cli). The researcher CLI; built-in file/shell/web tools used directly.
