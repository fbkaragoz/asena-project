# Phase-1 idea seed (reference, non-binding)

This is a curated list of ~25 well-known, single-GPU, small-LM training
improvements. Kimi may pick from this list OR research and propose other
ideas. The eval is the authority on what works.

## Optimizer variants

- **Muon (Jordan et al.)**: orthogonalized SGD-like updates for hidden layers;
  keep AdamW for embeddings + lm_head. Reported faster convergence at small scale.
- **Lion**: sign-based optimizer; fewer state tensors, sometimes wins on small models.
- **Schedule-free AdamW**: no warmup/cosine, single LR throughout. Test if our
  cosine schedule is leaving signal on the table.

## LR schedules

- **WSD (warmup-stable-decay)**: trapezoidal. Better for variable-length runs.
- **Cosine warmdown**: linear warmup, cosine decay to 0.1× peak then constant.
- **Higher peak LR with shorter warmup**: aggressive but sometimes wins at small scale.

## Architecture knobs

- **GQA group size**: try n_kv_heads ∈ {1, 2, 3} for n_head=6. KV memory tradeoff.
- **MLP ratio**: try {2.67, 3.5, 4.0}. Llama uses 2.67, GPT-2 uses 4.
- **RoPE theta**: try {10k, 50k, 500k}. Higher = longer-range positional info.
- **Norm placement**: pre-norm (current) vs sandwich-norm (extra norm before residual).
- **Tied embeddings**: currently true; try untied for promotion only (more params).
- **Embedding init scale**: try init_std ∈ {0.01, 0.02, 0.03}.

## Regularization

- **Z-loss**: encourages logits to stay small; can stabilize training.
- **Attention dropout schedule**: 0 → 0.1 over warmup, then back to 0.
- **Stochastic depth**: drop blocks at random during training; small networks benefit.

## Data mix sweeps

- **Era ratios**: try alternative weights (e.g., 0.10/0.65/0.25, 0.30/0.50/0.20).
- **Length bucketing**: prefer shorter sequences early in training.

## Cleaning-rule improvements

- **Stricter modern-loanword cutoff**: drop max_ratio from 0.04 → 0.02.
- **Tighter min_chars**: 40 → 80 to reject very-short fragments.
- **Higher dedup threshold**: 0.85 → 0.92 keeps more near-paraphrases.

Each entry above is a starting point. The expected effect, mechanism, and
hypothesis you state in the commit message is what makes the diff legible.
