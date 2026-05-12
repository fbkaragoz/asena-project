# SAFETY — asena-project

## Protected paths (Tier 0 + 1)

The factory's pre-run guard refuses operations modifying:
- `eval/**`
- `tokenizer/asena-bpe-24k.json`
- `factory/**`
- `**/FROZEN.lock`
- `SAFETY.md`, `cli.py`, `README.md`
- `data/modern_loanwords.txt`
- `agent/prompts/**`

Unfreezing the tokenizer or heldout is destructive — invalidates every prior
experiment. Use `cli.py unfreeze --i-know-what-im-doing --clear-ledger` only
when restarting Phase 1.

## Forbidden architectural changes (v1)

The validator rejects patches importing or defining:
- `torch.distributed`
- `MoE` / `MixtureOfExperts`
- `mamba`, `s4`, `hyena` (non-attention architectures)
- encoder modules (we are decoder-only)

## Bounds enforced before GPU

- param count: sprint 10-80M, promotion 100-350M
- wall clock: sprint ≤ 6 min, promotion ≤ 48h
- peak VRAM: ≤ 22 GB

## Hard halts (require human intervention)

- FROZEN.lock hash mismatch
- Eval crash
- Disk < 20 GB free
- Git merge failure on accept (state corruption)

## Routine failures (log + continue)

- Smoke NaN/Inf
- Sprint OOM
- Patch validation failure
- Kimi subprocess timeout

## What kimi may NEVER do

- Edit any protected path (above)
- Bypass `cli.py train-sprint` for training
- Modify `SAFETY.md`
- Disable the freeze invariant check
- Use `--no-verify` on git commits
- Force-push to main
- Delete branches not its own (only `train-sprint` may delete experiment branches)
