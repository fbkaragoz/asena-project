# asena-project — autoresearch pipeline for `cdli/asena`

Train `cdli/asena-base`, a 150-300M-parameter decoder-only language model for
Latinized Ottoman Turkish, **from scratch** on a single RTX 4090, via an
evaluator-driven autoresearch loop where kimi is the researcher and a
deterministic factory is the judge.

## Quickstart

```bash
# 1. Install deps
pip install -e ".[dev]"

# 2. Place your raw parquet files in data/raw/ (schema in docs)
# 3. Prepare data + train tokenizer + freeze
python cli.py prepare-data
python cli.py train-tokenizer
python cli.py freeze

# 4. Run the autoresearch loop (kimi-driven)
python cli.py autoresearch-run --duration 8h

# 5. Once you're happy with the recipe, do a promotion run
python cli.py train-promotion

# 6. Export to GGUF for ollama
python cli.py export-gguf --checkpoint checkpoints/asena-base-v0.1 --out asena-base-v0.1.q8_0.gguf
```

## Repository structure

See `docs/superpowers/specs/2026-05-12-ottoman-autoresearch-phase1-design.md` (§2)
for the authoritative layout and ownership tiers.

## Hardware requirements

- RTX 4090 (24 GB VRAM) or equivalent
- 64 GB RAM recommended
- 100+ GB free disk for promotion runs
- CUDA 12.x, PyTorch 2.3+

## How the autoresearch loop works

Kimi runs as a long-lived `kimi --yolo` session. Each cycle: it reads the
ledger, decides what to try, edits files in `train/`, runs `cli.py train-sprint`,
reads the outcome JSON, and iterates. The factory enforces immutability of the
evaluator and tokenizer; the eval policy decides accept/reject by strict
no-trades multi-metric comparison.

## License

- Code: Apache-2.0
- Model weights: Apache-2.0
- Dataset: CC-BY-4.0
