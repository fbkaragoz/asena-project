"""asena-project command-line interface."""
from __future__ import annotations
from pathlib import Path
import click


@click.group()
def cli():
    """asena-project — autoresearch pipeline for cdli/asena."""


@cli.command("prepare-data")
@click.option("--raw-dir", type=click.Path(exists=True, path_type=Path), default=Path("data/raw"))
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("data/clean"))
@click.option("--rules", type=click.Path(exists=True, path_type=Path), default=Path("data/cleaning_rules.yaml"))
@click.option("--heldout-pct", type=int, default=2)
def prepare_data(raw_dir, out_dir, rules, heldout_pct):
    """Run Stages 1-4 over raw_dir → write clean train/heldout to out_dir."""
    from data.pipeline import run_prepare_data
    summary = run_prepare_data(raw_dir=raw_dir, out_dir=out_dir, rules_path=rules, heldout_pct=heldout_pct)
    click.echo(f"prepare-data: {summary}")


@cli.command("train-tokenizer")
@click.option("--train-glob", type=str, default="data/clean/train/*.parquet")
@click.option("--out", type=click.Path(path_type=Path), default=Path("tokenizer/asena-bpe-24k.json"))
@click.option("--vocab-size", type=int, default=24000)
def train_tokenizer_cmd(train_glob, out, vocab_size):
    """Train the asena BPE tokenizer on the cleaned train split."""
    from tokenizer.train_bpe import train_bpe
    train_bpe(train_glob=train_glob, out_path=out, vocab_size=vocab_size)
    click.echo(f"train-tokenizer: wrote {out}")


TOKENIZER_PATH = Path("tokenizer/asena-bpe-24k.json")
TOKENIZER_LOCK = Path("tokenizer/FROZEN.lock")
HELDOUT_DIR = Path("data/clean/heldout")
HELDOUT_LOCK = Path("eval/heldout/FROZEN.lock")
EVAL_HELDOUT_DIR = Path("eval/heldout")


@cli.command("freeze")
def freeze_cmd():
    """Lock the tokenizer and the cleaned heldout corpus.

    From here forward, every cli.py train-sprint verifies these hashes.
    Unfreezing requires `cli.py unfreeze --i-know-what-im-doing --clear-ledger`.
    """
    from factory.guards import write_freeze_lock
    if not TOKENIZER_PATH.exists():
        raise click.ClickException(f"missing tokenizer: {TOKENIZER_PATH}. Run train-tokenizer first.")
    write_freeze_lock(TOKENIZER_LOCK, {"tokenizer.json": TOKENIZER_PATH}, frozen_by="cli-freeze")

    # Copy/symlink heldout parquet files into eval/heldout/text/ and lock there.
    eval_text = EVAL_HELDOUT_DIR / "text"
    eval_text.mkdir(parents=True, exist_ok=True)
    heldout_files = sorted(HELDOUT_DIR.glob("*.parquet"))
    if not heldout_files:
        raise click.ClickException(f"no heldout parquet in {HELDOUT_DIR}. Run prepare-data with --heldout-pct > 0.")
    import shutil
    files = {}
    for src in heldout_files:
        dst = eval_text / src.name
        if not dst.exists():
            shutil.copy(src, dst)
        files[f"text/{src.name}"] = dst
    write_freeze_lock(HELDOUT_LOCK, files, frozen_by="cli-freeze")
    click.echo(f"freeze: locked {TOKENIZER_LOCK} and {HELDOUT_LOCK}")


@cli.command("unfreeze")
@click.option("--i-know-what-im-doing", is_flag=True, required=True)
@click.option("--clear-ledger", is_flag=True, required=True)
def unfreeze_cmd(i_know_what_im_doing, clear_ledger):
    """Destructive: delete FROZEN.lock files. Invalidates all prior experiments."""
    if not (i_know_what_im_doing and clear_ledger):
        raise click.ClickException("refusing without explicit destructive flags")
    for lock in (TOKENIZER_LOCK, HELDOUT_LOCK):
        if lock.exists():
            lock.unlink()
            click.echo(f"unfreeze: removed {lock}")
    if Path("experiments.sqlite").exists():
        Path("experiments.sqlite").unlink()
        click.echo("unfreeze: removed experiments.sqlite")


@cli.command("train-sprint")
def train_sprint_cmd():
    """Run one autoresearch sprint cycle end-to-end.

    Pre-flight → smoke (30s) → sprint (~5min) → eval → accept/reject.
    Prints outcome JSON to stdout for kimi to parse.
    """
    from factory.orchestrator import run_train_sprint
    import json
    result = run_train_sprint()
    click.echo(json.dumps(result, indent=2, default=str))


@cli.group("ledger")
def ledger_cmd():
    """Query the experiment ledger."""


@ledger_cmd.command("tail")
@click.argument("n", type=int, default=20)
def ledger_tail(n):
    """Show last N experiments."""
    import json
    from factory.db import Ledger
    rows = Ledger(Path("experiments.sqlite")).list_experiments(limit=n)
    click.echo(json.dumps(rows, indent=2, default=str))


@ledger_cmd.command("query")
@click.option("--scope", type=str, default=None)
@click.option("--outcome", type=str, default=None)
@click.option("--limit", type=int, default=50)
def ledger_query(scope, outcome, limit):
    """Filtered ledger query."""
    import json
    from factory.db import Ledger
    rows = Ledger(Path("experiments.sqlite")).query(scope=scope, outcome=outcome, limit=limit)
    click.echo(json.dumps(rows, indent=2, default=str))


@cli.group("baseline")
def baseline_cmd():
    """Current baseline."""


@baseline_cmd.command("show")
def baseline_show():
    """Print current baseline JSON."""
    import json
    from factory.db import Ledger
    b = Ledger(Path("experiments.sqlite")).get_baseline()
    click.echo(json.dumps(b, indent=2, default=str) if b else "(no baseline yet)")


@cli.command("autoresearch-run")
@click.option("--duration", type=str, default="8h", help="Session duration (informational only — kimi loops until killed).")
def autoresearch_run_cmd(duration):
    """Launch a kimi session pointed at agent/prompts/run-autoresearch.md.

    Requires `kimi` CLI installed (https://github.com/MoonshotAI/kimi-cli).
    """
    import subprocess
    prompt_path = Path("agent/prompts/run-autoresearch.md")
    if not prompt_path.exists():
        raise click.ClickException(f"missing {prompt_path}")
    prompt = prompt_path.read_text()
    click.echo(f"autoresearch-run: starting kimi session (intended duration: {duration})")
    try:
        subprocess.run(["kimi", "--yolo", "-p", prompt], check=False)
    except FileNotFoundError:
        raise click.ClickException("kimi CLI not found in PATH; install: curl -L code.kimi.com/install.sh | bash")


@cli.command("train-promotion")
@click.option("--config", type=click.Path(path_type=Path, exists=True),
              default=Path("train/configs/promotion.yaml"))
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("checkpoints/asena-base-v0.1"))
def train_promotion_cmd(config, out):
    """Long-running promotion training run (~24-36h on RTX 4090).

    Uses the current ACCEPTED baseline architecture/recipe, scaled to promotion size.
    Saves model + tokenizer + config + sample generations to `out`.
    """
    from train.train import run_training
    out.mkdir(parents=True, exist_ok=True)
    result = run_training(
        config_path=config,
        tokenizer_path=Path("tokenizer/asena-bpe-24k.json"),
        train_glob="data/clean/train/*.parquet",
        checkpoint_out=out / "model.pt",
        device="cuda",
    )
    import shutil
    shutil.copy("tokenizer/asena-bpe-24k.json", out / "tokenizer.json")
    (out / "eval_report.md").write_text(
        f"# Promotion result\n\nFinal loss: {result['final_loss']:.4f}\n"
        f"Wall time: {result['wall_seconds']:.1f}s\n"
        f"Tokens seen: {result['tokens_seen']}\n"
    )
    click.echo(f"train-promotion: wrote {out}/")


@cli.command("eval")
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--tokenizer", type=click.Path(exists=True, path_type=Path),
              default=Path("tokenizer/asena-bpe-24k.json"))
def eval_cmd(checkpoint, tokenizer):
    """Run all four evaluators against an arbitrary checkpoint; print Scores."""
    import json
    from eval.heldout_ppl import compute_heldout_bpb
    from eval.lexicon_score import compute_lexicon_score
    from eval.flatness import compute_flatness
    from eval.smoke import evaluate_smoke_prompts

    bpb = compute_heldout_bpb(
        checkpoint_path=checkpoint, tokenizer_path=tokenizer,
        heldout_glob="eval/heldout/text/*.parquet",
    )
    fail_rate, results = evaluate_smoke_prompts(
        checkpoint_path=checkpoint, tokenizer_path=tokenizer,
        prompts_path=Path("eval/heldout/smoke_prompts.yaml"),
        blacklist_path=Path("data/modern_loanwords.txt"),
    )
    gens = [r.generation for r in results]
    lex = compute_lexicon_score(gens, lexicon_path=Path("eval/heldout/ottoman_lexicon.txt"))
    flat = compute_flatness(gens, blacklist_path=Path("data/modern_loanwords.txt"))
    click.echo(json.dumps({
        "ppl_bpb": bpb, "lexicon": lex, "flatness": flat, "smoke": fail_rate,
        "smoke_results": [r.__dict__ for r in results],
    }, indent=2))


@cli.command("export-gguf")
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--quant", type=str, default="q8_0")
def export_gguf_cmd(checkpoint, out, quant):
    """Convert a saved checkpoint dir -> GGUF for ollama/llama.cpp."""
    from tools.convert_to_gguf import export_to_gguf
    export_to_gguf(checkpoint_dir=checkpoint, out_path=out, quant=quant)
    click.echo(f"export-gguf: wrote {out}")


if __name__ == "__main__":
    cli()
