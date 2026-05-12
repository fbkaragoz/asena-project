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


if __name__ == "__main__":
    cli()
