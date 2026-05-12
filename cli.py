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


if __name__ == "__main__":
    cli()
