"""Publish the current Fuzuli baseline to HuggingFace Hub.

Reads the ledger + corpus + arch config + the released checkpoint, generates
an HF model card, stages a release folder, and (optionally) uploads it.

Pure functions are unit-tested in tests/test_publish_baseline.py; the actual
HuggingFace upload step is exercised manually with --dry-run.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from glob import glob
from pathlib import Path
from textwrap import dedent
from typing import Any

import pyarrow.parquet as pq
import yaml


class NoBaselineError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def load_baseline_metadata(sqlite_path: Path | str) -> dict[str, Any]:
    """Return the current baseline pointer's scores + git SHA + experiment id.

    Raises NoBaselineError if the ledger has no baseline set.
    """
    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT * FROM baselines ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise NoBaselineError(
            f"no baseline set in {sqlite_path}; run a sprint to establish one"
        )
    return {
        "experiment_id": row["experiment_id"],
        "git_sha": row["git_sha"],
        "score_ppl_bpb": row["score_ppl_bpb"],
        "score_lexicon": row["score_lexicon"],
        "score_flatness": row["score_flatness"],
        "score_smoke": row["score_smoke"],
        "set_utc": row["set_utc"],
    }


def load_corpus_stats(train_glob: str, tokenizer_path: str | Path | None = None) -> dict[str, Any]:
    """Count rows, chars, and (optionally) unique BPE tokens across train shards."""
    files = sorted(glob(train_glob))
    rows = 0
    chars = 0
    tokens = 0
    tokenizer = None
    if tokenizer_path is not None:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    for f in files:
        tbl = pq.read_table(f)
        rows += tbl.num_rows
        chars += sum(tbl.column("length_chars").to_pylist())
        if tokenizer is not None:
            for text in tbl.column("text").to_pylist():
                tokens += len(tokenizer.encode(text).ids)

    stats: dict[str, Any] = {"rows": rows, "chars": chars, "files": len(files)}
    if tokenizer is not None:
        stats["tokens"] = tokens
    return stats


def load_eval_full(checkpoint_dir: Path | str) -> dict[str, Any] | None:
    """Read eval-full.json next to the checkpoint, if present.

    Returns scores remapped to the same `score_*` keys used by the SQLite
    baseline row, plus the raw smoke_results list. Returns None if the file
    is absent — caller should fall back to the ledger baseline.
    """
    p = Path(checkpoint_dir) / "eval-full.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    return {
        "score_ppl_bpb": raw["ppl_bpb"],
        "score_lexicon": raw["lexicon"],
        "score_flatness": raw["flatness"],
        "score_smoke": raw["smoke"],
        "smoke_results": raw.get("smoke_results", []),
    }


def load_arch_config(config_path: Path | str) -> dict[str, Any]:
    """Flatten the promotion-config model + training dicts into a single arch dict."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    out: dict[str, Any] = {}
    out.update(cfg.get("model", {}))
    out.update({k: v for k, v in cfg.get("training", {}).items()
                if k in ("total_tokens", "lr_peak", "weight_decay", "seq_len",
                         "batch_size", "warmup_steps")})
    return out


def _format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{round(n / 1e6)}M"
    return str(n)


def _repo_id_from_meta(meta: dict[str, Any]) -> str:
    return meta.get("repo_id") or f"unknown/{meta['model_name']}"


def generate_model_card(meta: dict[str, Any]) -> str:
    """Render an HF model-card README.md (YAML frontmatter + body)."""
    arxiv_id = meta.get("arxiv_id")
    samples = meta.get("samples")
    baseline = meta["baseline"]
    corpus = meta["corpus"]
    arch = meta["arch"]

    frontmatter_dict = {
        "language": ["ota"],
        "license": meta["license"],
        "library_name": "pytorch",
        "tags": [
            "ottoman-turkish", "historical-nlp", "causal-lm",
            "small-lm", "low-resource", "from-scratch",
        ],
        "datasets": meta["datasets"],
        "pipeline_tag": "text-generation",
    }
    frontmatter = "---\n" + yaml.safe_dump(
        frontmatter_dict, sort_keys=False, allow_unicode=True
    ) + "---\n"

    arxiv_block = ""
    if arxiv_id:
        arxiv_block = (
            f"\n**Paper:** [arXiv:{arxiv_id}](https://arxiv.org/abs/{arxiv_id})\n"
        )

    scores_md = (
        "| Metric | Value |\n"
        "|---|---|\n"
        f"| Heldout perplexity (bits/byte) | **{baseline['score_ppl_bpb']:.4f}** |\n"
        f"| Ottoman lexicon score (lower = more Ottoman flavor) | **{baseline['score_lexicon']:.4f}** |\n"
        f"| Modern-Turkish flatness (loanword rate) | **{baseline['score_flatness']:.4f}** |\n"
        f"| Smoke fail rate (5 prompts) | **{baseline['score_smoke']:.2f}** |\n"
    )

    arch_md = (
        "| Component | Choice |\n"
        "|---|---|\n"
        f"| Layers | {arch['n_layers']} |\n"
        f"| Hidden dim | {arch['n_embd']} |\n"
        f"| Attention heads (Q) | {arch['n_head']} |\n"
        f"| KV heads (GQA) | {arch['n_kv_heads']} |\n"
        f"| MLP | SwiGLU, mlp_ratio {arch['mlp_ratio']} |\n"
        "| Positional embedding | RoPE |\n"
        "| Normalization | RMSNorm, no bias |\n"
        "| Tokenizer | BPE, vocab 24,000 |\n"
        f"| Embedding tying | {arch['tie_embeddings']} |\n"
        f"| Total parameters | **~{_format_params(meta['params'])}** |\n"
    )

    corpus_md = (
        "| | |\n"
        "|---|---|\n"
        f"| Documents | {corpus['documents']} |\n"
        f"| Pages (post-cleaning) | {corpus['rows']:,} |\n"
        f"| Characters | {corpus['chars']:,} |\n"
        f"| Unique BPE tokens | **{corpus.get('tokens', 0):,}** |\n"
    )

    samples_md = ""
    if samples:
        lines = ["## Sample generations\n"]
        for s in samples:
            if isinstance(s, dict):
                prompt_id = s.get("prompt_id", "")
                completion = s.get("generation", "")
                passed = s.get("passed", True)
                reason = s.get("reason", "")
                verdict = ("✓ accepted" if passed
                           else f"✗ rejected — {reason}" if reason
                           else "✗ rejected")
                lines.append(f"**Prompt id:** `{prompt_id}`  ")
                lines.append(f"**Completion:** {completion}  ")
                lines.append(f"**Verdict:** {verdict}\n")
            else:
                prompt, completion = s
                lines.append(f"**Prompt:** `{prompt}`  ")
                lines.append(f"**Completion:** {completion}\n")
        samples_md = "\n".join(lines) + "\n"

    sha_short = meta["git_sha"][:8] if meta.get("git_sha") else "unknown"
    arxiv_note = f"arXiv:{arxiv_id}" if arxiv_id else "HuggingFace Hub release"
    citation = (
        "```bibtex\n"
        "@misc{karagoz2026fuzuli,\n"
        f"  title  = {{Fuzuli {meta['version']} — A from-scratch Latinized Ottoman Turkish language model}},\n"
        "  author = {Karagöz, Fatih Burak},\n"
        "  year   = {2026},\n"
        f"  note   = {{CDLI technical report; {arxiv_note}}},\n"
        f"  url    = {{https://huggingface.co/{_repo_id_from_meta(meta)}}}\n"
        "}\n"
        "```\n"
    )

    title = meta["model_name"].title().replace("-", " ")
    repo_id = _repo_id_from_meta(meta)
    body = (
        f"# {title} {meta['version']}\n"
        "## A from-scratch decoder-only language model for Latinized Ottoman Turkish\n\n"
        f"**{meta['author']}** · {meta.get('affiliation', 'Independent')}{arxiv_block}\n\n"
        f"Fuzuli is a small (~{_format_params(meta['params'])} parameter), decoder-only "
        "language model trained **from scratch** on a curated corpus of Latinized Ottoman "
        "Turkish texts. No foreign weights were inherited; no model was fine-tuned. "
        "Tokenizer, training data, and weights are all produced end-to-end in the project.\n\n"
        "This is a **research artifact**, not a production tool. See the limitations "
        "section before using it.\n\n"
        "## Evaluation scores\n\n"
        "Frozen heldout split, four-evaluator harness, strict-no-trades policy.\n\n"
        f"{scores_md}\n"
        f"Baseline checkpoint git SHA: `{sha_short}`\n\n"
        "## Architecture\n\n"
        "Standard post-2023 dense decoder transformer (RoPE + GQA + SwiGLU + RMSNorm).\n\n"
        f"{arch_md}\n"
        "## Training corpus\n\n"
        f"{corpus_md}\n"
        "Composed of: Evliya Çelebi *Seyahatnâme* (7 books), 25-volume *Sebilürreşad* / "
        "*Sırat-ı Müstakim* late-Ottoman periodicals, and 25 classical / late-classical "
        "works (Dîvâns, Mecmuas, historiographies). All Latinized; sourced from the two "
        "HuggingFace datasets listed in the frontmatter.\n\n"
        f"{samples_md}"
        "## Limitations\n\n"
        f"Fuzuli {meta['version']} is trained at roughly **200× under Chinchilla-optimal** "
        "(8 M unique tokens for 84 M parameters; Chinchilla recommends ~1.7 B tokens). "
        "Expect:\n\n"
        "- Plausible short completions (one-sentence) in Ottoman-Turkish style.\n"
        "- Useful for perplexity scoring of historical Turkish text.\n"
        "- **Not** suitable for long-form generation (will memorize 3–5-token windows).\n"
        "- **Not** capable on Arabic-script Ottoman, modern Turkish, or conversational input.\n"
        "- **Not** for any historical-translation or paleographic task without human review.\n\n"
        "## How to use\n\n"
        "Custom architecture (see `train/arch.py` in the source repo). For now, load via:\n\n"
        "```python\n"
        "import torch\n"
        "from huggingface_hub import hf_hub_download\n\n"
        f'weights = hf_hub_download("{repo_id}", "model.pt")\n'
        'state = torch.load(weights, map_location="cpu")\n'
        "# See train/arch.py:AsenaModel for the config schema.\n"
        "```\n\n"
        "A `transformers`-compatible loader is planned for v0.5.\n\n"
        "## Citation\n\n"
        f"{citation}\n"
        "## License\n\n"
        "Apache-2.0 (code + weights). Training data is CC-BY-4.0; the article is CC-BY-4.0.\n"
    )

    return frontmatter + body


# ---------------------------------------------------------------------------
# Release folder staging
# ---------------------------------------------------------------------------

def stage_release_folder(
    checkpoint_dir: Path,
    model_card: str,
    out_dir: Path,
    article_dir: Path | None = None,
) -> Path:
    """Assemble files to push: model.pt, tokenizer.json, README.md, optionally article/."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ("model.pt", "tokenizer.json"):
        src = Path(checkpoint_dir) / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    (out_dir / "README.md").write_text(model_card)

    if article_dir is not None:
        art_out = out_dir / "article"
        if art_out.exists():
            shutil.rmtree(art_out)
        shutil.copytree(article_dir, art_out)

    return out_dir


# ---------------------------------------------------------------------------
# HF upload (side-effect; not unit-tested)
# ---------------------------------------------------------------------------

def upload_to_hf(folder: Path, repo_id: str, public: bool = False,
                 commit_message: str = "Initial Fuzuli release") -> str:
    """Create the repo if needed, then upload the folder. Returns the repo URL."""
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(repo_id=repo_id, private=not public, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(folder), repo_id=repo_id, repo_type="model",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Publish the current Fuzuli baseline to HuggingFace.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--repo-id", required=True, type=str)
    ap.add_argument("--version", type=str, default="v0.1")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--include-article", action="store_true")
    ap.add_argument("--arxiv-id", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ledger", type=Path, default=Path("experiments.sqlite"))
    ap.add_argument("--promotion-config", type=Path, default=Path("train/configs/promotion.yaml"))
    ap.add_argument("--train-glob", type=str, default="data/clean/train/*.parquet")
    ap.add_argument("--tokenizer", type=Path, default=Path("tokenizer/asena-bpe-24k.json"))
    args = ap.parse_args()

    baseline = load_baseline_metadata(args.ledger)
    corpus = load_corpus_stats(args.train_glob, tokenizer_path=args.tokenizer)
    arch = load_arch_config(args.promotion_config)

    # Prefer scores measured against the actual checkpoint over the sprint
    # baseline pointer in the ledger (which reflects the 30M proxy, not the
    # 84M promotion model). Smoke results become the sample generations.
    eval_full = load_eval_full(args.checkpoint)
    samples = None
    if eval_full is not None:
        for k in ("score_ppl_bpb", "score_lexicon", "score_flatness", "score_smoke"):
            baseline[k] = eval_full[k]
        samples = eval_full["smoke_results"] or None

    from factory.bounds import estimate_param_count
    params = estimate_param_count(
        n_layers=arch["n_layers"], n_embd=arch["n_embd"],
        n_head=arch["n_head"], n_kv_heads=arch["n_kv_heads"],
        mlp_ratio=arch["mlp_ratio"], vocab_size=24000,
        tied=arch["tie_embeddings"],
    )

    meta = {
        "model_name": "fuzuli-base",
        "version": args.version,
        "params": params,
        "git_sha": baseline["git_sha"],
        "baseline": baseline,
        "corpus": {
            "rows": corpus["rows"], "chars": corpus["chars"],
            "tokens": corpus.get("tokens", 0),
            "documents": 59,
        },
        "arch": arch,
        "datasets": [
            "fatihburakkaragoz/anadolu-ocr-corpus",
            "fatihburakkaragoz/evliya-celebi-seyahatname-ocr",
        ],
        "license": "apache-2.0",
        "author": "Fatih Burak Karagöz",
        "affiliation": "CDLI",
        "arxiv_id": args.arxiv_id,
        "samples": samples,
        "repo_id": args.repo_id,
    }
    card = generate_model_card(meta)

    out_dir = Path(f".release/{args.repo_id.replace('/', '__')}")
    article_dir = Path("docs/article-fuzuli-v0.1") if args.include_article else None
    staged = stage_release_folder(
        checkpoint_dir=args.checkpoint, model_card=card,
        out_dir=out_dir, article_dir=article_dir,
    )
    print(f"Staged release folder: {staged.resolve()}")
    print(f"  files: {sorted(p.name for p in staged.iterdir())}")
    print(f"  README.md preview (first 600 chars):")
    print("  " + "\n  ".join(card[:600].splitlines()))

    if args.dry_run:
        print("\n--dry-run: skipping HF upload.")
        return

    url = upload_to_hf(staged, repo_id=args.repo_id, public=args.public,
                       commit_message=f"Fuzuli {args.version} release")
    print(f"\nUploaded: {url}")


if __name__ == "__main__":
    main()
