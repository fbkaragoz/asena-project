"""Tests for tools/publish_baseline.py — HF release pipeline for Fuzuli checkpoints.

Covers the pure-function surface: ledger lookup, corpus stats, arch config,
model-card generation, release-folder staging. The actual HF upload step
(huggingface_hub.upload_folder) is not tested here; it's a thin wrapper
exercised manually with --dry-run.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from factory.db import Ledger, ExperimentRow

from tools.publish_baseline import (
    load_baseline_metadata,
    load_corpus_stats,
    load_arch_config,
    load_eval_full,
    generate_model_card,
    stage_release_folder,
    NoBaselineError,
)


# ---------------------------------------------------------------------------
# load_baseline_metadata
# ---------------------------------------------------------------------------

def _insert_accept(ledger: Ledger, *, sha: str, scores: tuple[float, float, float, float]) -> int:
    p, l, f, s = scores
    return ledger.insert(ExperimentRow(
        started_utc="t", finished_utc="t",
        git_sha_before="prev", git_sha_after=sha, branch_name=f"exp/{sha[:8]}",
        scope=None, hypothesis="", diff="",
        outcome="accept", reject_reason=None,
        delta_ppl_bpb=0.0, delta_lexicon=0.0, delta_flatness=0.0, delta_smoke=0.0,
        score_ppl_bpb=p, score_lexicon=l, score_flatness=f, score_smoke=s,
        train_tokens=25_000_000, train_steps=400, train_seconds=76.0, peak_vram_mb=18_000,
    ))


def test_load_baseline_metadata_returns_latest_accept(tmp_path):
    sqlite = tmp_path / "experiments.sqlite"
    ledger = Ledger(sqlite)
    eid1 = _insert_accept(ledger, sha="aaa111", scores=(2.20, 10.05, 0.011, 0.4))
    ledger.set_baseline(eid1, git_sha="aaa111",
                        scores={"score_ppl_bpb": 2.20, "score_lexicon": 10.05,
                                "score_flatness": 0.011, "score_smoke": 0.4})
    eid2 = _insert_accept(ledger, sha="bbb222", scores=(2.20, 2.85, 0.000, 0.4))
    ledger.set_baseline(eid2, git_sha="bbb222",
                        scores={"score_ppl_bpb": 2.20, "score_lexicon": 2.85,
                                "score_flatness": 0.000, "score_smoke": 0.4})

    meta = load_baseline_metadata(sqlite)
    assert meta["git_sha"] == "bbb222"
    assert meta["score_lexicon"] == pytest.approx(2.85)
    assert meta["score_flatness"] == pytest.approx(0.000)
    assert meta["experiment_id"] == eid2


def test_load_baseline_metadata_raises_on_empty_ledger(tmp_path):
    sqlite = tmp_path / "experiments.sqlite"
    Ledger(sqlite)  # initializes schema, no rows
    with pytest.raises(NoBaselineError, match="no baseline"):
        load_baseline_metadata(sqlite)


# ---------------------------------------------------------------------------
# load_corpus_stats
# ---------------------------------------------------------------------------

def _write_parquet(path: Path, texts: list[str]) -> None:
    pq.write_table(pa.table({
        "text": texts,
        "length_chars": [len(t) for t in texts],
    }), path)


def test_load_corpus_stats_counts_rows_chars_and_files(tmp_path):
    d = tmp_path / "clean" / "train"
    d.mkdir(parents=True)
    _write_parquet(d / "part-00000.parquet", ["hello world", "foo bar baz"])
    _write_parquet(d / "part-00001.parquet", ["another row of text"])

    stats = load_corpus_stats(str(d / "*.parquet"))
    assert stats["rows"] == 3
    assert stats["chars"] == sum(len(t) for t in ["hello world", "foo bar baz", "another row of text"])
    assert stats["files"] == 2


def test_load_corpus_stats_with_tokenizer_counts_tokens(tmp_path):
    """If a tokenizer path is provided, the stats include unique-token count."""
    d = tmp_path / "clean" / "train"
    d.mkdir(parents=True)
    _write_parquet(d / "part.parquet", ["şu beyit", "gül ile bülbül"])

    # Use the real tokenizer in the repo.
    tok_path = Path("tokenizer/asena-bpe-24k.json")
    if not tok_path.exists():
        pytest.skip("project tokenizer not available")
    stats = load_corpus_stats(str(d / "*.parquet"), tokenizer_path=str(tok_path))
    assert "tokens" in stats
    assert stats["tokens"] > 0
    assert stats["tokens"] < stats["chars"]  # BPE compression


# ---------------------------------------------------------------------------
# load_arch_config
# ---------------------------------------------------------------------------

def test_load_arch_config_extracts_model_dimensions(tmp_path):
    cfg = tmp_path / "promotion.yaml"
    cfg.write_text(yaml.safe_dump({
        "profile": "promotion",
        "model": {
            "n_layers": 12, "n_embd": 640, "n_head": 10, "n_kv_heads": 4,
            "mlp_ratio": 2.67, "rope_theta": 10000.0,
            "tie_embeddings": False, "init_std": 0.015, "max_seq_len": 2048,
        },
        "training": {
            "total_tokens": 60_000_000, "lr_peak": 6.0e-4,
            "weight_decay": 0.1, "seq_len": 2048,
        },
    }))
    arch = load_arch_config(cfg)
    assert arch["n_layers"] == 12
    assert arch["n_embd"] == 640
    assert arch["total_tokens"] == 60_000_000


# ---------------------------------------------------------------------------
# generate_model_card
# ---------------------------------------------------------------------------

def _meta(**overrides):
    base = {
        "model_name": "fuzuli-base",
        "version": "v0.1",
        "params": 84_000_000,
        "git_sha": "bbb222def",
        "baseline": {
            "score_ppl_bpb": 2.199,
            "score_lexicon": 2.677,
            "score_flatness": 0.0037,
            "score_smoke": 0.40,
        },
        "corpus": {
            "rows": 15_852,
            "chars": 28_798_649,
            "tokens": 7_968_478,
            "documents": 59,
        },
        "arch": {
            "n_layers": 12, "n_embd": 640, "n_head": 10, "n_kv_heads": 4,
            "mlp_ratio": 2.67, "tie_embeddings": False,
            "total_tokens": 60_000_000, "lr_peak": 6.0e-4,
        },
        "datasets": [
            "fatihburakkaragoz/anadolu-ocr-corpus",
            "fatihburakkaragoz/evliya-celebi-seyahatname-ocr",
        ],
        "license": "apache-2.0",
        "author": "Fatih Burak Karagöz",
        "affiliation": "CDLI",
        "arxiv_id": None,
        "samples": None,
    }
    base.update(overrides)
    return base


def test_generate_model_card_has_yaml_frontmatter():
    card = generate_model_card(_meta())
    assert card.startswith("---\n")
    head_end = card.index("\n---\n", 4) + len("\n---\n")
    front = card[:head_end]
    assert "license: apache-2.0" in front
    assert "language:" in front
    assert "library_name:" in front


def test_generate_model_card_includes_fuzuli_branding():
    card = generate_model_card(_meta())
    assert "Fuzuli" in card
    assert "Ottoman" in card


def test_generate_model_card_includes_scores_table():
    card = generate_model_card(_meta())
    # All four metrics from baseline should appear with their numeric values.
    assert "2.199" in card  # ppl
    assert "2.677" in card  # lexicon
    assert "0.0037" in card or "0.004" in card  # flatness rounded
    assert "0.40" in card or "0.4" in card  # smoke


def test_generate_model_card_includes_corpus_stats():
    card = generate_model_card(_meta())
    assert "7,968,478" in card or "7.97" in card  # token count
    assert "59" in card  # documents


def test_generate_model_card_includes_arch_table():
    card = generate_model_card(_meta())
    # Architecture dimensions should be visible
    assert "12" in card  # n_layers
    assert "640" in card  # n_embd
    assert "10" in card  # n_head
    assert "84" in card  # params (84M)


def test_generate_model_card_includes_datasets_in_frontmatter():
    card = generate_model_card(_meta())
    front = card.split("---", 2)[1]
    assert "fatihburakkaragoz/anadolu-ocr-corpus" in front
    assert "fatihburakkaragoz/evliya-celebi-seyahatname-ocr" in front


def test_generate_model_card_includes_arxiv_link_when_provided():
    card = generate_model_card(_meta(arxiv_id="2606.12345"))
    assert "2606.12345" in card
    assert "arxiv.org" in card.lower()


def test_generate_model_card_omits_arxiv_section_when_no_id():
    card = generate_model_card(_meta(arxiv_id=None))
    # No arxiv URL fragment should appear.
    assert "arxiv.org/abs/" not in card.lower()


def test_generate_model_card_includes_author_and_affiliation():
    card = generate_model_card(_meta())
    assert "Fatih Burak Karagöz" in card
    assert "CDLI" in card


def test_generate_model_card_renders_sample_generations_when_provided():
    card = generate_model_card(_meta(samples=[
        ("Şu beyitlerin manası", "...ki gülşen-i hüsne nazar etmek aşıkın."),
        ("İstanbul'un", "...kal'asınun ahvâli ve binâ-yı şehri evvelen..."),
    ]))
    assert "Şu beyitlerin manası" in card
    assert "İstanbul'un" in card
    # Some "samples" / "examples" heading should appear
    assert any(h in card.lower() for h in ("sample", "example", "generation"))


def test_generate_model_card_omits_samples_section_when_none():
    card = generate_model_card(_meta(samples=None))
    # Should NOT contain a "Sample generations" subsection
    assert "## Sample generations" not in card


def test_generate_model_card_body_has_no_8space_indented_prose():
    """Regression: a previous bug interpolated 0-indent tables into an 8-space
    indented dedent() block, defeating dedent and rendering the whole body as
    a code block on HuggingFace. No prose line should start with 8+ spaces."""
    card = generate_model_card(_meta())
    body = card.split("\n---\n", 1)[1]
    offending = [
        line for line in body.splitlines()
        if line.startswith("        ") and line.strip()
    ]
    assert not offending, (
        "found body lines indented with 8+ spaces (would render as code block "
        f"on HF):\n  " + "\n  ".join(offending[:5])
    )


def test_generate_model_card_renders_smoke_dict_samples_with_verdict():
    """Samples may also be dicts with prompt_id/generation/passed/reason
    (the shape produced by cli.py eval --include-samples). The card should
    render the prompt, completion, and a pass/fail verdict."""
    card = generate_model_card(_meta(samples=[
        {
            "prompt_id": "tanzimat_fermani",
            "generation": "Tanzimat fermanı bu kadar ma'mûr u müzeyyen ve müzeyyen ve müzeyyen…",
            "passed": False,
            "reason": "5gram repetition",
        },
        {
            "prompt_id": "bedestende_sarraflar",
            "generation": "Şehrin bedesteninde yirmi bin kadar...",
            "passed": True,
            "reason": "ok",
        },
    ]))
    assert "Tanzimat fermanı" in card
    assert "Şehrin bedesteninde" in card
    assert "5gram repetition" in card or "rejected" in card.lower()
    assert "accepted" in card.lower() or "✓" in card or "passed" in card.lower()


# ---------------------------------------------------------------------------
# load_eval_full
# ---------------------------------------------------------------------------

def test_load_eval_full_returns_scores_and_smoke_results(tmp_path):
    """eval-full.json (the real promotion eval output) should be parsed into
    a dict with the four scalar scores plus a list of smoke results."""
    cp = tmp_path / "checkpoints" / "fuzuli-v0.1"
    cp.mkdir(parents=True)
    (cp / "eval-full.json").write_text(
        '{"ppl_bpb": 1.929, "lexicon": 9.873, "flatness": 0.011, "smoke": 0.8, '
        '"smoke_results": [{"prompt_id": "p", "generation": "g", '
        '"passed": false, "reason": "loanword: x"}]}'
    )

    eval_data = load_eval_full(cp)
    assert eval_data["score_ppl_bpb"] == pytest.approx(1.929)
    assert eval_data["score_lexicon"] == pytest.approx(9.873)
    assert eval_data["score_flatness"] == pytest.approx(0.011)
    assert eval_data["score_smoke"] == pytest.approx(0.8)
    assert len(eval_data["smoke_results"]) == 1
    assert eval_data["smoke_results"][0]["prompt_id"] == "p"


def test_load_eval_full_returns_none_when_missing(tmp_path):
    cp = tmp_path / "checkpoints" / "fuzuli-v0.1"
    cp.mkdir(parents=True)
    assert load_eval_full(cp) is None


def test_generate_model_card_has_citation_block():
    card = generate_model_card(_meta())
    # BibTeX-like citation block must be present.
    assert "@" in card  # at least one @-prefixed citation entry
    assert "fuzuli" in card.lower()


def test_generate_model_card_states_honest_limitations():
    card = generate_model_card(_meta())
    # The model card must include a Limitations section warning users.
    assert "## Limitations" in card or "## Honest limitations" in card


# ---------------------------------------------------------------------------
# stage_release_folder
# ---------------------------------------------------------------------------

def _make_checkpoint(tmp_path: Path) -> Path:
    cp = tmp_path / "checkpoints" / "fuzuli-v0.1"
    cp.mkdir(parents=True)
    (cp / "model.pt").write_bytes(b"fake model weights")
    (cp / "tokenizer.json").write_text("{}")
    (cp / "eval_report.md").write_text("# eval\nfinal_loss: 1.23\n")
    return cp


def test_stage_release_folder_includes_required_files(tmp_path):
    cp = _make_checkpoint(tmp_path)
    staged = stage_release_folder(checkpoint_dir=cp, model_card="dummy card content",
                                  out_dir=tmp_path / "release")
    assert (staged / "model.pt").exists()
    assert (staged / "tokenizer.json").exists()
    assert (staged / "README.md").exists()
    assert (staged / "README.md").read_text() == "dummy card content"


def test_stage_release_folder_optional_article(tmp_path):
    cp = _make_checkpoint(tmp_path)
    article = tmp_path / "article"
    (article / "assets").mkdir(parents=True)
    (article / "README.md").write_text("# article")
    (article / "assets" / "diag.png").write_bytes(b"png")

    staged = stage_release_folder(checkpoint_dir=cp, model_card="card",
                                  out_dir=tmp_path / "release", article_dir=article)
    assert (staged / "article" / "README.md").exists()
    assert (staged / "article" / "assets" / "diag.png").exists()


def test_stage_release_folder_overwrites_existing(tmp_path):
    """Re-staging into the same out_dir should not fail or duplicate files."""
    cp = _make_checkpoint(tmp_path)
    out = tmp_path / "release"
    stage_release_folder(checkpoint_dir=cp, model_card="first", out_dir=out)
    staged = stage_release_folder(checkpoint_dir=cp, model_card="second", out_dir=out)
    assert (staged / "README.md").read_text() == "second"
