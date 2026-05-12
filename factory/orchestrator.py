"""train-sprint pipeline (spec §6.1).

Orchestrates: pre-flight checks → branch → smoke → sprint → eval → accept/reject.
Designed to be called by `cli.py train-sprint`. Pure procedural; no global state.

IMMUTABLE (Tier 1).
"""
from __future__ import annotations
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml

from factory.guards import (
    verify_freeze_invariants, check_protected_paths, scan_forbidden_patterns,
    FreezeViolation, ProtectedPathViolation, ForbiddenPatternViolation,
)
from factory.bounds import (
    estimate_param_count, check_sprint_bounds, free_vram_mb, BoundsViolation,
)
from factory.db import Ledger, ExperimentRow
from factory.git_ops import (
    create_experiment_branch, accept_branch, reject_branch,
    list_diff_paths, get_current_sha,
)
from factory.janitor import check_disk_floor, cleanup_sprint_checkpoints
from eval.policy import Scores, decide, Accept, Reject


REPO_ROOT = Path(".")
TOKENIZER_LOCK = Path("tokenizer/FROZEN.lock")
TOKENIZER_PATH = Path("tokenizer/asena-bpe-24k.json")
HELDOUT_LOCK = Path("eval/heldout/FROZEN.lock")
HELDOUT_DIR = Path("eval/heldout/text")
SMOKE_PROMPTS = Path("eval/heldout/smoke_prompts.yaml")
LEXICON = Path("eval/heldout/ottoman_lexicon.txt")
LOANWORDS = Path("data/modern_loanwords.txt")
CHECKPOINT_DIR = Path("checkpoints/sprints")
LEDGER_PATH = Path("experiments.sqlite")
TRAIN_DIR = Path("train")
TRAIN_GLOB = "data/clean/train/*.parquet"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pre_flight() -> None:
    """Step 1-3 of spec §6.1: working tree clean of protected mods + VRAM + freeze."""
    diff_paths = list_diff_paths(REPO_ROOT, base="HEAD")
    check_protected_paths(diff_paths)
    free_mb = free_vram_mb()
    if free_mb and free_mb < 20_000:
        raise BoundsViolation(f"free VRAM {free_mb}MB < 20000MB (kill VLLM or any process holding VRAM)")
    check_disk_floor(REPO_ROOT, min_gb=20.0)
    if TOKENIZER_LOCK.exists():
        verify_freeze_invariants(TOKENIZER_LOCK, {"tokenizer.json": TOKENIZER_PATH})
    if HELDOUT_LOCK.exists():
        heldout_files = sorted(HELDOUT_DIR.glob("*.parquet"))
        verify_freeze_invariants(HELDOUT_LOCK, {f"text/{p.name}": p for p in heldout_files})


def _patch_scan() -> None:
    """Scan the agent's edits for forbidden patterns (spec §7.4)."""
    for py in TRAIN_DIR.rglob("*.py"):
        scan_forbidden_patterns(py.read_text(encoding="utf-8"))


def _run_smoke() -> tuple[bool, str]:
    """30-second smoke training; abort if NaN/Inf or import error."""
    from train.train import run_training
    tmp_out = CHECKPOINT_DIR / "smoke.pt"
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = run_training(
            config_path=Path("train/configs/sprint.yaml"),
            tokenizer_path=TOKENIZER_PATH,
            train_glob=TRAIN_GLOB,
            checkpoint_out=tmp_out,
            max_steps=10,
            device="cuda",
        )
    except Exception as e:
        return False, f"smoke import/exec error: {e}"
    for v in r["losses"]:
        if v != v or v == float("inf"):
            return False, "smoke loss NaN/Inf"
    return True, "ok"


def _run_sprint() -> tuple[Path, dict]:
    from train.train import run_training
    ckpt = CHECKPOINT_DIR / f"sprint_{int(time.time())}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    r = run_training(
        config_path=Path("train/configs/sprint.yaml"),
        tokenizer_path=TOKENIZER_PATH,
        train_glob=TRAIN_GLOB,
        checkpoint_out=ckpt,
        device="cuda",
    )
    return ckpt, r


def _evaluate(checkpoint: Path) -> Scores:
    from eval.heldout_ppl import compute_heldout_bpb
    from eval.lexicon_score import compute_lexicon_score
    from eval.flatness import compute_flatness
    from eval.smoke import evaluate_smoke_prompts

    bpb = compute_heldout_bpb(
        checkpoint_path=checkpoint, tokenizer_path=TOKENIZER_PATH,
        heldout_glob=str(HELDOUT_DIR / "*.parquet"),
    )
    fail_rate, results = evaluate_smoke_prompts(
        checkpoint_path=checkpoint, tokenizer_path=TOKENIZER_PATH,
        prompts_path=SMOKE_PROMPTS, blacklist_path=LOANWORDS,
    )
    generations = [r.generation for r in results]
    lex = compute_lexicon_score(generations, lexicon_path=LEXICON)
    flat = compute_flatness(generations, blacklist_path=LOANWORDS)
    return Scores(ppl_bpb=bpb, lexicon=lex, flatness=flat, smoke=fail_rate)


def _row(started_utc, git_sha_before, git_sha_after, branch_name, outcome, reason,
         scores, train_stats, diff) -> ExperimentRow:
    return ExperimentRow(
        started_utc=started_utc, finished_utc=_now_utc(),
        git_sha_before=git_sha_before, git_sha_after=git_sha_after,
        branch_name=branch_name, scope=None, hypothesis="",
        diff=diff, outcome=outcome, reject_reason=reason,
        delta_ppl_bpb=0.0, delta_lexicon=0.0, delta_flatness=0.0, delta_smoke=0.0,
        score_ppl_bpb=scores.ppl_bpb if scores else 0.0,
        score_lexicon=scores.lexicon if scores else 0.0,
        score_flatness=scores.flatness if scores else 0.0,
        score_smoke=scores.smoke if scores else 0.0,
        train_tokens=train_stats["tokens_seen"] if train_stats else 0,
        train_steps=0,
        train_seconds=train_stats["wall_seconds"] if train_stats else 0.0,
        peak_vram_mb=0,
    )


def run_train_sprint() -> dict[str, Any]:
    """Run one full sprint cycle. Print + return outcome JSON."""
    started_utc = _now_utc()
    ledger = Ledger(LEDGER_PATH)
    git_sha_before = get_current_sha(REPO_ROOT)
    branch_name = f"exp/{int(time.time())}"

    _pre_flight()
    _patch_scan()

    cfg = yaml.safe_load(open("train/configs/sprint.yaml"))
    from tokenizers import Tokenizer
    vocab = Tokenizer.from_file(str(TOKENIZER_PATH)).get_vocab_size() if TOKENIZER_PATH.exists() else 24000
    params = estimate_param_count(
        n_layers=cfg["model"]["n_layers"], n_embd=cfg["model"]["n_embd"],
        n_head=cfg["model"]["n_head"], n_kv_heads=cfg["model"]["n_kv_heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"], vocab_size=vocab,
        tied=cfg["model"]["tie_embeddings"],
    )
    check_sprint_bounds(params=params, estimated_seconds=350,
                        estimated_vram_mb=int(8 * params / 1_000_000 + 4000))

    branch_name = create_experiment_branch(
        REPO_ROOT, name=branch_name, commit_message=f"exp: {branch_name}"
    )

    ok, smoke_msg = _run_smoke()
    if not ok:
        ledger.insert(_row(started_utc, git_sha_before, None, branch_name,
                           outcome="reject_smoke", reason=smoke_msg,
                           scores=None, train_stats=None, diff=""))
        reject_branch(REPO_ROOT, branch_name)
        cleanup_sprint_checkpoints(CHECKPOINT_DIR)
        return {"outcome": "reject_smoke", "reason": smoke_msg}

    ckpt, train_stats = _run_sprint()
    new_scores = _evaluate(ckpt)
    baseline = ledger.get_baseline()
    if baseline is None:
        accept_branch(REPO_ROOT, branch_name)
        sha_after = get_current_sha(REPO_ROOT)
        eid = ledger.insert(_row(started_utc, git_sha_before, sha_after, branch_name,
                                 outcome="accept", reason=None,
                                 scores=new_scores, train_stats=train_stats, diff=""))
        ledger.set_baseline(eid, git_sha=sha_after,
                            scores={"score_ppl_bpb": new_scores.ppl_bpb,
                                    "score_lexicon": new_scores.lexicon,
                                    "score_flatness": new_scores.flatness,
                                    "score_smoke":   new_scores.smoke})
        cleanup_sprint_checkpoints(CHECKPOINT_DIR)
        return {"outcome": "accept", "first_baseline": True, "scores": asdict(new_scores)}

    base_scores = Scores(
        ppl_bpb=baseline["score_ppl_bpb"], lexicon=baseline["score_lexicon"],
        flatness=baseline["score_flatness"], smoke=baseline["score_smoke"],
    )
    decision = decide(base_scores, new_scores)
    if isinstance(decision, Accept):
        accept_branch(REPO_ROOT, branch_name)
        sha_after = get_current_sha(REPO_ROOT)
        eid = ledger.insert(_row(started_utc, git_sha_before, sha_after, branch_name,
                                 outcome="accept", reason=None,
                                 scores=new_scores, train_stats=train_stats, diff=""))
        ledger.set_baseline(eid, git_sha=sha_after,
                            scores={"score_ppl_bpb": new_scores.ppl_bpb,
                                    "score_lexicon": new_scores.lexicon,
                                    "score_flatness": new_scores.flatness,
                                    "score_smoke":   new_scores.smoke})
        out = {"outcome": "accept", "deltas": decision.deltas, "scores": asdict(new_scores)}
    else:
        ledger.insert(_row(started_utc, git_sha_before, None, branch_name,
                           outcome="reject_eval", reason=decision.reason,
                           scores=new_scores, train_stats=train_stats, diff=""))
        reject_branch(REPO_ROOT, branch_name)
        out = {"outcome": "reject_eval", "reason": decision.reason,
               "deltas": decision.deltas, "scores": asdict(new_scores)}
    cleanup_sprint_checkpoints(CHECKPOINT_DIR)
    return out
