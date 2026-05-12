from pathlib import Path
import pytest
from factory.db import Ledger, ExperimentRow


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "experiments.sqlite")


def test_ledger_initializes_schema(ledger):
    assert ledger.list_experiments() == []


def test_ledger_insert_and_list(ledger):
    row = ExperimentRow(
        started_utc="2026-05-12T10:00:00Z",
        finished_utc="2026-05-12T10:05:00Z",
        git_sha_before="abc123",
        git_sha_after="def456",
        branch_name="exp/001",
        scope="optimizer-swap",
        hypothesis="Muon converges faster",
        diff="--- a\n+++ b\n",
        outcome="accept",
        reject_reason=None,
        delta_ppl_bpb=-0.05, delta_lexicon=0.0, delta_flatness=0.0, delta_smoke=0.0,
        score_ppl_bpb=4.10, score_lexicon=1.20, score_flatness=0.005, score_smoke=0.10,
        train_tokens=25_000_000, train_steps=400, train_seconds=298.0, peak_vram_mb=18_000,
    )
    eid = ledger.insert(row)
    rows = ledger.list_experiments()
    assert len(rows) == 1
    assert rows[0]["id"] == eid
    assert rows[0]["outcome"] == "accept"


def test_baseline_pointer(ledger):
    eid = ledger.insert(ExperimentRow(
        started_utc="t", finished_utc="t", git_sha_before="x", git_sha_after="y",
        branch_name="exp/001", scope="hparam", hypothesis="", diff="",
        outcome="accept", reject_reason=None,
        delta_ppl_bpb=0, delta_lexicon=0, delta_flatness=0, delta_smoke=0,
        score_ppl_bpb=4.0, score_lexicon=1.0, score_flatness=0.01, score_smoke=0.1,
        train_tokens=0, train_steps=0, train_seconds=0.0, peak_vram_mb=0,
    ))
    ledger.set_baseline(eid, git_sha="y",
                        scores={"score_ppl_bpb": 4.0, "score_lexicon": 1.0,
                                "score_flatness": 0.01, "score_smoke": 0.1})
    b = ledger.get_baseline()
    assert b["score_ppl_bpb"] == 4.0
    assert b["git_sha"] == "y"


def test_query_filters_by_scope(ledger):
    for scope in ("optimizer-swap", "optimizer-swap", "data-mix"):
        ledger.insert(ExperimentRow(
            started_utc="t", finished_utc=None, git_sha_before="x", git_sha_after=None,
            branch_name="exp/x", scope=scope, hypothesis="", diff="",
            outcome="reject_eval", reject_reason="meh",
            delta_ppl_bpb=0, delta_lexicon=0, delta_flatness=0, delta_smoke=0,
            score_ppl_bpb=4.0, score_lexicon=1.0, score_flatness=0.01, score_smoke=0.1,
            train_tokens=0, train_steps=0, train_seconds=0.0, peak_vram_mb=0,
        ))
    assert len(ledger.query(scope="optimizer-swap")) == 2
    assert len(ledger.query(scope="data-mix")) == 1
