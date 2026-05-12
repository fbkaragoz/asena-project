from pathlib import Path
import pytest


@pytest.mark.slow
def test_run_sprint_cycle_end_to_end(tiny_corpus_dir, tmp_path, monkeypatch):
    """Driving a full train-sprint cycle requires a complete bootstrapped repo
    with frozen tokenizer + heldout. This is exercised by Task 32's E2E smoke."""
    pytest.skip("orchestrator E2E is exercised by the cli integration test in Task 32")
