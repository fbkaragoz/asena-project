import json
import os
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.mark.slow
def test_full_pipeline_on_tiny_corpus(tiny_corpus_dir, tmp_path, monkeypatch):
    """Walk through every CLI command (except autoresearch-run) on a 3-row corpus."""
    repo = tmp_path / "tiny_repo"
    repo.mkdir()
    # Bootstrap a self-contained git repo with our source tree.
    proj_root = Path(__file__).parent.parent
    for top in ("data", "tokenizer", "train", "eval", "factory", "agent",
                "tests", "cli.py", "pyproject.toml", "README.md", "SAFETY.md"):
        src = proj_root / top
        if src.is_dir():
            import shutil; shutil.copytree(src, repo / top)
        elif src.is_file():
            import shutil; shutil.copy(src, repo / top)
    # Replace the empty data/raw with our tiny corpus.
    (repo / "data" / "raw").mkdir(exist_ok=True)
    for p in tiny_corpus_dir.glob("*.parquet"):
        import shutil; shutil.copy(p, repo / "data" / "raw" / p.name)

    # Init git
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)

    py = sys.executable
    def _run(args):
        return subprocess.run([py, "cli.py", *args], cwd=repo, capture_output=True, text=True, check=True)

    # Phase A: data + tokenizer + freeze
    _run(["prepare-data", "--heldout-pct", "33"])
    _run(["train-tokenizer", "--vocab-size", "300"])
    _run(["freeze"])

    # Phase B: introspection (no checkpoint yet, so eval/train-sprint skip)
    out = _run(["ledger", "tail", "5"])
    assert out.stdout.strip().startswith("[")
    out = _run(["baseline", "show"])
    assert "(no baseline yet)" in out.stdout
