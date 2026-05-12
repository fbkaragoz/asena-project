"""Thin wrappers around git for the autoresearch loop (spec §6.3)."""
from __future__ import annotations
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path, capture_output: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=capture_output, check=check)


def get_current_sha(repo: Path) -> str:
    out = _run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True).stdout
    return out.decode().strip()


def list_diff_paths(repo: Path, base: str = "HEAD") -> list[str]:
    """List paths that differ from `base` (working tree + staged + untracked)."""
    staged = _run(["git", "diff", "--name-only", base], cwd=repo, capture_output=True).stdout.decode().splitlines()
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, capture_output=True).stdout.decode().splitlines()
    return sorted({*staged, *untracked, *_run(["git", "diff", "--name-only", "--cached"], cwd=repo, capture_output=True).stdout.decode().splitlines()})


def create_experiment_branch(repo: Path, name: str, commit_message: str) -> str:
    """Stage all working changes, commit on a new branch off main.

    Caller is responsible for having put the desired changes in the working tree.
    """
    _run(["git", "checkout", "-b", name], cwd=repo)
    _run(["git", "add", "-A"], cwd=repo)
    _run(["git", "commit", "-m", commit_message], cwd=repo)
    return name


def accept_branch(repo: Path, name: str) -> None:
    """Fast-forward merge `name` into main; delete the branch."""
    _run(["git", "checkout", "main"], cwd=repo)
    _run(["git", "merge", "--ff-only", name], cwd=repo)
    _run(["git", "branch", "-D", name], cwd=repo)


def reject_branch(repo: Path, name: str) -> None:
    """Checkout main, force-delete the branch."""
    _run(["git", "checkout", "main"], cwd=repo)
    _run(["git", "branch", "-D", name], cwd=repo)
