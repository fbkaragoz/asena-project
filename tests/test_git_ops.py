from pathlib import Path
import pytest
import subprocess
from factory.git_ops import (
    create_experiment_branch, accept_branch, reject_branch,
    list_diff_paths, get_current_sha, _run,
)


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """A tiny throwaway git repo with one initial commit on main."""
    _run(["git", "init", "-b", "main"], cwd=tmp_path)
    _run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path)
    _run(["git", "config", "user.name", "test"], cwd=tmp_path)
    (tmp_path / "README.md").write_text("hello\n")
    _run(["git", "add", "README.md"], cwd=tmp_path)
    _run(["git", "commit", "-m", "initial"], cwd=tmp_path)
    return tmp_path


def test_get_current_sha(tiny_repo):
    sha = get_current_sha(tiny_repo)
    assert len(sha) == 40


def test_create_branch_and_accept(tiny_repo):
    (tiny_repo / "train").mkdir()
    (tiny_repo / "train" / "train.py").write_text("# tiny\n")
    branch = create_experiment_branch(tiny_repo, name="exp/001-tiny", commit_message="exp")
    assert branch == "exp/001-tiny"
    accept_branch(tiny_repo, branch)
    # branch was merged + deleted; main now has the file
    assert (tiny_repo / "train" / "train.py").exists()
    out = _run(["git", "branch", "--list", branch], cwd=tiny_repo, capture_output=True).stdout
    assert out.strip() == b""


def test_reject_branch_deletes_and_restores_main(tiny_repo):
    (tiny_repo / "junk.txt").write_text("oops\n")
    branch = create_experiment_branch(tiny_repo, name="exp/002-junk", commit_message="junk")
    reject_branch(tiny_repo, branch)
    assert not (tiny_repo / "junk.txt").exists()
    out = _run(["git", "branch", "--list", branch], cwd=tiny_repo, capture_output=True).stdout
    assert out.strip() == b""


def test_list_diff_paths(tiny_repo):
    (tiny_repo / "train").mkdir()
    (tiny_repo / "train" / "train.py").write_text("# tiny\n")
    paths = list_diff_paths(tiny_repo, base="HEAD")
    assert "train/train.py" in paths
