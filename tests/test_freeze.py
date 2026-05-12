import json
from pathlib import Path
import pytest
from factory.guards import (
    write_freeze_lock, verify_freeze_invariants, FreezeViolation, file_sha256,
)


def test_write_and_verify_freeze_lock(tmp_path):
    target = tmp_path / "thing.json"
    target.write_text('{"hello": "world"}')
    lock = tmp_path / "FROZEN.lock"

    write_freeze_lock(lock, {"thing.json": target}, frozen_by="test")
    verify_freeze_invariants(lock, {"thing.json": target})  # should not raise


def test_verify_freeze_invariants_detects_mutation(tmp_path):
    target = tmp_path / "thing.json"
    target.write_text("original")
    lock = tmp_path / "FROZEN.lock"
    write_freeze_lock(lock, {"thing.json": target}, frozen_by="test")
    target.write_text("mutated")
    with pytest.raises(FreezeViolation, match="hash mismatch"):
        verify_freeze_invariants(lock, {"thing.json": target})


def test_freeze_lock_format(tmp_path):
    target = tmp_path / "thing.json"
    target.write_text("x")
    lock = tmp_path / "FROZEN.lock"
    write_freeze_lock(lock, {"thing.json": target}, frozen_by="alice")
    data = json.loads(lock.read_text())
    assert "thing.json" in data["files"]
    assert "frozen_utc" in data
    assert data["frozen_by"] == "alice"


def test_file_sha256_is_stable(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"hello world")
    h1 = file_sha256(p)
    h2 = file_sha256(p)
    assert h1 == h2
    assert len(h1) == 64


from factory.guards import (
    check_protected_paths, ProtectedPathViolation,
    scan_forbidden_patterns, ForbiddenPatternViolation,
    PROTECTED_PATHS, FORBIDDEN_IMPORTS,
)


def test_check_protected_paths_blocks_eval_edit():
    diff_paths = ["eval/policy.py"]
    import pytest
    with pytest.raises(ProtectedPathViolation, match="eval/policy.py"):
        check_protected_paths(diff_paths)


def test_check_protected_paths_blocks_frozen_lock():
    import pytest
    with pytest.raises(ProtectedPathViolation):
        check_protected_paths(["tokenizer/FROZEN.lock"])


def test_check_protected_paths_blocks_safety_md():
    import pytest
    with pytest.raises(ProtectedPathViolation):
        check_protected_paths(["SAFETY.md"])


def test_check_protected_paths_allows_train_edit():
    check_protected_paths(["train/train.py", "data/cleaning_rules.yaml"])  # no raise


def test_scan_forbidden_patterns_blocks_distributed():
    code = "import torch.distributed as dist\n"
    import pytest
    with pytest.raises(ForbiddenPatternViolation, match="torch.distributed"):
        scan_forbidden_patterns(code)


def test_scan_forbidden_patterns_blocks_moe():
    code = "class MixtureOfExperts(nn.Module): pass"
    import pytest
    with pytest.raises(ForbiddenPatternViolation):
        scan_forbidden_patterns(code)


def test_scan_forbidden_patterns_allows_normal_torch():
    code = "import torch\nimport torch.nn as nn\nclass Block(nn.Module): pass\n"
    scan_forbidden_patterns(code)  # no raise


def test_forbidden_imports_includes_expected():
    assert "torch.distributed" in FORBIDDEN_IMPORTS
    assert any("MoE" in p or "MixtureOfExperts" in p for p in FORBIDDEN_IMPORTS)
