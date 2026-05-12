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
