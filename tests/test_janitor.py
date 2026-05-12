from pathlib import Path
import pytest
from factory.janitor import (
    cleanup_sprint_checkpoints, cleanup_promotion_checkpoints,
    free_disk_gb, DiskFloorViolation, check_disk_floor,
)


def test_cleanup_sprint_keeps_only_last(tmp_path):
    cps = tmp_path / "sprint_checkpoints"
    cps.mkdir()
    import time
    for name in ["a.pt", "b.pt", "c.pt"]:
        p = cps / name
        p.write_text("x")
        time.sleep(0.01)
    cleanup_sprint_checkpoints(cps)
    remaining = sorted(p.name for p in cps.glob("*.pt"))
    assert remaining == ["c.pt"]   # most recent only


def test_cleanup_promotion_retention(tmp_path):
    cps = tmp_path / "promo"
    cps.mkdir()
    import time
    for name in [f"step_{i:04d}.pt" for i in range(1, 11)]:
        p = cps / name
        p.write_text("x")
        time.sleep(0.005)
    cleanup_promotion_checkpoints(cps, keep_last_n=3, best_n_paths=[cps / "step_0007.pt"])
    remaining = sorted(p.name for p in cps.glob("*.pt"))
    assert "step_0010.pt" in remaining
    assert "step_0007.pt" in remaining
    assert "step_0001.pt" not in remaining


def test_free_disk_gb(tmp_path):
    v = free_disk_gb(tmp_path)
    assert v > 0


def test_check_disk_floor_passes(tmp_path):
    check_disk_floor(tmp_path, min_gb=0)


def test_check_disk_floor_violates_high_floor(tmp_path):
    with pytest.raises(DiskFloorViolation):
        check_disk_floor(tmp_path, min_gb=10**9)
