"""Disk retention + free-space gating (spec §6.4)."""
from __future__ import annotations
import shutil
from pathlib import Path


class DiskFloorViolation(RuntimeError):
    pass


def free_disk_gb(path: Path) -> float:
    s = shutil.disk_usage(path)
    return s.free / (1024 ** 3)


def check_disk_floor(path: Path, min_gb: float = 20.0) -> None:
    free = free_disk_gb(path)
    if free < min_gb:
        raise DiskFloorViolation(f"free disk {free:.1f} GB < min {min_gb:.1f} GB at {path}")


def cleanup_sprint_checkpoints(checkpoint_dir: Path) -> None:
    """Keep the single most recently modified .pt file; delete the rest."""
    files = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    for p in files[:-1]:
        p.unlink()


def cleanup_promotion_checkpoints(
    checkpoint_dir: Path,
    keep_last_n: int = 5,
    best_n_paths: list[Path] | None = None,
) -> None:
    """Keep the last N checkpoints by mtime, plus any explicit best-N paths."""
    best = set(best_n_paths or [])
    files = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    keep = set(files[-keep_last_n:]) | best
    for p in files:
        if p not in keep:
            p.unlink()
