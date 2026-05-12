"""Freeze-invariant + protected-path guards (spec §3.5, §7.4)."""
from __future__ import annotations
import hashlib
import json
import re
import fnmatch
import socket
from datetime import datetime, timezone
from pathlib import Path


class FreezeViolation(RuntimeError):
    pass


class ProtectedPathViolation(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_freeze_lock(lock_path: Path, files: dict[str, Path], frozen_by: str = "") -> None:
    """Compute SHA-256 for each file, write a FROZEN.lock JSON manifest."""
    if not frozen_by:
        frozen_by = f"{socket.gethostname()}"
    manifest = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_by": frozen_by,
        "files": {label: file_sha256(p) for label, p in files.items()},
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def verify_freeze_invariants(lock_path: Path, files: dict[str, Path]) -> None:
    """Raise FreezeViolation if any tracked file's hash doesn't match the lock."""
    if not lock_path.exists():
        raise FreezeViolation(f"freeze lock missing: {lock_path}")
    manifest = json.loads(lock_path.read_text())
    expected = manifest["files"]
    for label, path in files.items():
        if label not in expected:
            raise FreezeViolation(f"{label}: tracked file absent from lock")
        if not path.exists():
            raise FreezeViolation(f"{label}: file missing on disk ({path})")
        actual = file_sha256(path)
        if actual != expected[label]:
            raise FreezeViolation(
                f"hash mismatch for {label} ({path}): "
                f"expected {expected[label][:12]}..., got {actual[:12]}..."
            )


PROTECTED_PATHS: tuple[str, ...] = (
    "eval/**",
    "tokenizer/asena-bpe-24k.json",
    "factory/**",
    "cli.py",
    "SAFETY.md",
    "README.md",
    "data/modern_loanwords.txt",
    "agent/prompts/**",
    "**/FROZEN.lock",
)


def check_protected_paths(diff_paths: list[str]) -> None:
    """Raise ProtectedPathViolation if any path in diff_paths matches a protected glob."""
    for path in diff_paths:
        for pattern in PROTECTED_PATHS:
            if fnmatch.fnmatchcase(path, pattern):
                raise ProtectedPathViolation(
                    f"diff modifies protected path: {path} (pattern: {pattern})"
                )


class ForbiddenPatternViolation(RuntimeError):
    pass


FORBIDDEN_IMPORTS: tuple[str, ...] = (
    "torch.distributed",
    "MixtureOfExperts",
    "MoE",
    "mamba",
    "mamba_ssm",
    "s4",
    "hyena",
)

_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in FORBIDDEN_IMPORTS) + r")\b"
)


def scan_forbidden_patterns(code: str) -> None:
    """Raise ForbiddenPatternViolation if `code` contains any forbidden pattern."""
    m = _FORBIDDEN_RE.search(code)
    if m:
        raise ForbiddenPatternViolation(f"forbidden pattern: {m.group(1)}")
