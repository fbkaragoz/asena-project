"""Modern-Turkish loanword penalty (spec §5.1).

flatness = (# modern-loanword tokens in generations) / (total tokens)

IMMUTABLE (Tier 1).
"""
from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


@lru_cache(maxsize=4)
def _load_blacklist(path: str) -> frozenset[str]:
    return frozenset(
        line.strip().lower()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )


def compute_flatness(generations: list[str], blacklist_path: str | Path) -> float:
    """Return the modern-loanword ratio across all generated texts."""
    bl = _load_blacklist(str(blacklist_path))
    tokens: list[str] = []
    for g in generations:
        tokens.extend(t.lower() for t in _TOKEN_RE.findall(g))
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in bl) / len(tokens)
