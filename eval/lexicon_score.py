"""Ottoman lexicon coverage score (spec §5.1).

Lower = better (more Ottoman). score = -log(fraction_of_tokens_in_lexicon).

IMMUTABLE (Tier 1).
"""
from __future__ import annotations
import math
import re
from functools import lru_cache
from pathlib import Path


_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
_STOPWORDS: frozenset[str] = frozenset({
    "ve", "ile", "bir", "bu", "şu", "o", "ki", "ya", "de", "da", "den", "dan",
    "için", "gibi", "kadar", "her", "hem", "ne", "fakat", "ama", "lakin",
})


@lru_cache(maxsize=4)
def _load_lexicon(lexicon_path: str) -> frozenset[str]:
    return frozenset(
        line.strip().lower()
        for line in Path(lexicon_path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )


def _content_tokens(text: str) -> list[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in _STOPWORDS and not t.isdigit()]


def lexicon_score_from_text(text: str, lexicon_path: str) -> float:
    """Compute -log(fraction-in-lexicon) for a single text blob."""
    lex = _load_lexicon(lexicon_path)
    toks = _content_tokens(text)
    if not toks:
        return 20.0
    frac = sum(1 for t in toks if t in lex) / len(toks)
    if frac <= 0:
        return 20.0
    return -math.log(frac)


def compute_lexicon_score(
    generations: list[str],
    lexicon_path: Path,
) -> float:
    """Aggregate lexicon score across a list of generated texts (lower = better)."""
    if not generations:
        return 20.0
    scores = [lexicon_score_from_text(g, str(lexicon_path)) for g in generations]
    return sum(scores) / len(scores)
