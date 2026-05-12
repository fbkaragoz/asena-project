"""Cleaning pipeline stages (spec §3.2).

Stages 1, 3, 4 are LOCKED — must never be edited by the agent (Tier 1).
Stage 2 (apply_cleaning_rules) is agent-editable via data/cleaning_rules.yaml.
"""
from __future__ import annotations
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
import yaml
from datasketch import MinHash, MinHashLSH

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Stage 1: locked normalization.

    - Apply Unicode NFC (combines decomposed diacritics).
    - Strip ASCII control characters except \\t \\n.
    - Convert \\r and \\r\\n to space, then collapse all whitespace runs to a
      single ASCII space.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Stage 2: agent-editable cleaning (Tier 2)
# ---------------------------------------------------------------------------

@dataclass
class CleaningRules:
    version: int
    substitutions: list[tuple[re.Pattern, str]]
    min_chars: int
    max_chars: int
    modern_loanwords: frozenset[str]
    max_modern_ratio: float
    era_weights: dict[str, float]


def load_cleaning_rules(path: Path) -> CleaningRules:
    """Load Stage 2 cleaning rules from YAML; load modern loanwords from blacklist file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    subs = [(re.compile(s["pattern"]), s["replace"]) for s in cfg.get("substitutions", [])]
    lf = cfg["length_filters"]
    mt = cfg["modern_turkish_filter"]
    er = cfg["era_routing"]
    blacklist_path = Path(mt["blacklist_file"])
    if not blacklist_path.is_absolute():
        blacklist_path = path.parent.parent / blacklist_path
    with open(blacklist_path) as f:
        loanwords = frozenset(w.strip().lower() for w in f if w.strip() and not w.startswith("#"))
    return CleaningRules(
        version=cfg["version"], substitutions=subs,
        min_chars=lf["min_chars"], max_chars=lf["max_chars"],
        modern_loanwords=loanwords, max_modern_ratio=mt["max_ratio"],
        era_weights={k: v["weight"] for k, v in er.items()},
    )


_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def apply_cleaning_rules(text: str, rules: CleaningRules) -> str | None:
    """Stage 2: agent-editable cleaning.

    Returns the cleaned string, or None if the line should be dropped (filtered).
    """
    for pat, repl in rules.substitutions:
        text = pat.sub(repl, text)
    text = text.strip()
    if not (rules.min_chars <= len(text) <= rules.max_chars):
        return None
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if not tokens:
        return None
    modern_count = sum(1 for t in tokens if t in rules.modern_loanwords)
    if modern_count / len(tokens) > rules.max_modern_ratio:
        return None
    return text


# ---------------------------------------------------------------------------
# Stage 3: locked MinHash near-duplicate removal (Tier 1)
# ---------------------------------------------------------------------------


def _shingles(text: str, k: int = 5) -> set[str]:
    """Character k-shingles for MinHash."""
    text = text.lower()
    return {text[i:i+k] for i in range(max(0, len(text) - k + 1))}


def dedup_minhash(texts: list[str], threshold: float = 0.85, num_perm: int = 128) -> list[int]:
    """Stage 3: locked MinHash near-duplicate removal.

    Returns the list of INDICES into `texts` that should be kept. Greedy: first
    occurrence wins. Deterministic given input order.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep: list[int] = []
    for i, t in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        for sh in _shingles(t):
            m.update(sh.encode("utf-8"))
        if lsh.query(m):
            continue
        lsh.insert(str(i), m)
        keep.append(i)
    return keep


# ---------------------------------------------------------------------------
# Stage 4: locked deterministic train/heldout split (Tier 1)
# ---------------------------------------------------------------------------


def split_train_heldout(
    rows: list[dict], heldout_pct: int = 2
) -> tuple[list[dict], list[dict]]:
    """Stage 4: deterministic per-document train/heldout split.

    Hashes source_pdf; rows where hash(source_pdf) % 100 < heldout_pct → heldout.
    All rows from one source_pdf land in the same split.
    """
    train, heldout = [], []
    for row in rows:
        key = row["source_pdf"].encode("utf-8")
        bucket = int(hashlib.sha256(key).hexdigest(), 16) % 100
        (heldout if bucket < heldout_pct else train).append(row)
    return train, heldout
