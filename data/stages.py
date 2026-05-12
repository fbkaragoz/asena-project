"""Cleaning pipeline stages (spec §3.2).

Stages 1, 3, 4 are LOCKED — must never be edited by the agent (Tier 1).
Stage 2 (apply_cleaning_rules) is agent-editable via data/cleaning_rules.yaml.
"""
from __future__ import annotations
import re
import unicodedata

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
