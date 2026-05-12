"""Strict-no-trades policy combiner (spec §5.2).

The factory's accept/reject judge. ALL metrics must improve or stay flat;
at least one must improve beyond IMPROVEMENT_THRESHOLD. No weighted sums,
no trading.

IMMUTABLE (Tier 1). Editing this file invalidates baseline comparisons.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Scores:
    ppl_bpb: float
    lexicon: float
    flatness: float
    smoke: float


@dataclass
class Accept:
    deltas: dict[str, float]


@dataclass
class Reject:
    reason: str
    deltas: dict[str, float]


REGRESSION_TOLERANCE: dict[str, float] = {
    "ppl_bpb":  0.005,
    "lexicon":  0.02,
    "flatness": 0.002,
    "smoke":    0.0,
}

IMPROVEMENT_THRESHOLD: dict[str, float] = {
    "ppl_bpb":  0.015,
    "lexicon":  0.05,
    "flatness": 0.005,
    "smoke":    0.02,
}

NOISE_FLOOR: dict[str, float] = {
    "ppl_bpb":  0.003,
    "lexicon":  0.01,
    "flatness": 0.001,
    "smoke":    0.0,
}

_METRICS = ("ppl_bpb", "lexicon", "flatness", "smoke")


def decide(baseline: Scores, new: Scores) -> Accept | Reject:
    """Apply the accept/reject policy. ALL metrics must not regress."""
    deltas = {m: getattr(new, m) - getattr(baseline, m) for m in _METRICS}

    for m in _METRICS:
        if deltas[m] > REGRESSION_TOLERANCE[m]:
            return Reject(reason=f"regression in {m}: +{deltas[m]:.4f}", deltas=deltas)

    real_improvements = [m for m in _METRICS if deltas[m] < -IMPROVEMENT_THRESHOLD[m]]
    if real_improvements:
        return Accept(deltas=deltas)
    return Reject(reason="no real improvement (all within noise)", deltas=deltas)
