"""Diversity metrics over judge outputs and rationales."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def unique_verdict_fraction(verdicts: list[str]) -> float:
    """Fraction of distinct verdict labels present."""
    if not verdicts:
        return 0.0
    return len(set(verdicts)) / len(verdicts)


def lexical_diversity(texts: list[str]) -> float:
    """Type-token ratio across a set of texts."""
    tokens = []
    for text in texts:
        tokens.extend(_TOKEN_RE.findall(text.lower()))
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def pairwise_verdict_disagreement(verdicts_a: list[str], verdicts_b: list[str]) -> float:
    """Fraction of items where two judges disagree."""
    if len(verdicts_a) != len(verdicts_b):
        raise ValueError("verdict lists must have the same length")
    if not verdicts_a:
        return 0.0
    return sum(a != b for a, b in zip(verdicts_a, verdicts_b)) / len(verdicts_a)
