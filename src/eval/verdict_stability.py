"""Stability metrics for repeated verdict generation."""

from __future__ import annotations

from collections import Counter


def verdict_mode(verdicts: list[str]) -> str:
    """Return the modal verdict."""
    if not verdicts:
        raise ValueError("verdicts must be non-empty")
    return Counter(verdicts).most_common(1)[0][0]


def verdict_stability(verdicts: list[str]) -> float:
    """Fraction of verdicts matching the modal verdict."""
    if not verdicts:
        return 0.0
    mode = verdict_mode(verdicts)
    return sum(verdict == mode for verdict in verdicts) / len(verdicts)


def verdict_flip_rate(verdicts: list[str]) -> float:
    """Fraction of adjacent verdict transitions that flip labels."""
    if len(verdicts) < 2:
        return 0.0
    flips = sum(left != right for left, right in zip(verdicts, verdicts[1:]))
    return flips / (len(verdicts) - 1)
