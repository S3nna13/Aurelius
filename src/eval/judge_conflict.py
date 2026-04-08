"""Conflict metrics across judge panels."""

from __future__ import annotations

from collections import Counter


def conflict_rate(verdicts: list[str]) -> float:
    """One minus the modal-verdict fraction."""
    if not verdicts:
        return 0.0
    counts = Counter(verdicts)
    return 1.0 - max(counts.values()) / len(verdicts)


def unanimous(verdicts: list[str]) -> bool:
    """Whether all verdicts agree."""
    return len(set(verdicts)) <= 1


def conflict_entropy(verdicts: list[str]) -> float:
    """Shannon entropy over discrete verdict labels."""
    if not verdicts:
        return 0.0
    counts = Counter(verdicts)
    total = len(verdicts)
    entropy = 0.0
    for count in counts.values():
        prob = count / total
        entropy -= prob * (0.0 if prob == 0.0 else __import__("math").log(prob))
    return entropy
