"""Agreement metrics between paired judge outputs."""

from __future__ import annotations

from collections import Counter


def pair_agreement(left: list[str], right: list[str]) -> float:
    """Fraction of matching pairwise judge labels."""
    if len(left) != len(right):
        raise ValueError("left and right must have the same length")
    if not left:
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


def majority_pair_label(labels: list[str]) -> str:
    """Most common pairwise label."""
    if not labels:
        raise ValueError("labels must be non-empty")
    return Counter(labels).most_common(1)[0][0]


def pair_label_diversity(labels: list[str]) -> float:
    """Fraction of unique labels in a list."""
    if not labels:
        return 0.0
    return len(set(labels)) / len(labels)
