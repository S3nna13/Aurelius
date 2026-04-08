"""Precision-style metrics for evidence retrieval lists."""

from __future__ import annotations


def precision_at_k(relevant: list[bool], k: int) -> float:
    """Precision at k over a ranked relevance list."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    top = relevant[:k]
    return sum(top) / len(top)


def average_precision(relevant: list[bool]) -> float:
    """Average precision over a ranked relevance list."""
    hits = 0
    total = 0.0
    for index, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            hits += 1
            total += hits / index
    if hits == 0:
        return 0.0
    return total / hits


def recall_at_k(relevant: list[bool], k: int) -> float:
    """Recall at k over a ranked relevance list."""
    if k <= 0:
        raise ValueError("k must be positive")
    total_relevant = sum(relevant)
    if total_relevant == 0:
        return 0.0
    return sum(relevant[:k]) / total_relevant
