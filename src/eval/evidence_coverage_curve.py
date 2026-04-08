"""Coverage-curve metrics for ranked evidence lists."""

from __future__ import annotations


def cumulative_recall(relevant: list[bool]) -> list[float]:
    """Cumulative recall after each ranked evidence item."""
    total_relevant = sum(relevant)
    if total_relevant == 0:
        return [0.0 for _ in relevant]
    hits = 0
    curve = []
    for is_relevant in relevant:
        if is_relevant:
            hits += 1
        curve.append(hits / total_relevant)
    return curve


def area_under_coverage_curve(relevant: list[bool]) -> float:
    """Average of the cumulative recall curve."""
    if not relevant:
        return 0.0
    curve = cumulative_recall(relevant)
    return sum(curve) / len(curve)


def first_full_coverage_index(relevant: list[bool]) -> int | None:
    """Index where cumulative recall first reaches 1.0, if ever."""
    curve = cumulative_recall(relevant)
    for index, value in enumerate(curve):
        if value >= 1.0:
            return index
    return None
