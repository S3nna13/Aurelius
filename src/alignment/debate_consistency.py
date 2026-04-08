"""Consistency checks for debate outcomes across multiple runs."""

from __future__ import annotations


def winner_consistency(winners: list[str]) -> float:
    """Fraction of debate runs matching the modal winner."""
    if not winners:
        return 0.0
    counts: dict[str, int] = {}
    for winner in winners:
        counts[winner] = counts.get(winner, 0) + 1
    return max(counts.values()) / len(winners)


def argument_overlap(arguments_a: list[str], arguments_b: list[str]) -> float:
    """Jaccard overlap between two argument sets."""
    set_a = {argument.lower() for argument in arguments_a}
    set_b = {argument.lower() for argument in arguments_b}
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def debate_run_consistency(debate_runs: list[list[str]]) -> float:
    """Average pairwise overlap between debate runs."""
    if len(debate_runs) < 2:
        return 1.0 if debate_runs else 0.0
    total = 0.0
    count = 0
    for index, left in enumerate(debate_runs):
        for right in debate_runs[index + 1 :]:
            total += argument_overlap(left, right)
            count += 1
    return total / count

