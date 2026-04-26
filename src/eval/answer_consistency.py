"""Consistency metrics across multiple generated answers."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def normalize_answer(text: str) -> tuple[str, ...]:
    """Normalize an answer into lexical tokens."""
    return tuple(_TOKEN_RE.findall(text.lower()))


def exact_match_rate(answers: list[str]) -> float:
    """Fraction of answers equal to the modal normalized answer."""
    if not answers:
        return 0.0
    normalized = [normalize_answer(answer) for answer in answers]
    counts: dict[tuple[str, ...], int] = {}
    for answer in normalized:
        counts[answer] = counts.get(answer, 0) + 1
    majority = max(counts.values())
    return majority / len(answers)


def lexical_overlap(a: str, b: str) -> float:
    """Jaccard overlap between normalized answers."""
    set_a = set(normalize_answer(a))
    set_b = set(normalize_answer(b))
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def mean_pairwise_consistency(answers: list[str]) -> float:
    """Average pairwise lexical overlap across answers."""
    if len(answers) < 2:
        return 1.0 if answers else 0.0
    total = 0.0
    count = 0
    for index, left in enumerate(answers):
        for right in answers[index + 1 :]:
            total += lexical_overlap(left, right)
            count += 1
    return total / count
