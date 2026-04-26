"""Token-overlap metrics between evidence passages."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def evidence_jaccard(left: str, right: str) -> float:
    """Jaccard overlap between two evidence passages."""
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def mean_evidence_overlap(passages: list[str]) -> float:
    """Average pairwise evidence overlap."""
    if len(passages) < 2:
        return 1.0 if passages else 0.0
    total = 0.0
    count = 0
    for index, left in enumerate(passages):
        for right in passages[index + 1 :]:
            total += evidence_jaccard(left, right)
            count += 1
    return total / count


def max_evidence_overlap(query: str, passages: list[str]) -> float:
    """Maximum overlap between a query and any evidence passage."""
    if not passages:
        return 0.0
    return max(evidence_jaccard(query, passage) for passage in passages)
