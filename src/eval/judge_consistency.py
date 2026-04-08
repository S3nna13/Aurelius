"""Consistency metrics for sets of judge decisions."""

from __future__ import annotations

from collections import Counter


def pairwise_agreement(labels_a: list[str], labels_b: list[str]) -> float:
    """Fraction of matching labels across two judge outputs."""
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have the same length")
    if not labels_a:
        return 0.0
    return sum(left == right for left, right in zip(labels_a, labels_b)) / len(labels_a)


def majority_label(labels: list[str]) -> str:
    """Return the most common label."""
    if not labels:
        raise ValueError("labels must be non-empty")
    return Counter(labels).most_common(1)[0][0]


def consensus_rate(judge_outputs: list[list[str]]) -> float:
    """Fraction of items where all judges agree."""
    if not judge_outputs:
        return 0.0
    n_items = len(judge_outputs[0])
    if any(len(output) != n_items for output in judge_outputs):
        raise ValueError("All judge outputs must have the same length")
    if n_items == 0:
        return 0.0
    agreed = 0
    for item_idx in range(n_items):
        labels = {output[item_idx] for output in judge_outputs}
        if len(labels) == 1:
            agreed += 1
    return agreed / n_items


def majority_vote(judge_outputs: list[list[str]]) -> list[str]:
    """Compute per-item majority votes across judges."""
    if not judge_outputs:
        return []
    n_items = len(judge_outputs[0])
    if any(len(output) != n_items for output in judge_outputs):
        raise ValueError("All judge outputs must have the same length")
    votes = []
    for item_idx in range(n_items):
        votes.append(majority_label([output[item_idx] for output in judge_outputs]))
    return votes
