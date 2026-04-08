"""Evidence ranking helpers for retrieval and faithfulness analysis."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EvidenceItem:
    item_id: str
    score: float
    relevant: bool


def rank_evidence(scores: torch.Tensor) -> torch.Tensor:
    """Return descending ranking indices for evidence scores."""
    if scores.dim() != 1:
        raise ValueError("scores must be 1D")
    return torch.argsort(scores, descending=True)


def topk_precision(items: list[EvidenceItem], k: int) -> float:
    """Precision at k over already-scored evidence items."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not items:
        return 0.0
    ranked = sorted(items, key=lambda item: item.score, reverse=True)[:k]
    return sum(item.relevant for item in ranked) / len(ranked)


def mean_reciprocal_rank(items: list[EvidenceItem]) -> float:
    """MRR for a ranked evidence list."""
    ranked = sorted(items, key=lambda item: item.score, reverse=True)
    for rank, item in enumerate(ranked, start=1):
        if item.relevant:
            return 1.0 / rank
    return 0.0

