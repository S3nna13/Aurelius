"""Token-level replay bookkeeping for continual training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TokenReplayRecord:
    token_id: int
    loss: float
    count: int = 0


def replay_weight(losses: torch.Tensor, counts: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Replay weight favoring high-loss and under-seen tokens."""
    if losses.shape != counts.shape:
        raise ValueError("losses and counts must match")
    return losses * (counts.to(dtype=losses.dtype) + 1.0).pow(-alpha)


def replay_distribution(records: list[TokenReplayRecord], alpha: float = 1.0) -> torch.Tensor:
    """Convert token replay records into a sampling distribution."""
    if not records:
        return torch.empty(0)
    losses = torch.tensor([record.loss for record in records], dtype=torch.float32)
    counts = torch.tensor([record.count for record in records], dtype=torch.float32)
    weights = replay_weight(losses, counts, alpha=alpha)
    return torch.softmax(weights, dim=0)


def select_replay_tokens(records: list[TokenReplayRecord], k: int) -> list[int]:
    """Select the top-k token ids by replay weight."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if not records:
        return []
    probs = replay_distribution(records)
    indices = torch.topk(probs, k=min(k, len(records))).indices.tolist()
    return [records[index].token_id for index in indices]


def update_replay_counts(records: list[TokenReplayRecord], token_ids: list[int]) -> None:
    """Increment counts for replayed token ids."""
    selected = set(token_ids)
    for record in records:
        if record.token_id in selected:
            record.count += 1
