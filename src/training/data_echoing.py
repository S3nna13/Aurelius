"""Data echoing utilities for repeating informative examples during training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EchoExample:
    example_id: str
    loss: float
    difficulty: float
    last_seen_step: int
    echo_count: int = 0


def echo_score(
    loss: torch.Tensor,
    difficulty: torch.Tensor,
    age: torch.Tensor,
    loss_weight: float = 1.0,
    difficulty_weight: float = 0.5,
    recency_weight: float = 0.1,
) -> torch.Tensor:
    """Score examples for replay based on loss, difficulty, and recency."""
    if loss.shape != difficulty.shape or loss.shape != age.shape:
        raise ValueError("loss, difficulty, and age must share the same shape")
    recency_bonus = torch.log1p(age.clamp_min(0))
    return loss_weight * loss + difficulty_weight * difficulty + recency_weight * recency_bonus


def echo_probabilities(
    examples: list[EchoExample],
    current_step: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Turn example scores into replay probabilities."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not examples:
        return torch.empty(0)
    losses = torch.tensor([example.loss for example in examples], dtype=torch.float32)
    difficulty = torch.tensor([example.difficulty for example in examples], dtype=torch.float32)
    age = torch.tensor([current_step - example.last_seen_step for example in examples], dtype=torch.float32)
    scores = echo_score(losses, difficulty, age)
    return torch.softmax(scores / temperature, dim=0)


def select_echoes(
    examples: list[EchoExample],
    current_step: int,
    n_select: int,
) -> list[EchoExample]:
    """Select the top-scoring examples to replay."""
    if n_select < 0:
        raise ValueError(f"n_select must be non-negative, got {n_select}")
    if not examples or n_select == 0:
        return []
    probs = echo_probabilities(examples, current_step)
    k = min(n_select, len(examples))
    indices = torch.topk(probs, k=k).indices.tolist()
    return [examples[index] for index in indices]


def update_echo_metadata(
    examples: list[EchoExample],
    echoed_ids: list[str],
    current_step: int,
) -> None:
    """Update replay counters after a batch uses echoed examples."""
    echoed = set(echoed_ids)
    for example in examples:
        if example.example_id in echoed:
            example.echo_count += 1
            example.last_seen_step = current_step
