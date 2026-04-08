"""Curriculum sampling schedules for staged difficulty exposure."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DifficultyBucket:
    name: str
    difficulty: float
    weight: float


def curriculum_progress(step: int, total_steps: int) -> float:
    """Normalized curriculum progress in [0, 1]."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    return min(max(step / total_steps, 0.0), 1.0)


def curriculum_weights(
    buckets: list[DifficultyBucket],
    step: int,
    total_steps: int,
) -> dict[str, float]:
    """Increase focus on harder buckets as training progresses."""
    if not buckets:
        return {}
    progress = curriculum_progress(step, total_steps)
    raw = torch.tensor(
        [bucket.weight * (1.0 + progress * bucket.difficulty) for bucket in buckets],
        dtype=torch.float32,
    )
    probs = raw / raw.sum()
    return {bucket.name: prob.item() for bucket, prob in zip(buckets, probs)}


def sample_bucket_order(
    buckets: list[DifficultyBucket],
    step: int,
    total_steps: int,
) -> list[str]:
    """Return buckets ordered from most to least likely at a given step."""
    weights = curriculum_weights(buckets, step, total_steps)
    return sorted(weights, key=weights.get, reverse=True)


def allocate_curriculum_budget(
    buckets: list[DifficultyBucket],
    step: int,
    total_steps: int,
    total_examples: int,
) -> dict[str, int]:
    """Allocate integer sample counts across curriculum buckets."""
    if total_examples < 0:
        raise ValueError(f"total_examples must be non-negative, got {total_examples}")
    weights = curriculum_weights(buckets, step, total_steps)
    names = [bucket.name for bucket in buckets]
    remaining = total_examples
    allocation: dict[str, int] = {}
    for index, name in enumerate(names):
        if index == len(names) - 1:
            allocation[name] = remaining
        else:
            share = int(round(total_examples * weights[name]))
            share = min(share, remaining)
            allocation[name] = share
            remaining -= share
    return allocation
