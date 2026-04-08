"""Scheduling strategies for continual pretraining."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContinualStage:
    task_id: str
    token_budget: int
    replay_ratio: float
    ewc_lambda: float


def linear_ewc_schedule(base_lambda: float, stage_idx: int, growth: float = 1.0) -> float:
    """Scale EWC regularization across continual stages."""
    if base_lambda < 0 or growth < 0:
        raise ValueError("base_lambda and growth must be non-negative")
    if stage_idx < 0:
        raise ValueError("stage_idx must be non-negative")
    return base_lambda * (growth ** stage_idx)


def replay_ratio_schedule(stage_idx: int, n_stages: int, min_ratio: float = 0.0, max_ratio: float = 0.3) -> float:
    """Linearly increase replay ratio across stages."""
    if n_stages <= 0:
        raise ValueError("n_stages must be positive")
    if not (0.0 <= min_ratio <= max_ratio <= 1.0):
        raise ValueError("Expected 0 <= min_ratio <= max_ratio <= 1")
    progress = min(max(stage_idx / n_stages, 0.0), 1.0)
    return min_ratio + progress * (max_ratio - min_ratio)


def stage_token_budget(total_tokens: int, n_stages: int, stage_idx: int) -> int:
    """Allocate total tokens approximately evenly across stages."""
    if total_tokens < 0 or n_stages <= 0:
        raise ValueError("total_tokens must be non-negative and n_stages positive")
    base = total_tokens // n_stages
    remainder = total_tokens % n_stages
    return base + int(stage_idx < remainder)


def build_continual_plan(
    task_ids: list[str],
    total_tokens: int,
    base_lambda: float,
    growth: float = 1.0,
    min_replay_ratio: float = 0.0,
    max_replay_ratio: float = 0.3,
) -> list[ContinualStage]:
    """Create a per-stage continual-training plan."""
    if not task_ids:
        return []
    n_stages = len(task_ids)
    plan = []
    for stage_idx, task_id in enumerate(task_ids):
        plan.append(
            ContinualStage(
                task_id=task_id,
                token_budget=stage_token_budget(total_tokens, n_stages, stage_idx),
                replay_ratio=replay_ratio_schedule(stage_idx, n_stages, min_replay_ratio, max_replay_ratio),
                ewc_lambda=linear_ewc_schedule(base_lambda, stage_idx, growth),
            )
        )
    return plan

