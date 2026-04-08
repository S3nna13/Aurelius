"""Replay scheduling strategies for continual pretraining."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayTask:
    task_id: str
    loss: float
    age: int
    token_budget: int


def replay_priority(task: ReplayTask, loss_weight: float = 1.0, age_weight: float = 0.1) -> float:
    """Priority score for replaying a task."""
    return loss_weight * task.loss + age_weight * max(task.age, 0)


def allocate_replay_budget(tasks: list[ReplayTask], total_budget: int) -> dict[str, int]:
    """Allocate replay tokens proportional to task priority."""
    if total_budget < 0:
        raise ValueError(f"total_budget must be non-negative, got {total_budget}")
    if not tasks:
        return {}
    priorities = [max(replay_priority(task), 0.0) for task in tasks]
    total_priority = sum(priorities)
    if total_priority == 0.0:
        equal = total_budget // len(tasks)
        return {task.task_id: min(equal, task.token_budget) for task in tasks}

    remaining = total_budget
    allocation: dict[str, int] = {}
    for index, task in enumerate(tasks):
        if index == len(tasks) - 1:
            share = remaining
        else:
            share = int(round(total_budget * priorities[index] / total_priority))
            share = min(share, remaining)
        allocation[task.task_id] = min(share, task.token_budget)
        remaining -= allocation[task.task_id]
    return allocation


def select_replay_tasks(tasks: list[ReplayTask], k: int) -> list[ReplayTask]:
    """Select the highest-priority replay tasks."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    return sorted(tasks, key=replay_priority, reverse=True)[:k]
