"""Tests for replay scheduling strategies."""

import pytest

from src.training.replay_strategy import (
    ReplayTask,
    allocate_replay_budget,
    replay_priority,
    select_replay_tasks,
)


def make_tasks():
    return [
        ReplayTask("task_a", loss=0.4, age=1, token_budget=100),
        ReplayTask("task_b", loss=1.2, age=5, token_budget=100),
        ReplayTask("task_c", loss=0.8, age=2, token_budget=20),
    ]


def test_replay_priority_increases_with_loss_and_age():
    low = replay_priority(ReplayTask("a", loss=0.2, age=1, token_budget=10))
    high = replay_priority(ReplayTask("b", loss=1.0, age=3, token_budget=10))
    assert high > low


def test_select_replay_tasks_returns_highest_priority():
    selected = select_replay_tasks(make_tasks(), 2)
    assert [task.task_id for task in selected] == ["task_b", "task_c"]


def test_allocate_replay_budget_respects_total_budget():
    allocation = allocate_replay_budget(make_tasks(), total_budget=60)
    assert sum(allocation.values()) <= 60


def test_allocate_replay_budget_respects_task_caps():
    allocation = allocate_replay_budget(make_tasks(), total_budget=500)
    assert allocation["task_c"] <= 20


def test_allocate_replay_budget_handles_empty_input():
    assert allocate_replay_budget([], total_budget=10) == {}


def test_allocate_replay_budget_rejects_negative_budget():
    with pytest.raises(ValueError):
        allocate_replay_budget(make_tasks(), total_budget=-1)


def test_select_replay_tasks_rejects_negative_k():
    with pytest.raises(ValueError):
        select_replay_tasks(make_tasks(), -1)
