"""Tests for continual strategy helpers."""

import pytest

from src.training.continual_strategy import (
    build_continual_plan,
    linear_ewc_schedule,
    replay_ratio_schedule,
    stage_token_budget,
)


def test_linear_ewc_schedule_scales_by_growth():
    assert linear_ewc_schedule(10.0, 2, growth=2.0) == pytest.approx(40.0)


def test_replay_ratio_schedule_increases_with_stage():
    early = replay_ratio_schedule(0, 10, 0.0, 0.3)
    late = replay_ratio_schedule(10, 10, 0.0, 0.3)
    assert late > early


def test_stage_token_budget_distributes_remainder():
    budgets = [stage_token_budget(10, 3, idx) for idx in range(3)]
    assert sum(budgets) == 10


def test_build_continual_plan_returns_all_tasks():
    plan = build_continual_plan(["a", "b", "c"], total_tokens=9, base_lambda=1.0)
    assert [stage.task_id for stage in plan] == ["a", "b", "c"]


def test_build_continual_plan_handles_empty_tasks():
    assert build_continual_plan([], total_tokens=10, base_lambda=1.0) == []


def test_linear_ewc_schedule_rejects_bad_inputs():
    with pytest.raises(ValueError):
        linear_ewc_schedule(-1.0, 0)


def test_replay_ratio_schedule_rejects_bad_ranges():
    with pytest.raises(ValueError):
        replay_ratio_schedule(0, 0)
    with pytest.raises(ValueError):
        replay_ratio_schedule(0, 10, 0.5, 0.4)

