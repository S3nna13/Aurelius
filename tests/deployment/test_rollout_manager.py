"""Tests for src/deployment/rollout_manager.py."""

from __future__ import annotations

import pytest

from src.deployment.rollout_manager import (
    CANARY_3_STAGE,
    ROLLOUT_REGISTRY,
    RolloutManager,
    RolloutPlan,
    RolloutStage,
    RolloutStrategy,
)


# ---------------------------------------------------------------------------
# RolloutStrategy enum
# ---------------------------------------------------------------------------

class TestRolloutStrategy:
    def test_canary_value(self):
        assert RolloutStrategy.CANARY == "canary"

    def test_blue_green_value(self):
        assert RolloutStrategy.BLUE_GREEN == "blue_green"

    def test_rolling_value(self):
        assert RolloutStrategy.ROLLING == "rolling"

    def test_is_str_subclass(self):
        assert isinstance(RolloutStrategy.CANARY, str)

    def test_members_count(self):
        assert len(RolloutStrategy) == 3


# ---------------------------------------------------------------------------
# RolloutStage dataclass
# ---------------------------------------------------------------------------

class TestRolloutStage:
    def test_name_and_traffic(self):
        s = RolloutStage("10pct", 10.0)
        assert s.name == "10pct"
        assert s.traffic_pct == 10.0

    def test_default_min_healthy_pct(self):
        s = RolloutStage("x", 50.0)
        assert s.min_healthy_pct == 95.0

    def test_default_duration_seconds(self):
        s = RolloutStage("x", 50.0)
        assert s.duration_seconds == 300

    def test_custom_min_healthy(self):
        s = RolloutStage("x", 50.0, min_healthy_pct=80.0)
        assert s.min_healthy_pct == 80.0

    def test_custom_duration(self):
        s = RolloutStage("x", 50.0, duration_seconds=60)
        assert s.duration_seconds == 60


# ---------------------------------------------------------------------------
# RolloutPlan dataclass
# ---------------------------------------------------------------------------

class TestRolloutPlan:
    def test_strategy_and_stages(self):
        stages = [RolloutStage("a", 10.0)]
        plan = RolloutPlan(strategy=RolloutStrategy.CANARY, stages=stages)
        assert plan.strategy == RolloutStrategy.CANARY
        assert plan.stages is stages

    def test_rollback_on_error_default(self):
        plan = RolloutPlan(RolloutStrategy.ROLLING, [])
        assert plan.rollback_on_error is True

    def test_rollback_on_error_false(self):
        plan = RolloutPlan(RolloutStrategy.ROLLING, [], rollback_on_error=False)
        assert plan.rollback_on_error is False


# ---------------------------------------------------------------------------
# RolloutManager
# ---------------------------------------------------------------------------

def _make_canary_manager(num_stages: int = 3) -> RolloutManager:
    stages = [RolloutStage(f"s{i}", float(i * 10)) for i in range(1, num_stages + 1)]
    plan = RolloutPlan(strategy=RolloutStrategy.CANARY, stages=stages)
    return RolloutManager(plan)


def _make_bg_manager() -> RolloutManager:
    plan = RolloutPlan(
        strategy=RolloutStrategy.BLUE_GREEN,
        stages=[RolloutStage("pre", 50.0), RolloutStage("cutover", 100.0)],
    )
    return RolloutManager(plan)


class TestRolloutManagerCurrentStage:
    def test_returns_first_stage_initially(self):
        mgr = _make_canary_manager()
        assert mgr.current_stage() is mgr._plan.stages[0]

    def test_name_of_first_stage(self):
        mgr = _make_canary_manager()
        assert mgr.current_stage().name == "s1"

    def test_empty_plan_returns_none(self):
        plan = RolloutPlan(RolloutStrategy.CANARY, [])
        mgr = RolloutManager(plan)
        assert mgr.current_stage() is None


class TestRolloutManagerAdvance:
    def test_advance_returns_true_when_moved(self):
        mgr = _make_canary_manager(3)
        assert mgr.advance() is True

    def test_advance_moves_to_next_stage(self):
        mgr = _make_canary_manager(3)
        mgr.advance()
        assert mgr.current_stage().name == "s2"

    def test_advance_returns_false_at_end(self):
        mgr = _make_canary_manager(1)
        result = mgr.advance()
        assert result is False

    def test_advance_stays_at_end(self):
        mgr = _make_canary_manager(2)
        mgr.advance()
        mgr.advance()  # already at end
        assert mgr.current_stage().name == "s2"

    def test_advance_through_all_stages(self):
        mgr = _make_canary_manager(3)
        assert mgr.advance() is True
        assert mgr.advance() is True
        assert mgr.advance() is False

    def test_advance_three_stages_current_stage(self):
        mgr = _make_canary_manager(3)
        mgr.advance()
        mgr.advance()
        assert mgr.current_stage().name == "s3"


class TestRolloutManagerRollback:
    def test_rollback_resets_to_first_stage(self):
        mgr = _make_canary_manager(3)
        mgr.advance()
        mgr.advance()
        mgr.rollback()
        assert mgr.current_stage().name == "s1"

    def test_rollback_from_beginning_stays_at_beginning(self):
        mgr = _make_canary_manager(3)
        mgr.rollback()
        assert mgr.current_stage().name == "s1"

    def test_rollback_index_is_zero(self):
        mgr = _make_canary_manager(3)
        mgr.advance()
        mgr.rollback()
        assert mgr._stage_index == 0


class TestRolloutManagerTrafficSplit:
    def test_canary_split_sums_to_one(self):
        mgr = _make_canary_manager(3)
        split = mgr.traffic_split()
        assert abs(split["stable"] + split["canary"] - 1.0) < 1e-9

    def test_canary_first_stage_10pct(self):
        stages = [RolloutStage("10pct", 10.0)]
        plan = RolloutPlan(RolloutStrategy.CANARY, stages)
        mgr = RolloutManager(plan)
        split = mgr.traffic_split()
        assert abs(split["canary"] - 0.10) < 1e-9
        assert abs(split["stable"] - 0.90) < 1e-9

    def test_canary_100pct_stage(self):
        stages = [RolloutStage("5pct", 5.0), RolloutStage("100pct", 100.0)]
        plan = RolloutPlan(RolloutStrategy.CANARY, stages)
        mgr = RolloutManager(plan)
        mgr.advance()
        split = mgr.traffic_split()
        assert abs(split["canary"] - 1.0) < 1e-9
        assert abs(split["stable"] - 0.0) < 1e-9

    def test_blue_green_start_blue_gets_all(self):
        mgr = _make_bg_manager()
        split = mgr.traffic_split()
        assert split["blue"] == 1.0
        assert split["green"] == 0.0

    def test_blue_green_final_stage_green_gets_all(self):
        mgr = _make_bg_manager()
        mgr.advance()  # move to final stage
        split = mgr.traffic_split()
        assert split["blue"] == 0.0
        assert split["green"] == 1.0

    def test_blue_green_keys(self):
        mgr = _make_bg_manager()
        split = mgr.traffic_split()
        assert set(split.keys()) == {"blue", "green"}

    def test_canary_keys(self):
        mgr = _make_canary_manager()
        split = mgr.traffic_split()
        assert set(split.keys()) == {"stable", "canary"}


class TestRolloutManagerProgress:
    def test_initial_stage_index_is_zero(self):
        mgr = _make_canary_manager(3)
        p = mgr.progress()
        assert p["stage_index"] == 0

    def test_total_stages(self):
        mgr = _make_canary_manager(3)
        p = mgr.progress()
        assert p["total_stages"] == 3

    def test_strategy_string(self):
        mgr = _make_canary_manager(3)
        p = mgr.progress()
        assert p["strategy"] == "canary"

    def test_current_stage_name(self):
        mgr = _make_canary_manager(3)
        p = mgr.progress()
        assert p["current_stage_name"] == "s1"

    def test_progress_after_advance(self):
        mgr = _make_canary_manager(3)
        mgr.advance()
        p = mgr.progress()
        assert p["stage_index"] == 1
        assert p["current_stage_name"] == "s2"

    def test_blue_green_strategy_string(self):
        mgr = _make_bg_manager()
        p = mgr.progress()
        assert p["strategy"] == "blue_green"


# ---------------------------------------------------------------------------
# Registry and pre-built plans
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_canary_3stage(self):
        assert "canary_3stage" in ROLLOUT_REGISTRY

    def test_registry_has_blue_green(self):
        assert "blue_green" in ROLLOUT_REGISTRY

    def test_canary_3stage_is_rollout_plan(self):
        assert isinstance(ROLLOUT_REGISTRY["canary_3stage"], RolloutPlan)

    def test_blue_green_is_rollout_plan(self):
        assert isinstance(ROLLOUT_REGISTRY["blue_green"], RolloutPlan)


class TestCanary3Stage:
    def test_has_three_stages(self):
        assert len(CANARY_3_STAGE.stages) == 3

    def test_strategy_is_canary(self):
        assert CANARY_3_STAGE.strategy == RolloutStrategy.CANARY

    def test_first_stage_name(self):
        assert CANARY_3_STAGE.stages[0].name == "5pct"

    def test_first_stage_traffic(self):
        assert CANARY_3_STAGE.stages[0].traffic_pct == 5.0

    def test_second_stage_name(self):
        assert CANARY_3_STAGE.stages[1].name == "25pct"

    def test_third_stage_name(self):
        assert CANARY_3_STAGE.stages[2].name == "100pct"

    def test_third_stage_traffic(self):
        assert CANARY_3_STAGE.stages[2].traffic_pct == 100.0

    def test_registry_canary_3stage_is_canary_3stage(self):
        assert ROLLOUT_REGISTRY["canary_3stage"] is CANARY_3_STAGE
