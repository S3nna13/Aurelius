"""Tests for src/deployment/canary_controller.py — ≥28 test cases."""

from __future__ import annotations

import dataclasses
import pytest

from src.deployment.canary_controller import (
    CANARY_CONTROLLER_REGISTRY,
    DEFAULT_STAGES,
    CanaryController,
    CanaryStage,
    CanaryState,
)


# ---------------------------------------------------------------------------
# CanaryStage dataclass
# ---------------------------------------------------------------------------

class TestCanaryStage:
    def test_fields_set(self):
        s = CanaryStage(10.0, 30.0)
        assert s.traffic_pct == 10.0
        assert s.min_duration_s == 30.0

    def test_default_success_threshold(self):
        s = CanaryStage(5.0, 60.0)
        assert s.success_threshold == 0.99

    def test_custom_success_threshold(self):
        s = CanaryStage(5.0, 60.0, success_threshold=0.95)
        assert s.success_threshold == 0.95

    def test_frozen_traffic_pct(self):
        s = CanaryStage(5.0, 60.0)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            s.traffic_pct = 99.0  # type: ignore[misc]

    def test_frozen_min_duration_s(self):
        s = CanaryStage(5.0, 60.0)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            s.min_duration_s = 0.0  # type: ignore[misc]

    def test_frozen_success_threshold(self):
        s = CanaryStage(5.0, 60.0)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            s.success_threshold = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Default stages
# ---------------------------------------------------------------------------

class TestDefaultStages:
    def test_default_stages_count(self):
        assert len(DEFAULT_STAGES) == 4

    def test_default_first_stage_traffic(self):
        assert DEFAULT_STAGES[0].traffic_pct == 5.0

    def test_default_last_stage_traffic(self):
        assert DEFAULT_STAGES[-1].traffic_pct == 100.0

    def test_default_last_stage_min_duration(self):
        assert DEFAULT_STAGES[-1].min_duration_s == 0.0


# ---------------------------------------------------------------------------
# CanaryController — initial state
# ---------------------------------------------------------------------------

class TestCanaryControllerInit:
    def test_starts_at_stage_zero(self):
        cc = CanaryController()
        assert cc.current_stage_idx == 0

    def test_starts_running(self):
        cc = CanaryController()
        assert cc.state == CanaryState.RUNNING

    def test_default_stages_used_when_none_provided(self):
        cc = CanaryController()
        assert len(cc._stages) == 4

    def test_custom_stages_stored(self):
        stages = [CanaryStage(10.0, 30.0), CanaryStage(100.0, 0.0)]
        cc = CanaryController(stages=stages)
        assert len(cc._stages) == 2


# ---------------------------------------------------------------------------
# CanaryController — advance (success path)
# ---------------------------------------------------------------------------

class TestCanaryControllerAdvanceSuccess:
    def test_advance_success_moves_to_next_stage(self):
        cc = CanaryController()
        cc.advance(success_rate=1.0)
        assert cc.current_stage_idx == 1

    def test_advance_success_stays_running(self):
        cc = CanaryController()
        state = cc.advance(success_rate=1.0)
        assert state == CanaryState.RUNNING

    def test_advance_at_threshold_boundary(self):
        cc = CanaryController(stages=[CanaryStage(5.0, 60.0, success_threshold=0.99)])
        # Exactly at threshold — should succeed
        state = cc.advance(success_rate=0.99)
        assert state == CanaryState.SUCCEEDED

    def test_advance_all_stages_succeeds(self):
        stages = [
            CanaryStage(5.0, 60.0),
            CanaryStage(25.0, 120.0),
            CanaryStage(100.0, 0.0),
        ]
        cc = CanaryController(stages=stages)
        cc.advance(success_rate=1.0)
        cc.advance(success_rate=1.0)
        state = cc.advance(success_rate=1.0)
        assert state == CanaryState.SUCCEEDED

    def test_advance_last_stage_state_is_succeeded(self):
        cc = CanaryController(stages=[CanaryStage(100.0, 0.0)])
        state = cc.advance(success_rate=1.0)
        assert state == CanaryState.SUCCEEDED

    def test_state_does_not_change_after_succeeded(self):
        cc = CanaryController(stages=[CanaryStage(100.0, 0.0)])
        cc.advance(success_rate=1.0)
        state = cc.advance(success_rate=1.0)
        assert state == CanaryState.SUCCEEDED


# ---------------------------------------------------------------------------
# CanaryController — advance (failure path)
# ---------------------------------------------------------------------------

class TestCanaryControllerAdvanceFailure:
    def test_advance_below_threshold_sets_failed(self):
        cc = CanaryController()
        state = cc.advance(success_rate=0.50)
        assert state == CanaryState.FAILED

    def test_advance_just_below_threshold_sets_failed(self):
        cc = CanaryController()
        state = cc.advance(success_rate=0.9899)
        assert state == CanaryState.FAILED

    def test_state_after_failure_is_failed(self):
        cc = CanaryController()
        cc.advance(success_rate=0.0)
        assert cc.state == CanaryState.FAILED

    def test_advance_no_op_when_already_failed(self):
        cc = CanaryController()
        cc.advance(success_rate=0.0)
        state = cc.advance(success_rate=1.0)
        assert state == CanaryState.FAILED
        assert cc.current_stage_idx == 0


# ---------------------------------------------------------------------------
# CanaryController — rollback
# ---------------------------------------------------------------------------

class TestCanaryControllerRollback:
    def test_rollback_sets_rolled_back(self):
        cc = CanaryController()
        state = cc.rollback()
        assert state == CanaryState.ROLLED_BACK

    def test_rollback_after_advance(self):
        cc = CanaryController()
        cc.advance(success_rate=1.0)
        state = cc.rollback()
        assert state == CanaryState.ROLLED_BACK

    def test_rollback_state_attribute(self):
        cc = CanaryController()
        cc.rollback()
        assert cc.state == CanaryState.ROLLED_BACK


# ---------------------------------------------------------------------------
# CanaryController — traffic_pct
# ---------------------------------------------------------------------------

class TestCanaryControllerTrafficPct:
    def test_traffic_pct_initial(self):
        cc = CanaryController()
        assert cc.traffic_pct() == 5.0

    def test_traffic_pct_after_advance(self):
        cc = CanaryController()
        cc.advance(success_rate=1.0)
        assert cc.traffic_pct() == 25.0

    def test_traffic_pct_when_failed(self):
        cc = CanaryController()
        cc.advance(success_rate=0.0)
        assert cc.traffic_pct() == 0.0

    def test_traffic_pct_when_rolled_back(self):
        cc = CanaryController()
        cc.rollback()
        assert cc.traffic_pct() == 0.0

    def test_traffic_pct_100_at_last_stage(self):
        stages = [CanaryStage(50.0, 60.0), CanaryStage(100.0, 0.0)]
        cc = CanaryController(stages=stages)
        cc.advance(success_rate=1.0)
        assert cc.traffic_pct() == 100.0


# ---------------------------------------------------------------------------
# CanaryController — summary
# ---------------------------------------------------------------------------

class TestCanaryControllerSummary:
    def test_summary_has_required_keys(self):
        cc = CanaryController()
        s = cc.summary()
        assert set(s.keys()) == {"state", "stage", "traffic_pct", "total_stages"}

    def test_summary_initial_state(self):
        cc = CanaryController()
        s = cc.summary()
        assert s["state"] == "RUNNING"
        assert s["stage"] == 0
        assert s["traffic_pct"] == 5.0
        assert s["total_stages"] == 4

    def test_summary_after_rollback(self):
        cc = CanaryController()
        cc.rollback()
        s = cc.summary()
        assert s["state"] == "ROLLED_BACK"
        assert s["traffic_pct"] == 0.0

    def test_summary_state_is_string(self):
        cc = CanaryController()
        assert isinstance(cc.summary()["state"], str)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default_key(self):
        assert "default" in CANARY_CONTROLLER_REGISTRY

    def test_registry_default_is_canary_controller_class(self):
        assert CANARY_CONTROLLER_REGISTRY["default"] is CanaryController

    def test_registry_default_is_instantiable(self):
        cls = CANARY_CONTROLLER_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, CanaryController)
