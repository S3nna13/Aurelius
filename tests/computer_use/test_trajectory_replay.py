"""Tests for src/computer_use/trajectory_replay.py.

Inspired by OpenDevin/OpenDevin (Apache-2.0, browser tool), MoonshotAI/Kimi-Dev
(Apache-2.0, patch synthesis), WebArena trajectory replay, clean-room reimplementation.
"""

from __future__ import annotations

import pytest

from src.computer_use.browser_driver import StubBrowserDriver
from src.computer_use.gui_action import ActionType, GUIAction
from src.computer_use.screen_parser import AccessibilityNode, ScreenSnapshot
from src.computer_use.trajectory_replay import (
    TRAJECTORY_REGISTRY,
    Trajectory,
    TrajectoryError,
    TrajectoryRecorder,
    TrajectoryReplayer,
    TrajectoryStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(label: str = "root") -> ScreenSnapshot:
    return ScreenSnapshot(
        width=1280,
        height=800,
        root_node=AccessibilityNode(role="application", name=label),
    )


def _make_click_action(selector: str = "#btn") -> GUIAction:
    return GUIAction(action_type=ActionType.CLICK, target_selector=selector)


def _make_type_action(selector: str = "#input", value: str = "hello") -> GUIAction:
    return GUIAction(action_type=ActionType.TYPE, target_selector=selector, value=value)


# ---------------------------------------------------------------------------
# TrajectoryStep dataclass
# ---------------------------------------------------------------------------

class TestTrajectoryStep:
    def test_step_id_is_int(self):
        step = TrajectoryStep(step_id=0, action=_make_click_action())
        assert isinstance(step.step_id, int)

    def test_step_defaults(self):
        step = TrajectoryStep(step_id=1, action=_make_click_action())
        assert step.snapshot_before is None
        assert step.snapshot_after is None
        assert isinstance(step.metadata, dict)
        assert isinstance(step.timestamp, float)


# ---------------------------------------------------------------------------
# Trajectory dataclass
# ---------------------------------------------------------------------------

class TestTrajectory:
    def test_success_defaults_none(self):
        traj = Trajectory(trajectory_id="t1", goal="test")
        assert traj.success is None

    def test_total_duration_defaults_zero(self):
        traj = Trajectory(trajectory_id="t1", goal="test")
        assert traj.total_duration_s == 0.0

    def test_steps_defaults_empty(self):
        traj = Trajectory(trajectory_id="t1", goal="test")
        assert traj.steps == []


# ---------------------------------------------------------------------------
# TrajectoryRecorder
# ---------------------------------------------------------------------------

class TestTrajectoryRecorder:
    def test_record_step_first_step_id_is_zero(self):
        recorder = TrajectoryRecorder(goal="click button")
        step = recorder.record_step(_make_click_action())
        assert step.step_id == 0

    def test_record_step_increments_step_id(self):
        recorder = TrajectoryRecorder(goal="multi-step")
        s0 = recorder.record_step(_make_click_action("#a"))
        s1 = recorder.record_step(_make_click_action("#b"))
        s2 = recorder.record_step(_make_click_action("#c"))
        assert s0.step_id == 0
        assert s1.step_id == 1
        assert s2.step_id == 2

    def test_record_step_stores_snapshots(self):
        recorder = TrajectoryRecorder(goal="test")
        snap = _make_snapshot()
        step = recorder.record_step(_make_click_action(), snapshot_before=snap)
        assert step.snapshot_before is snap

    def test_finalize_returns_trajectory(self):
        recorder = TrajectoryRecorder(goal="finish")
        recorder.record_step(_make_click_action())
        traj = recorder.finalize(success=True)
        assert isinstance(traj, Trajectory)

    def test_finalize_success_true(self):
        recorder = TrajectoryRecorder(goal="finish")
        recorder.record_step(_make_click_action())
        traj = recorder.finalize(success=True)
        assert traj.success is True

    def test_finalize_success_false(self):
        recorder = TrajectoryRecorder(goal="finish")
        recorder.record_step(_make_click_action())
        traj = recorder.finalize(success=False)
        assert traj.success is False

    def test_finalize_preserves_steps(self):
        recorder = TrajectoryRecorder(goal="two steps")
        recorder.record_step(_make_click_action("#a"))
        recorder.record_step(_make_click_action("#b"))
        traj = recorder.finalize(success=True)
        assert len(traj.steps) == 2

    def test_finalize_goal_preserved(self):
        recorder = TrajectoryRecorder(goal="my goal")
        recorder.record_step(_make_click_action())
        traj = recorder.finalize(success=True)
        assert traj.goal == "my goal"

    def test_finalize_duration_positive(self):
        recorder = TrajectoryRecorder(goal="speed")
        recorder.record_step(_make_click_action())
        traj = recorder.finalize(success=True)
        assert traj.total_duration_s >= 0.0

    def test_reset_clears_steps(self):
        recorder = TrajectoryRecorder(goal="reset test")
        recorder.record_step(_make_click_action())
        recorder.reset()
        traj = recorder.finalize(success=True)
        assert len(traj.steps) == 0

    def test_reset_step_ids_restart_from_zero(self):
        recorder = TrajectoryRecorder(goal="reset ids")
        recorder.record_step(_make_click_action("#a"))
        recorder.record_step(_make_click_action("#b"))
        recorder.reset()
        step = recorder.record_step(_make_click_action("#c"))
        assert step.step_id == 0

    def test_success_none_before_finalize(self):
        """Trajectory.success should be None before finalize is called."""
        # Build a Trajectory manually (not via recorder) to test the default.
        traj = Trajectory(trajectory_id="raw", goal="check default")
        assert traj.success is None


# ---------------------------------------------------------------------------
# TrajectoryReplayer
# ---------------------------------------------------------------------------

class TestTrajectoryReplayer:
    def _make_trajectory(self, n: int = 2) -> Trajectory:
        recorder = TrajectoryRecorder(goal="replay test")
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        for i in range(n):
            snap = _make_snapshot(f"snap-{i}")
            action = _make_click_action(f"#btn-{i}")
            recorder.record_step(action, snapshot_before=snap)
        return recorder.finalize(success=True)

    def test_replay_returns_list(self):
        traj = self._make_trajectory(2)
        replayer = TrajectoryReplayer()
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        statuses = replayer.replay(traj, driver)
        assert isinstance(statuses, list)

    def test_replay_status_count_equals_steps(self):
        traj = self._make_trajectory(3)
        replayer = TrajectoryReplayer()
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        statuses = replayer.replay(traj, driver)
        assert len(statuses) == 3

    def test_replay_all_ok(self):
        traj = self._make_trajectory(2)
        replayer = TrajectoryReplayer()
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        statuses = replayer.replay(traj, driver)
        assert all("ok" in s for s in statuses)

    def test_replay_status_labels_match_step_ids(self):
        traj = self._make_trajectory(2)
        replayer = TrajectoryReplayer()
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        statuses = replayer.replay(traj, driver)
        assert statuses[0].startswith("step-0")
        assert statuses[1].startswith("step-1")

    def test_replay_type_action(self):
        recorder = TrajectoryRecorder(goal="type test")
        recorder.record_step(_make_type_action("#input", "world"))
        traj = recorder.finalize(success=True)
        replayer = TrajectoryReplayer()
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        statuses = replayer.replay(traj, driver)
        assert "ok" in statuses[0]

    def test_replay_scroll_is_noop(self):
        recorder = TrajectoryRecorder(goal="scroll test")
        scroll_action = GUIAction(action_type=ActionType.SCROLL)
        recorder.record_step(scroll_action)
        traj = recorder.finalize(success=True)
        replayer = TrajectoryReplayer()
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        statuses = replayer.replay(traj, driver)
        assert "ok" in statuses[0]

    def test_replay_key_is_noop(self):
        recorder = TrajectoryRecorder(goal="key test")
        key_action = GUIAction(action_type=ActionType.KEY)
        recorder.record_step(key_action)
        traj = recorder.finalize(success=True)
        replayer = TrajectoryReplayer()
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        statuses = replayer.replay(traj, driver)
        assert "ok" in statuses[0]

    def test_replay_empty_trajectory_returns_empty_list(self):
        traj = Trajectory(trajectory_id="empty", goal="nothing", steps=[])
        replayer = TrajectoryReplayer()
        driver = StubBrowserDriver()
        statuses = replayer.replay(traj, driver)
        assert statuses == []


# ---------------------------------------------------------------------------
# TrajectoryReplayer.verify
# ---------------------------------------------------------------------------

class TestTrajectoryReplayerVerify:
    def test_verify_zero_steps_returns_false(self):
        traj = Trajectory(trajectory_id="empty", goal="nothing", steps=[])
        replayer = TrajectoryReplayer()
        ok, issues = replayer.verify(traj)
        assert ok is False
        assert len(issues) > 0

    def test_verify_zero_steps_issues_non_empty(self):
        traj = Trajectory(trajectory_id="empty", goal="nothing", steps=[])
        replayer = TrajectoryReplayer()
        _, issues = replayer.verify(traj)
        assert isinstance(issues, list)
        assert len(issues) >= 1

    def test_verify_valid_two_step_trajectory_returns_true(self):
        recorder = TrajectoryRecorder(goal="two steps")
        recorder.record_step(_make_click_action("#a"), snapshot_before=_make_snapshot("s0"))
        recorder.record_step(_make_click_action("#b"), snapshot_before=_make_snapshot("s1"))
        traj = recorder.finalize(success=True)
        replayer = TrajectoryReplayer()
        ok, issues = replayer.verify(traj)
        assert ok is True
        assert issues == []

    def test_verify_valid_one_step_returns_true(self):
        recorder = TrajectoryRecorder(goal="one step")
        recorder.record_step(_make_click_action("#x"))
        traj = recorder.finalize(success=True)
        replayer = TrajectoryReplayer()
        ok, issues = replayer.verify(traj)
        assert ok is True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestTrajectoryRegistry:
    def test_trajectory_registry_is_dict(self):
        assert isinstance(TRAJECTORY_REGISTRY, dict)

    def test_trajectory_registry_can_store_trajectory(self):
        traj = Trajectory(trajectory_id="reg-test", goal="test registry")
        TRAJECTORY_REGISTRY["reg-test"] = traj
        assert "reg-test" in TRAJECTORY_REGISTRY
        # Clean up.
        del TRAJECTORY_REGISTRY["reg-test"]
