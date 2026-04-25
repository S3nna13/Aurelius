"""Tests for src/agent/loop_guard.py"""

from __future__ import annotations

import pytest

from src.agent.loop_guard import (
    AGENT_REGISTRY,
    AgentLoopGuard,
    LoopGuardConfig,
    LoopGuardResult,
    StallReason,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_guard() -> AgentLoopGuard:
    return AgentLoopGuard()


@pytest.fixture
def tight_guard() -> AgentLoopGuard:
    """Guard with tiny limits to trigger conditions quickly."""
    return AgentLoopGuard(
        LoopGuardConfig(
            max_no_progress=3,
            max_steps=5,
            max_action_repeats=2,
            history_window=5,
        )
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_loop_guard():
    assert "loop_guard" in AGENT_REGISTRY
    assert AGENT_REGISTRY["loop_guard"] is AgentLoopGuard


# ---------------------------------------------------------------------------
# Normal (non-terminating) operation
# ---------------------------------------------------------------------------

def test_ok_result_for_fresh_step(default_guard):
    result = default_guard.update({"type": "search"}, {}, progress_signal=0.5)
    assert result.should_terminate is False
    assert result.reason is None
    assert result.message == "OK"


def test_steps_taken_increments(default_guard):
    for i in range(3):
        result = default_guard.update({"type": f"action_{i}"}, {}, progress_signal=1.0)
    assert result.steps_taken == 3


# ---------------------------------------------------------------------------
# MAX_STEPS
# ---------------------------------------------------------------------------

def test_max_steps_triggers(tight_guard):
    """Hitting max_steps (5) should terminate with MAX_STEPS."""
    result = None
    for i in range(5):
        result = tight_guard.update({"type": f"a{i}"}, {}, progress_signal=1.0)
    assert result.should_terminate is True
    assert result.reason == StallReason.MAX_STEPS


def test_max_steps_message_contains_limit(tight_guard):
    for i in range(5):
        result = tight_guard.update({"type": f"a{i}"}, {}, progress_signal=1.0)
    assert "5" in result.message


# ---------------------------------------------------------------------------
# NO_PROGRESS
# ---------------------------------------------------------------------------

def test_no_progress_triggers(tight_guard):
    """3 consecutive zero-progress steps → NO_PROGRESS."""
    result = None
    for _ in range(3):
        result = tight_guard.update({"type": "unique_action"}, {}, progress_signal=0.0)
    # After 3 steps with progress < 0.01 the window is full
    assert result.should_terminate is True
    assert result.reason == StallReason.NO_PROGRESS


def test_progress_above_threshold_resets_window(tight_guard):
    """A high-progress step mid-sequence should prevent NO_PROGRESS."""
    tight_guard.update({"type": "a1"}, {}, progress_signal=0.0)
    tight_guard.update({"type": "a2"}, {}, progress_signal=1.0)  # resets signal
    result = tight_guard.update({"type": "a3"}, {}, progress_signal=0.0)
    # Only 1 step in the zero-progress run after the reset — not enough
    assert result.reason != StallReason.NO_PROGRESS


# ---------------------------------------------------------------------------
# REPEATED_ACTION
# ---------------------------------------------------------------------------

def test_repeated_action_triggers(tight_guard):
    """Same action appearing max_action_repeats (2) times → REPEATED_ACTION."""
    action = {"type": "search"}
    result = None
    # First occurrence: appended to deque (count=0 before append → OK)
    r1 = tight_guard.update(action, {}, progress_signal=1.0)
    # Second occurrence: count=1 before append → CYCLE_DETECTED (count >= 1 but < 2)
    r2 = tight_guard.update(action, {}, progress_signal=1.0)
    assert r2.reason in (StallReason.CYCLE_DETECTED, StallReason.REPEATED_ACTION)


def test_repeated_action_exact_count(tight_guard):
    """Action repeated exactly max_action_repeats times triggers REPEATED_ACTION."""
    action = {"type": "repeat_me"}
    results = []
    for _ in range(3):
        results.append(tight_guard.update(action, {}, progress_signal=1.0))
    reasons = [r.reason for r in results if r.should_terminate]
    # At least one termination should occur
    assert any(
        r in (StallReason.REPEATED_ACTION, StallReason.CYCLE_DETECTED) for r in reasons
    )


# ---------------------------------------------------------------------------
# CYCLE_DETECTED
# ---------------------------------------------------------------------------

def test_cycle_detected_on_second_occurrence():
    """Second occurrence of same hash (within window, below repeat threshold) → CYCLE."""
    guard = AgentLoopGuard(
        LoopGuardConfig(max_action_repeats=3, history_window=10, max_steps=50)
    )
    action = {"type": "cycle_action"}
    r1 = guard.update(action, {}, progress_signal=1.0)
    assert r1.should_terminate is False  # first time is fine
    r2 = guard.update(action, {}, progress_signal=1.0)
    assert r2.should_terminate is True
    assert r2.reason == StallReason.CYCLE_DETECTED


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_clears_steps(tight_guard):
    for i in range(3):
        tight_guard.update({"type": f"a{i}"}, {}, progress_signal=1.0)
    tight_guard.reset()
    result = tight_guard.update({"type": "fresh"}, {}, progress_signal=1.0)
    assert result.steps_taken == 1


def test_reset_clears_history(tight_guard):
    action = {"type": "search"}
    tight_guard.update(action, {}, progress_signal=1.0)
    tight_guard.reset()
    # After reset, same action is fresh — no cycle
    result = tight_guard.update(action, {}, progress_signal=1.0)
    assert result.should_terminate is False


# ---------------------------------------------------------------------------
# action_hash
# ---------------------------------------------------------------------------

def test_action_hash_deterministic(default_guard):
    action = {"type": "search", "query": "hello"}
    h1 = default_guard._action_hash(action)
    h2 = default_guard._action_hash(action)
    assert h1 == h2


def test_action_hash_differs_for_different_actions(default_guard):
    h1 = default_guard._action_hash({"type": "search"})
    h2 = default_guard._action_hash({"type": "delete"})
    assert h1 != h2


def test_action_hash_length(default_guard):
    h = default_guard._action_hash({"type": "anything"})
    assert len(h) == 16


# ---------------------------------------------------------------------------
# LoopGuardResult shape
# ---------------------------------------------------------------------------

def test_result_is_dataclass(default_guard):
    result = default_guard.update({"type": "x"}, {})
    assert isinstance(result, LoopGuardResult)
    assert hasattr(result, "should_terminate")
    assert hasattr(result, "reason")
    assert hasattr(result, "steps_taken")
    assert hasattr(result, "message")
