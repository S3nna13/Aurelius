"""Tests for src/agent/tool_use_tracker.py"""

from __future__ import annotations

import pytest

from src.agent.tool_use_tracker import (
    TOOL_USE_TRACKER_REGISTRY,
    ToolBudget,
    ToolCall,
    ToolUseTracker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tracker(**kwargs) -> ToolUseTracker:
    return ToolUseTracker(**kwargs)


def record_call(
    tracker: ToolUseTracker,
    tool_name: str = "search",
    input_size: int = 10,
    output_size: int = 20,
    duration_ms: float = 50.0,
    success: bool = True,
    error: str = "",
) -> ToolCall:
    return tracker.record(
        tool_name=tool_name,
        input_size=input_size,
        output_size=output_size,
        duration_ms=duration_ms,
        success=success,
        error=error,
    )


# ---------------------------------------------------------------------------
# ToolCall – frozen / auto-id
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_create_returns_tool_call(self):
        tc = ToolCall.create(tool_name="foo", input_size=5, output_size=10, duration_ms=1.0)
        assert isinstance(tc, ToolCall)

    def test_auto_call_id_is_8_chars(self):
        tc = ToolCall.create(tool_name="foo", input_size=5, output_size=10, duration_ms=1.0)
        assert len(tc.call_id) == 8

    def test_explicit_call_id_preserved(self):
        tc = ToolCall.create(
            tool_name="foo",
            input_size=5,
            output_size=10,
            duration_ms=1.0,
            call_id="abcd1234",
        )
        assert tc.call_id == "abcd1234"

    def test_frozen_tool_call_cannot_be_mutated(self):
        tc = ToolCall.create(tool_name="foo", input_size=5, output_size=10, duration_ms=1.0)
        with pytest.raises((AttributeError, TypeError)):
            tc.tool_name = "bar"  # type: ignore[misc]

    def test_default_success_true(self):
        tc = ToolCall.create(tool_name="foo", input_size=0, output_size=0, duration_ms=0.0)
        assert tc.success is True

    def test_default_error_empty_string(self):
        tc = ToolCall.create(tool_name="foo", input_size=0, output_size=0, duration_ms=0.0)
        assert tc.error == ""


# ---------------------------------------------------------------------------
# ToolBudget – frozen
# ---------------------------------------------------------------------------


class TestToolBudget:
    def test_defaults(self):
        b = ToolBudget()
        assert b.max_calls == 100
        assert b.max_total_duration_ms == 30_000.0

    def test_frozen_cannot_be_mutated(self):
        b = ToolBudget()
        with pytest.raises((AttributeError, TypeError)):
            b.max_calls = 999  # type: ignore[misc]

    def test_custom_values(self):
        b = ToolBudget(max_calls=5, max_total_duration_ms=1000.0)
        assert b.max_calls == 5
        assert b.max_total_duration_ms == 1000.0


# ---------------------------------------------------------------------------
# ToolUseTracker – record
# ---------------------------------------------------------------------------


class TestToolUseTrackerRecord:
    def test_record_returns_tool_call(self):
        t = make_tracker()
        tc = record_call(t)
        assert isinstance(tc, ToolCall)

    def test_record_stores_call(self):
        t = make_tracker()
        tc = record_call(t, tool_name="write")
        assert t.total_calls() == 1
        assert t.calls_for("write")[0].call_id == tc.call_id

    def test_call_id_auto_generated(self):
        t = make_tracker()
        tc = record_call(t)
        assert len(tc.call_id) == 8

    def test_record_stores_tool_name(self):
        t = make_tracker()
        tc = record_call(t, tool_name="execute")
        assert tc.tool_name == "execute"

    def test_record_stores_sizes(self):
        t = make_tracker()
        tc = record_call(t, input_size=100, output_size=200)
        assert tc.input_size == 100
        assert tc.output_size == 200

    def test_record_stores_duration(self):
        t = make_tracker()
        tc = record_call(t, duration_ms=123.4)
        assert tc.duration_ms == pytest.approx(123.4)

    def test_record_failure(self):
        t = make_tracker()
        tc = record_call(t, success=False, error="timeout")
        assert tc.success is False
        assert tc.error == "timeout"


# ---------------------------------------------------------------------------
# ToolUseTracker – queries
# ---------------------------------------------------------------------------


class TestToolUseTrackerQueries:
    def test_calls_for_returns_only_matching_tool(self):
        t = make_tracker()
        record_call(t, tool_name="search")
        record_call(t, tool_name="fetch")
        record_call(t, tool_name="search")
        results = t.calls_for("search")
        assert len(results) == 2
        assert all(c.tool_name == "search" for c in results)

    def test_calls_for_unknown_tool_empty(self):
        t = make_tracker()
        record_call(t, tool_name="search")
        assert t.calls_for("nonexistent") == []

    def test_total_calls_zero_initially(self):
        t = make_tracker()
        assert t.total_calls() == 0

    def test_total_calls_increments(self):
        t = make_tracker()
        record_call(t)
        record_call(t)
        assert t.total_calls() == 2

    def test_total_duration_ms_sums_correctly(self):
        t = make_tracker()
        record_call(t, duration_ms=100.0)
        record_call(t, duration_ms=200.0)
        assert t.total_duration_ms() == pytest.approx(300.0)

    def test_total_duration_ms_zero_initially(self):
        t = make_tracker()
        assert t.total_duration_ms() == 0.0


# ---------------------------------------------------------------------------
# ToolUseTracker – budget_status
# ---------------------------------------------------------------------------


class TestBudgetStatus:
    def test_no_budget_all_ok(self):
        t = make_tracker()
        record_call(t)
        status = t.budget_status()
        assert status["all_ok"] is True
        assert status["calls_ok"] is True
        assert status["duration_ok"] is True

    def test_within_budget_all_ok(self):
        budget = ToolBudget(max_calls=5, max_total_duration_ms=1000.0)
        t = make_tracker(budget=budget)
        record_call(t, duration_ms=100.0)
        status = t.budget_status()
        assert status["all_ok"] is True

    def test_over_call_budget(self):
        budget = ToolBudget(max_calls=1, max_total_duration_ms=99999.0)
        t = make_tracker(budget=budget)
        record_call(t)
        record_call(t)
        status = t.budget_status()
        assert status["calls_ok"] is False
        assert status["all_ok"] is False

    def test_over_duration_budget(self):
        budget = ToolBudget(max_calls=100, max_total_duration_ms=50.0)
        t = make_tracker(budget=budget)
        record_call(t, duration_ms=100.0)
        status = t.budget_status()
        assert status["duration_ok"] is False
        assert status["all_ok"] is False

    def test_budget_status_contains_usage_fields(self):
        t = make_tracker()
        record_call(t, duration_ms=77.7)
        status = t.budget_status()
        assert "calls_used" in status
        assert "duration_used_ms" in status
        assert status["calls_used"] == 1
        assert status["duration_used_ms"] == pytest.approx(77.7)


# ---------------------------------------------------------------------------
# ToolUseTracker – most_used & success_rate
# ---------------------------------------------------------------------------


class TestMostUsedAndSuccessRate:
    def test_most_used_sorted_desc(self):
        t = make_tracker()
        for _ in range(3):
            record_call(t, tool_name="search")
        for _ in range(5):
            record_call(t, tool_name="fetch")
        record_call(t, tool_name="write")
        top = t.most_used(n=3)
        assert top[0][0] == "fetch"
        assert top[0][1] == 5
        assert top[1][0] == "search"

    def test_most_used_respects_n(self):
        t = make_tracker()
        for tool in ["a", "b", "c", "d"]:
            record_call(t, tool_name=tool)
        assert len(t.most_used(n=2)) == 2

    def test_most_used_empty_tracker(self):
        t = make_tracker()
        assert t.most_used() == []

    def test_success_rate_all_success(self):
        t = make_tracker()
        record_call(t, success=True)
        record_call(t, success=True)
        assert t.success_rate() == pytest.approx(1.0)

    def test_success_rate_with_failures(self):
        t = make_tracker()
        record_call(t, success=True)
        record_call(t, success=False)
        assert t.success_rate() == pytest.approx(0.5)

    def test_success_rate_per_tool(self):
        t = make_tracker()
        record_call(t, tool_name="search", success=True)
        record_call(t, tool_name="search", success=False)
        record_call(t, tool_name="fetch", success=False)
        assert t.success_rate("search") == pytest.approx(0.5)
        assert t.success_rate("fetch") == pytest.approx(0.0)

    def test_success_rate_zero_if_no_calls(self):
        t = make_tracker()
        assert t.success_rate() == 0.0

    def test_success_rate_zero_if_no_calls_for_tool(self):
        t = make_tracker()
        assert t.success_rate("nonexistent") == 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in TOOL_USE_TRACKER_REGISTRY

    def test_registry_default_is_tool_use_tracker_class(self):
        assert TOOL_USE_TRACKER_REGISTRY["default"] is ToolUseTracker

    def test_registry_default_is_instantiable(self):
        cls = TOOL_USE_TRACKER_REGISTRY["default"]
        obj = cls()
        assert isinstance(obj, ToolUseTracker)
