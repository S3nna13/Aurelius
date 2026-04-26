"""
Tests for src/agent/execution_monitor.py  (≥28 tests)
"""

import time

import pytest

from src.agent.execution_monitor import (
    EXECUTION_MONITOR_REGISTRY,
    BudgetConfig,
    ExecutionEvent,
    ExecutionMonitor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monitor():
    return ExecutionMonitor()


@pytest.fixture
def tight_monitor():
    """Monitor with very tight budget so it trips easily."""
    cfg = BudgetConfig(max_steps=2, max_time_s=1000.0, max_cost=0.5)
    return ExecutionMonitor(config=cfg)


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


def test_registry_key_exists():
    assert "default" in EXECUTION_MONITOR_REGISTRY


def test_registry_value_is_execution_monitor():
    assert EXECUTION_MONITOR_REGISTRY["default"] is ExecutionMonitor


# ---------------------------------------------------------------------------
# BudgetConfig
# ---------------------------------------------------------------------------


def test_budget_config_defaults():
    cfg = BudgetConfig()
    assert cfg.max_steps == 50
    assert cfg.max_time_s == 300.0
    assert cfg.max_cost == 1.0


def test_budget_config_frozen():
    cfg = BudgetConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.max_steps = 99  # type: ignore[misc]


def test_budget_config_custom():
    cfg = BudgetConfig(max_steps=10, max_time_s=60.0, max_cost=2.0)
    assert cfg.max_steps == 10
    assert cfg.max_time_s == 60.0
    assert cfg.max_cost == 2.0


# ---------------------------------------------------------------------------
# step_count()
# ---------------------------------------------------------------------------


def test_step_count_empty(monitor):
    assert monitor.step_count() == 0


def test_step_count_increments(monitor):
    monitor.record("step", {})
    assert monitor.step_count() == 1
    monitor.record("step", {})
    assert monitor.step_count() == 2


def test_step_count_after_reset(monitor):
    monitor.record("step", {})
    monitor.reset()
    assert monitor.step_count() == 0


# ---------------------------------------------------------------------------
# elapsed_s()
# ---------------------------------------------------------------------------


def test_elapsed_s_empty(monitor):
    assert monitor.elapsed_s() == 0.0


def test_elapsed_s_positive_after_record(monitor):
    monitor.record("step", {})
    elapsed = monitor.elapsed_s()
    assert elapsed >= 0.0


def test_elapsed_s_increases_over_time(monitor):
    monitor.record("step", {})
    e1 = monitor.elapsed_s()
    time.sleep(0.05)
    e2 = monitor.elapsed_s()
    assert e2 >= e1


# ---------------------------------------------------------------------------
# total_cost()
# ---------------------------------------------------------------------------


def test_total_cost_empty(monitor):
    assert monitor.total_cost() == 0.0


def test_total_cost_sums_payload_cost(monitor):
    monitor.record("llm_call", {"cost": 0.10})
    monitor.record("llm_call", {"cost": 0.25})
    assert abs(monitor.total_cost() - 0.35) < 1e-9


def test_total_cost_missing_cost_key(monitor):
    monitor.record("step", {"info": "no cost here"})
    assert monitor.total_cost() == 0.0


def test_total_cost_partial_cost_keys(monitor):
    monitor.record("step", {"cost": 0.5})
    monitor.record("step", {})
    assert abs(monitor.total_cost() - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# budget_status()
# ---------------------------------------------------------------------------


def test_budget_status_all_ok_initial(monitor):
    status = monitor.budget_status()
    assert status["steps_ok"] is True
    assert status["time_ok"] is True
    assert status["cost_ok"] is True
    assert status["all_ok"] is True


def test_budget_status_steps_over(tight_monitor):
    tight_monitor.record("s", {})
    tight_monitor.record("s", {})
    tight_monitor.record("s", {})  # now 3 > max_steps=2
    status = tight_monitor.budget_status()
    assert status["steps_ok"] is False
    assert status["all_ok"] is False


def test_budget_status_cost_over(tight_monitor):
    tight_monitor.record("s", {"cost": 0.6})  # > max_cost=0.5
    status = tight_monitor.budget_status()
    assert status["cost_ok"] is False
    assert status["all_ok"] is False


# ---------------------------------------------------------------------------
# is_over_budget()
# ---------------------------------------------------------------------------


def test_is_over_budget_false_initially(monitor):
    assert monitor.is_over_budget() is False


def test_is_over_budget_true_after_exceeding_steps(tight_monitor):
    for _ in range(3):
        tight_monitor.record("s", {})
    assert tight_monitor.is_over_budget() is True


def test_is_over_budget_true_after_cost(tight_monitor):
    tight_monitor.record("s", {"cost": 1.0})  # > max_cost=0.5
    assert tight_monitor.is_over_budget() is True


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_clears_events(monitor):
    monitor.record("step", {})
    monitor.record("step", {})
    monitor.reset()
    assert monitor.step_count() == 0


def test_reset_clears_cost(monitor):
    monitor.record("llm", {"cost": 0.9})
    monitor.reset()
    assert monitor.total_cost() == 0.0


# ---------------------------------------------------------------------------
# export_log()
# ---------------------------------------------------------------------------


def test_export_log_empty(monitor):
    assert monitor.export_log() == []


def test_export_log_returns_list_of_dicts(monitor):
    monitor.record("step", {"x": 1})
    log = monitor.export_log()
    assert isinstance(log, list)
    assert isinstance(log[0], dict)


def test_export_log_contains_event_type(monitor):
    monitor.record("my_event", {"k": "v"})
    log = monitor.export_log()
    assert log[0]["event_type"] == "my_event"


def test_export_log_contains_payload(monitor):
    monitor.record("step", {"cost": 0.1})
    log = monitor.export_log()
    assert log[0]["payload"] == {"cost": 0.1}


# ---------------------------------------------------------------------------
# event_id auto-generation
# ---------------------------------------------------------------------------


def test_event_id_auto_generated(monitor):
    ev = monitor.record("step", {})
    assert isinstance(ev.event_id, str)
    assert len(ev.event_id) == 8


def test_event_ids_unique(monitor):
    ev1 = monitor.record("step", {})
    ev2 = monitor.record("step", {})
    assert ev1.event_id != ev2.event_id


# ---------------------------------------------------------------------------
# ExecutionEvent frozen
# ---------------------------------------------------------------------------


def test_execution_event_frozen():
    ev = ExecutionEvent(event_id="abc12345", event_type="step", payload={}, timestamp=0.0)
    with pytest.raises((AttributeError, TypeError)):
        ev.event_type = "other"  # type: ignore[misc]
