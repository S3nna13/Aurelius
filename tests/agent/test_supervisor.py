"""Tests for agent_supervisor — lifecycle, health, and crash budgets."""

from __future__ import annotations

import pytest

from src.agent.supervisor import (
    DEFAULT_SUPERVISOR,
    SUPERVISOR_REGISTRY,
    AgentState,
    AgentSupervisor,
    SupervisorConfig,
)

# ---------------------------------------------------------------------------
# Registration and start
# ---------------------------------------------------------------------------


def test_register_creates_idle_agent():
    sv = AgentSupervisor()
    rec = sv.register("alice")
    assert rec.state == AgentState.IDLE
    assert rec.agent_id == "alice"


def test_start_transitions_to_running():
    sv = AgentSupervisor()
    sv.register("alice")
    rec = sv.start("alice")
    assert rec.state == AgentState.RUNNING
    assert rec.start_count == 1


def test_start_increments_start_count():
    sv = AgentSupervisor()
    sv.register("alice")
    sv.start("alice")
    sv.stop("alice")
    sv.start("alice")
    assert sv._agents["alice"].start_count == 2


# ---------------------------------------------------------------------------
# Stop and terminate
# ---------------------------------------------------------------------------


def test_stop_transitions_to_idle():
    sv = AgentSupervisor()
    sv.register("alice")
    sv.start("alice")
    rec = sv.stop("alice")
    assert rec.state == AgentState.IDLE


def test_terminate_prevents_restart():
    sv = AgentSupervisor()
    sv.register("alice")
    sv.terminate("alice")
    with pytest.raises(RuntimeError):
        sv.start("alice")


# ---------------------------------------------------------------------------
# Crash handling
# ---------------------------------------------------------------------------


def test_crash_increments_crash_count():
    sv = AgentSupervisor()
    sv.register("alice")
    sv.start("alice")
    sv.crash("alice")
    assert sv._agents["alice"].crash_count == 1


def test_crash_exhaustion_marks_crashed():
    sv = AgentSupervisor(config=SupervisorConfig(max_restarts=1))
    sv.register("alice")
    sv.start("alice")
    sv.crash("alice")
    sv.start("alice")
    sv.crash("alice")
    assert sv._agents["alice"].state == AgentState.CRASHED


def test_crash_within_budget_returns_to_idle():
    sv = AgentSupervisor(config=SupervisorConfig(max_restarts=5, restart_window_seconds=0.0))
    sv.register("alice")
    sv.start("alice")
    rec = sv.crash("alice")
    assert rec.state == AgentState.IDLE


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_unknown_agent():
    sv = AgentSupervisor()
    h = sv.health("bob")
    assert h["status"] == "unknown"
    assert h["registered"] is False


def test_health_running_agent():
    sv = AgentSupervisor()
    sv.register("alice")
    sv.start("alice")
    h = sv.health("alice")
    assert h["status"] == "running"
    assert h["registered"] is True


# ---------------------------------------------------------------------------
# Task counting
# ---------------------------------------------------------------------------


def test_increment_task():
    sv = AgentSupervisor()
    sv.register("alice")
    sv.increment_task("alice")
    sv.increment_task("alice")
    assert sv._agents["alice"].task_count == 2


def test_increment_task_unknown_is_noop():
    sv = AgentSupervisor()
    sv.increment_task("ghost")  # should not raise


# ---------------------------------------------------------------------------
# List agents
# ---------------------------------------------------------------------------


def test_list_agents():
    sv = AgentSupervisor()
    sv.register("alice")
    sv.register("bob")
    assert sv.list_agents() == ["alice", "bob"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in SUPERVISOR_REGISTRY
    assert isinstance(SUPERVISOR_REGISTRY["default"], AgentSupervisor)


def test_default_is_supervisor():
    assert isinstance(DEFAULT_SUPERVISOR, AgentSupervisor)
