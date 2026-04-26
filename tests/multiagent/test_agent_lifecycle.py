"""Tests for src/multiagent/agent_lifecycle.py."""

from __future__ import annotations

import pytest

from src.multiagent.agent_lifecycle import (
    AGENT_LIFECYCLE_REGISTRY,
    AgentLifecycleManager,
    AgentRecord,
    AgentStatus,
)

# ---------------------------------------------------------------------------
# AgentStatus enum
# ---------------------------------------------------------------------------


def test_agent_status_values():
    assert AgentStatus.INITIALIZING == "initializing"
    assert AgentStatus.RUNNING == "running"
    assert AgentStatus.PAUSED == "paused"
    assert AgentStatus.TERMINATED == "terminated"
    assert AgentStatus.CRASHED == "crashed"


def test_agent_status_count():
    assert len(AgentStatus) == 5


def test_agent_status_is_str():
    assert isinstance(AgentStatus.RUNNING, str)


# ---------------------------------------------------------------------------
# AgentLifecycleManager.spawn
# ---------------------------------------------------------------------------


def test_spawn_returns_agent_record():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("worker")
    assert isinstance(record, AgentRecord)


def test_spawn_status_is_initializing():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("worker")
    assert record.status == AgentStatus.INITIALIZING


def test_spawn_name_stored():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("my-agent")
    assert record.name == "my-agent"


def test_spawn_agent_id_length():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    assert len(record.agent_id) == 10


def test_spawn_agent_ids_unique():
    mgr = AgentLifecycleManager(max_agents=100)
    ids = {mgr.spawn(f"agent-{i}").agent_id for i in range(30)}
    assert len(ids) == 30


def test_spawn_created_at_positive():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    assert record.created_at > 0


def test_spawn_metadata_stored():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a", metadata={"role": "planner"})
    assert record.metadata["role"] == "planner"


def test_spawn_metadata_defaults_empty():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("b")
    assert record.metadata == {}


def test_spawn_raises_value_error_at_max_agents():
    mgr = AgentLifecycleManager(max_agents=2)
    mgr.spawn("a")
    mgr.spawn("b")
    with pytest.raises(ValueError):
        mgr.spawn("c")


def test_spawn_max_agents_counts_non_terminal_only():
    mgr = AgentLifecycleManager(max_agents=2)
    r1 = mgr.spawn("a")
    mgr.terminate(r1.agent_id)
    # terminated does not count toward limit
    mgr.spawn("b")
    mgr.spawn("c")  # should not raise


# ---------------------------------------------------------------------------
# AgentLifecycleManager.start
# ---------------------------------------------------------------------------


def test_start_initializing_to_running():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    assert mgr.start(record.agent_id) is True
    assert record.status == AgentStatus.RUNNING


def test_start_returns_false_if_not_initializing():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    mgr.start(record.agent_id)
    assert mgr.start(record.agent_id) is False  # already RUNNING


def test_start_returns_false_for_missing_agent():
    mgr = AgentLifecycleManager()
    assert mgr.start("no-such-id") is False


# ---------------------------------------------------------------------------
# AgentLifecycleManager.pause
# ---------------------------------------------------------------------------


def test_pause_running_to_paused():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    mgr.start(record.agent_id)
    assert mgr.pause(record.agent_id) is True
    assert record.status == AgentStatus.PAUSED


def test_pause_returns_false_if_not_running():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    assert mgr.pause(record.agent_id) is False  # still INITIALIZING


def test_pause_returns_false_for_missing_agent():
    mgr = AgentLifecycleManager()
    assert mgr.pause("ghost") is False


# ---------------------------------------------------------------------------
# AgentLifecycleManager.resume
# ---------------------------------------------------------------------------


def test_resume_paused_to_running():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    mgr.start(record.agent_id)
    mgr.pause(record.agent_id)
    assert mgr.resume(record.agent_id) is True
    assert record.status == AgentStatus.RUNNING


def test_resume_returns_false_if_not_paused():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    mgr.start(record.agent_id)
    assert mgr.resume(record.agent_id) is False  # RUNNING, not PAUSED


def test_resume_returns_false_for_missing_agent():
    mgr = AgentLifecycleManager()
    assert mgr.resume("ghost") is False


# ---------------------------------------------------------------------------
# AgentLifecycleManager.terminate
# ---------------------------------------------------------------------------


def test_terminate_any_state_to_terminated():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    assert mgr.terminate(record.agent_id) is True
    assert record.status == AgentStatus.TERMINATED


def test_terminate_running_agent():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    mgr.start(record.agent_id)
    mgr.terminate(record.agent_id)
    assert record.status == AgentStatus.TERMINATED


def test_terminate_returns_false_for_missing():
    mgr = AgentLifecycleManager()
    assert mgr.terminate("no-id") is False


# ---------------------------------------------------------------------------
# AgentLifecycleManager.crash
# ---------------------------------------------------------------------------


def test_crash_sets_status_crashed():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    mgr.start(record.agent_id)
    assert mgr.crash(record.agent_id, "segfault") is True
    assert record.status == AgentStatus.CRASHED


def test_crash_stores_reason_in_metadata():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    mgr.start(record.agent_id)
    mgr.crash(record.agent_id, reason="OOM error")
    assert record.metadata["crash_reason"] == "OOM error"


def test_crash_empty_reason():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    mgr.crash(record.agent_id)
    assert record.metadata["crash_reason"] == ""


def test_crash_returns_false_for_missing():
    mgr = AgentLifecycleManager()
    assert mgr.crash("ghost") is False


# ---------------------------------------------------------------------------
# AgentLifecycleManager.get
# ---------------------------------------------------------------------------


def test_get_existing_returns_record():
    mgr = AgentLifecycleManager()
    record = mgr.spawn("a")
    assert mgr.get(record.agent_id) is record


def test_get_missing_returns_none():
    mgr = AgentLifecycleManager()
    assert mgr.get("no-id") is None


# ---------------------------------------------------------------------------
# AgentLifecycleManager.list_by_status
# ---------------------------------------------------------------------------


def test_list_by_status_returns_matching():
    mgr = AgentLifecycleManager()
    r1 = mgr.spawn("a")
    r2 = mgr.spawn("b")
    mgr.start(r1.agent_id)
    result = mgr.list_by_status(AgentStatus.RUNNING)
    assert r1 in result
    assert r2 not in result


def test_list_by_status_sorted_by_agent_id():
    mgr = AgentLifecycleManager(max_agents=20)
    records = [mgr.spawn(f"agent-{i}") for i in range(5)]
    for r in records:
        mgr.start(r.agent_id)
    result = mgr.list_by_status(AgentStatus.RUNNING)
    agent_ids = [r.agent_id for r in result]
    assert agent_ids == sorted(agent_ids)


def test_list_by_status_empty_when_none_match():
    mgr = AgentLifecycleManager()
    mgr.spawn("a")
    assert mgr.list_by_status(AgentStatus.RUNNING) == []


# ---------------------------------------------------------------------------
# AgentLifecycleManager.active_count
# ---------------------------------------------------------------------------


def test_active_count_running_and_paused():
    mgr = AgentLifecycleManager()
    r1 = mgr.spawn("a")
    r2 = mgr.spawn("b")
    mgr.spawn("c")
    mgr.start(r1.agent_id)
    mgr.start(r2.agent_id)
    mgr.pause(r2.agent_id)
    # r1=RUNNING, r2=PAUSED, r3=INITIALIZING
    assert mgr.active_count() == 2


def test_active_count_excludes_terminated():
    mgr = AgentLifecycleManager()
    r1 = mgr.spawn("a")
    mgr.start(r1.agent_id)
    mgr.terminate(r1.agent_id)
    assert mgr.active_count() == 0


def test_active_count_excludes_crashed():
    mgr = AgentLifecycleManager()
    r1 = mgr.spawn("a")
    mgr.crash(r1.agent_id, "err")
    assert mgr.active_count() == 0


def test_active_count_zero_for_empty_manager():
    mgr = AgentLifecycleManager()
    assert mgr.active_count() == 0


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in AGENT_LIFECYCLE_REGISTRY


def test_registry_default_is_manager_class():
    assert AGENT_LIFECYCLE_REGISTRY["default"] is AgentLifecycleManager
