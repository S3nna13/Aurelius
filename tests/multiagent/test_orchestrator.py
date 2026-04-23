"""Tests for src/multiagent/orchestrator.py."""
from __future__ import annotations

import pytest

from src.multiagent.orchestrator import (
    AgentSpec,
    Orchestrator,
    TaskAssignment,
    ORCHESTRATOR,
)


# ---------------------------------------------------------------------------
# AgentSpec
# ---------------------------------------------------------------------------

def test_agent_spec_auto_id():
    spec = AgentSpec(name="alpha")
    assert spec.agent_id
    assert len(spec.agent_id) == 8


def test_agent_spec_unique_ids():
    a = AgentSpec(name="a")
    b = AgentSpec(name="b")
    assert a.agent_id != b.agent_id


def test_agent_spec_defaults():
    spec = AgentSpec(name="bot")
    assert spec.capabilities == []
    assert spec.max_concurrent == 1
    assert spec.priority == 0


def test_agent_spec_custom_fields():
    spec = AgentSpec(name="x", capabilities=["summarize"], max_concurrent=3, priority=5)
    assert spec.capabilities == ["summarize"]
    assert spec.max_concurrent == 3
    assert spec.priority == 5


def test_agent_spec_custom_id():
    spec = AgentSpec(name="x", agent_id="myid123")
    assert spec.agent_id == "myid123"


# ---------------------------------------------------------------------------
# TaskAssignment
# ---------------------------------------------------------------------------

def test_task_assignment_auto_id():
    ta = TaskAssignment(agent_id="a1", task="do thing")
    assert ta.task_id
    assert len(ta.task_id) == 8


def test_task_assignment_unique_ids():
    t1 = TaskAssignment(agent_id="a", task="t1")
    t2 = TaskAssignment(agent_id="a", task="t2")
    assert t1.task_id != t2.task_id


def test_task_assignment_defaults():
    ta = TaskAssignment(agent_id="a1", task="task")
    assert ta.status == "pending"
    assert ta.result == ""
    assert ta.attempts == 0


# ---------------------------------------------------------------------------
# Orchestrator.register_agent / agents()
# ---------------------------------------------------------------------------

def test_register_and_list():
    orch = Orchestrator()
    spec = AgentSpec(name="agent1", capabilities=["run"])
    orch.register_agent(spec)
    assert spec in orch.agents()


def test_register_multiple():
    orch = Orchestrator()
    s1 = AgentSpec(name="a1")
    s2 = AgentSpec(name="a2")
    orch.register_agent(s1)
    orch.register_agent(s2)
    assert len(orch.agents()) == 2


def test_agents_empty_initially():
    orch = Orchestrator()
    assert orch.agents() == []


# ---------------------------------------------------------------------------
# assign()
# ---------------------------------------------------------------------------

def test_assign_returns_task_assignment():
    orch = Orchestrator()
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    ta = orch.assign("write code", spec.agent_id)
    assert isinstance(ta, TaskAssignment)


def test_assign_correct_agent_and_task():
    orch = Orchestrator()
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    ta = orch.assign("do thing", spec.agent_id)
    assert ta.agent_id == spec.agent_id
    assert ta.task == "do thing"
    assert ta.status == "pending"


def test_assign_unknown_agent_raises():
    orch = Orchestrator()
    with pytest.raises((ValueError, KeyError)):
        orch.assign("task", "nonexistent-id")


def test_assign_stores_task():
    orch = Orchestrator()
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    ta = orch.assign("task", spec.agent_id)
    assert ta in orch.pending_tasks()


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------

def test_complete_returns_true_for_known_task():
    orch = Orchestrator()
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    ta = orch.assign("task", spec.agent_id)
    assert orch.complete(ta.task_id, "done result") is True


def test_complete_returns_false_for_unknown():
    orch = Orchestrator()
    assert orch.complete("fake-id", "result") is False


def test_complete_sets_status_done():
    orch = Orchestrator()
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    ta = orch.assign("task", spec.agent_id)
    orch.complete(ta.task_id, "ok")
    assert ta.status == "done"


def test_complete_sets_result():
    orch = Orchestrator()
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    ta = orch.assign("task", spec.agent_id)
    orch.complete(ta.task_id, "my result")
    assert ta.result == "my result"


# ---------------------------------------------------------------------------
# fail()
# ---------------------------------------------------------------------------

def test_fail_returns_true_for_known_task():
    orch = Orchestrator()
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    ta = orch.assign("task", spec.agent_id)
    assert orch.fail(ta.task_id) is True


def test_fail_returns_false_for_unknown():
    orch = Orchestrator()
    assert orch.fail("fake-id") is False


def test_fail_sets_status_failed():
    orch = Orchestrator()
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    ta = orch.assign("task", spec.agent_id)
    orch.fail(ta.task_id)
    assert ta.status == "failed"


def test_fail_increments_attempts():
    orch = Orchestrator()
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    ta = orch.assign("task", spec.agent_id)
    orch.fail(ta.task_id)
    assert ta.attempts == 1
    orch.fail(ta.task_id)
    assert ta.attempts == 2


# ---------------------------------------------------------------------------
# retry_eligible()
# ---------------------------------------------------------------------------

def test_retry_eligible_true_when_within_retries():
    orch = Orchestrator(max_retries=2)
    ta = TaskAssignment(agent_id="a", task="t", status="failed", attempts=1)
    assert orch.retry_eligible(ta) is True


def test_retry_eligible_true_at_max_retries():
    orch = Orchestrator(max_retries=2)
    ta = TaskAssignment(agent_id="a", task="t", status="failed", attempts=2)
    assert orch.retry_eligible(ta) is True


def test_retry_eligible_false_when_exceeded():
    orch = Orchestrator(max_retries=2)
    ta = TaskAssignment(agent_id="a", task="t", status="failed", attempts=3)
    assert orch.retry_eligible(ta) is False


def test_retry_eligible_false_for_pending():
    orch = Orchestrator(max_retries=2)
    ta = TaskAssignment(agent_id="a", task="t", status="pending", attempts=0)
    assert orch.retry_eligible(ta) is False


def test_retry_eligible_false_for_done():
    orch = Orchestrator(max_retries=2)
    ta = TaskAssignment(agent_id="a", task="t", status="done", attempts=0)
    assert orch.retry_eligible(ta) is False


# ---------------------------------------------------------------------------
# pending_tasks()
# ---------------------------------------------------------------------------

def test_pending_tasks_includes_pending():
    orch = Orchestrator()
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    ta = orch.assign("task", spec.agent_id)
    assert ta in orch.pending_tasks()


def test_pending_tasks_includes_retry_eligible_failed():
    orch = Orchestrator(max_retries=2)
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    ta = orch.assign("task", spec.agent_id)
    orch.fail(ta.task_id)
    assert ta in orch.pending_tasks()


def test_pending_tasks_excludes_done():
    orch = Orchestrator()
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    ta = orch.assign("task", spec.agent_id)
    orch.complete(ta.task_id, "ok")
    assert ta not in orch.pending_tasks()


def test_pending_tasks_excludes_exhausted_failed():
    orch = Orchestrator(max_retries=1)
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    ta = orch.assign("task", spec.agent_id)
    orch.fail(ta.task_id)
    orch.fail(ta.task_id)  # attempts=2, exceeds max_retries=1
    assert ta not in orch.pending_tasks()


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

def test_summary_empty():
    orch = Orchestrator()
    s = orch.summary()
    assert s == {"total": 0, "pending": 0, "done": 0, "failed": 0}


def test_summary_counts():
    orch = Orchestrator()
    spec = AgentSpec(name="bot")
    orch.register_agent(spec)
    t1 = orch.assign("t1", spec.agent_id)
    t2 = orch.assign("t2", spec.agent_id)
    t3 = orch.assign("t3", spec.agent_id)
    orch.complete(t1.task_id, "done")
    orch.fail(t2.task_id)
    s = orch.summary()
    assert s["total"] == 3
    assert s["pending"] == 1
    assert s["done"] == 1
    assert s["failed"] == 1


# ---------------------------------------------------------------------------
# ORCHESTRATOR singleton
# ---------------------------------------------------------------------------

def test_orchestrator_singleton_exists():
    assert ORCHESTRATOR is not None
    assert isinstance(ORCHESTRATOR, Orchestrator)
