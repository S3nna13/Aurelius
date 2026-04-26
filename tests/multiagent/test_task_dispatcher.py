"""Tests for src/multiagent/task_dispatcher.py."""

from __future__ import annotations

from src.multiagent.task_dispatcher import (
    TASK_DISPATCHER_REGISTRY,
    DispatchedTask,
    DispatchStatus,
    TaskDispatcher,
)

# ---------------------------------------------------------------------------
# DispatchStatus enum
# ---------------------------------------------------------------------------


def test_dispatch_status_values():
    assert DispatchStatus.PENDING == "pending"
    assert DispatchStatus.ASSIGNED == "assigned"
    assert DispatchStatus.COMPLETED == "completed"
    assert DispatchStatus.FAILED == "failed"
    assert DispatchStatus.TIMEOUT == "timeout"


def test_dispatch_status_count():
    assert len(DispatchStatus) == 5


def test_dispatch_status_is_str():
    assert isinstance(DispatchStatus.PENDING, str)


# ---------------------------------------------------------------------------
# DispatchedTask dataclass
# ---------------------------------------------------------------------------


def test_dispatched_task_defaults():
    task = DispatchedTask(task_id="abc", description="do stuff")
    assert task.assigned_to == ""
    assert task.status == DispatchStatus.PENDING
    assert task.result == ""
    assert task.created_at > 0


def test_dispatched_task_created_at_auto():
    task = DispatchedTask(task_id="t1", description="test")
    assert task.created_at > 0


def test_dispatched_task_created_at_explicit():
    task = DispatchedTask(task_id="t2", description="test", created_at=5.0)
    assert task.created_at == 5.0


# ---------------------------------------------------------------------------
# TaskDispatcher.submit
# ---------------------------------------------------------------------------


def test_submit_returns_dispatched_task():
    td = TaskDispatcher(agents=["a1"])
    task = td.submit("process data")
    assert isinstance(task, DispatchedTask)


def test_submit_task_id_auto_generated():
    td = TaskDispatcher(agents=["a1"])
    task = td.submit("work")
    assert len(task.task_id) == 8


def test_submit_task_id_unique():
    td = TaskDispatcher(agents=["a1"])
    ids = {td.submit("work").task_id for _ in range(20)}
    assert len(ids) == 20


def test_submit_status_is_pending():
    td = TaskDispatcher(agents=["a1"])
    task = td.submit("task")
    assert task.status == DispatchStatus.PENDING


def test_submit_description_stored():
    td = TaskDispatcher(agents=["a1"])
    task = td.submit("my description")
    assert task.description == "my description"


# ---------------------------------------------------------------------------
# TaskDispatcher.assign
# ---------------------------------------------------------------------------


def test_assign_returns_true_on_success():
    td = TaskDispatcher(agents=["agent-1"])
    task = td.submit("job")
    assert td.assign(task.task_id, "agent-1") is True


def test_assign_sets_status_assigned():
    td = TaskDispatcher(agents=["agent-1"])
    task = td.submit("job")
    td.assign(task.task_id, "agent-1")
    assert task.status == DispatchStatus.ASSIGNED


def test_assign_sets_assigned_to():
    td = TaskDispatcher(agents=["agent-1"])
    task = td.submit("job")
    td.assign(task.task_id, "agent-1")
    assert task.assigned_to == "agent-1"


def test_assign_unknown_task_returns_false():
    td = TaskDispatcher(agents=["agent-1"])
    assert td.assign("no-such-task", "agent-1") is False


def test_assign_unknown_agent_returns_false():
    td = TaskDispatcher(agents=["agent-1"])
    task = td.submit("job")
    assert td.assign(task.task_id, "ghost-agent") is False


# ---------------------------------------------------------------------------
# TaskDispatcher.complete
# ---------------------------------------------------------------------------


def test_complete_returns_true():
    td = TaskDispatcher(agents=["a1"])
    task = td.submit("x")
    assert td.complete(task.task_id, "done") is True


def test_complete_sets_status_completed():
    td = TaskDispatcher(agents=["a1"])
    task = td.submit("x")
    td.complete(task.task_id)
    assert task.status == DispatchStatus.COMPLETED


def test_complete_stores_result():
    td = TaskDispatcher(agents=["a1"])
    task = td.submit("x")
    td.complete(task.task_id, result="output-data")
    assert task.result == "output-data"


def test_complete_unknown_task_returns_false():
    td = TaskDispatcher(agents=["a1"])
    assert td.complete("no-task") is False


# ---------------------------------------------------------------------------
# TaskDispatcher.fail
# ---------------------------------------------------------------------------


def test_fail_returns_true():
    td = TaskDispatcher(agents=["a1"])
    task = td.submit("y")
    assert td.fail(task.task_id, "timeout") is True


def test_fail_sets_status_failed():
    td = TaskDispatcher(agents=["a1"])
    task = td.submit("y")
    td.fail(task.task_id)
    assert task.status == DispatchStatus.FAILED


def test_fail_stores_reason():
    td = TaskDispatcher(agents=["a1"])
    task = td.submit("y")
    td.fail(task.task_id, reason="OOM")
    assert task.result == "OOM"


def test_fail_unknown_task_returns_false():
    td = TaskDispatcher(agents=["a1"])
    assert td.fail("bad-id") is False


# ---------------------------------------------------------------------------
# TaskDispatcher.pending_tasks
# ---------------------------------------------------------------------------


def test_pending_tasks_returns_pending():
    td = TaskDispatcher(agents=["a1"])
    t1 = td.submit("one")
    t2 = td.submit("two")
    td.assign(t1.task_id, "a1")
    pending = td.pending_tasks()
    assert len(pending) == 1
    assert pending[0].task_id == t2.task_id


def test_pending_tasks_empty_after_all_assigned():
    td = TaskDispatcher(agents=["a1", "a2"])
    t1 = td.submit("one")
    t2 = td.submit("two")
    td.assign(t1.task_id, "a1")
    td.assign(t2.task_id, "a2")
    assert td.pending_tasks() == []


# ---------------------------------------------------------------------------
# TaskDispatcher.agent_load
# ---------------------------------------------------------------------------


def test_agent_load_initial_zero():
    td = TaskDispatcher(agents=["a1", "a2"])
    load = td.agent_load()
    assert load == {"a1": 0, "a2": 0}


def test_agent_load_counts_assigned():
    td = TaskDispatcher(agents=["a1", "a2"])
    t1 = td.submit("task1")
    t2 = td.submit("task2")
    td.assign(t1.task_id, "a1")
    td.assign(t2.task_id, "a1")
    load = td.agent_load()
    assert load["a1"] == 2
    assert load["a2"] == 0


# ---------------------------------------------------------------------------
# TaskDispatcher.add_agent / remove_agent
# ---------------------------------------------------------------------------


def test_add_agent_allows_assignment():
    td = TaskDispatcher(agents=[])
    td.add_agent("new-agent")
    task = td.submit("work")
    assert td.assign(task.task_id, "new-agent") is True


def test_remove_agent_returns_true():
    td = TaskDispatcher(agents=["a1"])
    assert td.remove_agent("a1") is True


def test_remove_agent_prevents_assignment():
    td = TaskDispatcher(agents=["a1"])
    td.remove_agent("a1")
    task = td.submit("work")
    assert td.assign(task.task_id, "a1") is False


def test_remove_agent_missing_returns_false():
    td = TaskDispatcher(agents=["a1"])
    assert td.remove_agent("ghost") is False


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in TASK_DISPATCHER_REGISTRY


def test_registry_default_is_task_dispatcher_class():
    assert TASK_DISPATCHER_REGISTRY["default"] is TaskDispatcher
