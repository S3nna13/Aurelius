import time

import pytest

from src.agent.run_store import (
    AGENT_RUN_STORE,
    AgentRunStore,
    RunStatus,
)


@pytest.fixture()
def store() -> AgentRunStore:
    return AgentRunStore()


def test_create_returns_pending(store):
    run = store.create("agent-1")
    assert run.status == RunStatus.PENDING


def test_create_assigns_uuid(store):
    r1 = store.create("agent-1")
    r2 = store.create("agent-1")
    assert r1.run_id != r2.run_id
    assert len(r1.run_id) == 36


def test_create_sets_max_retries(store):
    run = store.create("agent-1", max_retries=5)
    assert run.max_retries == 5
    assert run.retry_budget == 5


def test_create_default_max_retries(store):
    run = store.create("agent-1")
    assert run.max_retries == 3
    assert run.retry_budget == 3


def test_get_existing(store):
    run = store.create("agent-1")
    fetched = store.get(run.run_id)
    assert fetched is run


def test_get_missing_returns_none(store):
    assert store.get("nonexistent-id") is None


def test_transition_pending_to_running(store):
    run = store.create("agent-1")
    updated = store.transition(run.run_id, RunStatus.RUNNING)
    assert updated.status == RunStatus.RUNNING


def test_transition_running_to_completed(store):
    run = store.create("agent-1")
    store.transition(run.run_id, RunStatus.RUNNING)
    updated = store.transition(run.run_id, RunStatus.COMPLETED)
    assert updated.status == RunStatus.COMPLETED


def test_transition_running_to_failed(store):
    run = store.create("agent-1")
    store.transition(run.run_id, RunStatus.RUNNING)
    updated = store.transition(run.run_id, RunStatus.FAILED)
    assert updated.status == RunStatus.FAILED


def test_transition_running_to_paused(store):
    run = store.create("agent-1")
    store.transition(run.run_id, RunStatus.RUNNING)
    updated = store.transition(run.run_id, RunStatus.PAUSED)
    assert updated.status == RunStatus.PAUSED


def test_transition_paused_to_running(store):
    run = store.create("agent-1")
    store.transition(run.run_id, RunStatus.RUNNING)
    store.transition(run.run_id, RunStatus.PAUSED)
    updated = store.transition(run.run_id, RunStatus.RUNNING)
    assert updated.status == RunStatus.RUNNING


def test_transition_paused_to_cancelled(store):
    run = store.create("agent-1")
    store.transition(run.run_id, RunStatus.RUNNING)
    store.transition(run.run_id, RunStatus.PAUSED)
    updated = store.transition(run.run_id, RunStatus.CANCELLED)
    assert updated.status == RunStatus.CANCELLED


def test_transition_invalid_raises_value_error(store):
    run = store.create("agent-1")
    with pytest.raises(ValueError):
        store.transition(run.run_id, RunStatus.COMPLETED)


def test_transition_completed_terminal_raises(store):
    run = store.create("agent-1")
    store.transition(run.run_id, RunStatus.RUNNING)
    store.transition(run.run_id, RunStatus.COMPLETED)
    with pytest.raises(ValueError):
        store.transition(run.run_id, RunStatus.RUNNING)


def test_transition_failed_retry_decrements_budget(store):
    run = store.create("agent-1", max_retries=2)
    store.transition(run.run_id, RunStatus.RUNNING)
    store.transition(run.run_id, RunStatus.FAILED)
    store.transition(run.run_id, RunStatus.RUNNING)
    assert run.retry_budget == 1


def test_transition_failed_no_budget_raises(store):
    run = store.create("agent-1", max_retries=0)
    store.transition(run.run_id, RunStatus.RUNNING)
    store.transition(run.run_id, RunStatus.FAILED)
    with pytest.raises(ValueError):
        store.transition(run.run_id, RunStatus.RUNNING)


def test_transition_unknown_run_raises_key_error(store):
    with pytest.raises(KeyError):
        store.transition("bad-id", RunStatus.RUNNING)


def test_checkpoint_appends_entry(store):
    run = store.create("agent-1")
    store.transition(run.run_id, RunStatus.RUNNING)
    store.checkpoint(run.run_id, step=1, state={"loss": 0.5})
    assert len(run.checkpoint_log) == 1
    assert run.checkpoint_log[0]["step"] == 1
    assert run.checkpoint_log[0]["state"] == {"loss": 0.5}
    assert "ts" in run.checkpoint_log[0]


def test_checkpoint_multiple_entries(store):
    run = store.create("agent-1")
    store.transition(run.run_id, RunStatus.RUNNING)
    for i in range(5):
        store.checkpoint(run.run_id, step=i, state={"i": i})
    assert len(run.checkpoint_log) == 5


def test_checkpoint_unknown_run_raises(store):
    with pytest.raises(KeyError):
        store.checkpoint("bad-id", step=0, state={})


def test_retry_failed_transitions_to_running(store):
    run = store.create("agent-1", max_retries=2)
    store.transition(run.run_id, RunStatus.RUNNING)
    store.transition(run.run_id, RunStatus.FAILED)
    updated = store.retry(run.run_id)
    assert updated.status == RunStatus.RUNNING


def test_retry_exhausted_budget_raises_runtime_error(store):
    run = store.create("agent-1", max_retries=0)
    store.transition(run.run_id, RunStatus.RUNNING)
    store.transition(run.run_id, RunStatus.FAILED)
    with pytest.raises(RuntimeError):
        store.retry(run.run_id)


def test_retry_non_failed_raises_value_error(store):
    run = store.create("agent-1")
    with pytest.raises(ValueError):
        store.retry(run.run_id)


def test_list_by_status(store):
    r1 = store.create("agent-1")
    r2 = store.create("agent-2")
    store.transition(r1.run_id, RunStatus.RUNNING)
    pending = store.list_by_status(RunStatus.PENDING)
    running = store.list_by_status(RunStatus.RUNNING)
    assert r2 in pending
    assert r1 in running


def test_list_all(store):
    r1 = store.create("agent-1")
    r2 = store.create("agent-2")
    all_runs = store.list_all()
    assert r1 in all_runs
    assert r2 in all_runs


def test_updated_at_changes_on_transition(store):
    run = store.create("agent-1")
    before = run.updated_at
    time.sleep(0.01)
    store.transition(run.run_id, RunStatus.RUNNING)
    assert run.updated_at > before


def test_module_singleton_is_agent_run_store():
    assert isinstance(AGENT_RUN_STORE, AgentRunStore)
