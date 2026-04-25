import json
import time

import pytest

from src.agent.run_store import RunStatus
from src.agent.sqlite_run_store import SQLiteRunStore


@pytest.fixture()
def store() -> SQLiteRunStore:
    return SQLiteRunStore(db_path=":memory:")


def test_create_returns_pending(store):
    run = store.create("agent-1")
    assert run.status == RunStatus.PENDING


def test_create_assigns_unique_uuid(store):
    r1 = store.create("agent-1")
    r2 = store.create("agent-1")
    assert r1.run_id != r2.run_id
    assert len(r1.run_id) == 36


def test_create_default_max_retries(store):
    run = store.create("agent-1")
    assert run.max_retries == 3
    assert run.retry_budget == 3


def test_create_custom_max_retries(store):
    run = store.create("agent-1", max_retries=7)
    assert run.max_retries == 7
    assert run.retry_budget == 7


def test_create_sets_agent_id(store):
    run = store.create("my-agent")
    assert run.agent_id == "my-agent"


def test_get_existing(store):
    run = store.create("agent-1")
    fetched = store.get(run.run_id)
    assert fetched is not None
    assert fetched.run_id == run.run_id
    assert fetched.agent_id == run.agent_id


def test_get_missing_returns_none(store):
    assert store.get("nonexistent-id") is None


def test_update_persists_changes(store):
    run = store.create("agent-1")
    run.error = "something went wrong"
    store.update(run)
    fetched = store.get(run.run_id)
    assert fetched.error == "something went wrong"


def test_transition_pending_to_running(store):
    run = store.create("agent-1")
    updated = store.transition(run.run_id, RunStatus.RUNNING)
    assert updated.status == RunStatus.RUNNING
    assert store.get(run.run_id).status == RunStatus.RUNNING


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


def test_transition_completed_is_terminal(store):
    run = store.create("agent-1")
    store.transition(run.run_id, RunStatus.RUNNING)
    store.transition(run.run_id, RunStatus.COMPLETED)
    with pytest.raises(ValueError):
        store.transition(run.run_id, RunStatus.RUNNING)


def test_transition_failed_decrements_budget(store):
    run = store.create("agent-1", max_retries=2)
    store.transition(run.run_id, RunStatus.RUNNING)
    store.transition(run.run_id, RunStatus.FAILED)
    updated = store.transition(run.run_id, RunStatus.RUNNING)
    assert updated.retry_budget == 1


def test_transition_failed_no_budget_raises(store):
    run = store.create("agent-1", max_retries=0)
    store.transition(run.run_id, RunStatus.RUNNING)
    store.transition(run.run_id, RunStatus.FAILED)
    with pytest.raises(ValueError):
        store.transition(run.run_id, RunStatus.RUNNING)


def test_transition_unknown_run_raises_key_error(store):
    with pytest.raises(KeyError):
        store.transition("bad-id", RunStatus.RUNNING)


def test_checkpoint_persists_entry(store):
    run = store.create("agent-1")
    store.transition(run.run_id, RunStatus.RUNNING)
    store.checkpoint(run.run_id, step=1, state={"loss": 0.42})
    fetched = store.get(run.run_id)
    assert len(fetched.checkpoint_log) == 1
    assert fetched.checkpoint_log[0]["step"] == 1
    assert fetched.checkpoint_log[0]["state"] == {"loss": 0.42}
    assert "ts" in fetched.checkpoint_log[0]


def test_checkpoint_multiple_entries(store):
    run = store.create("agent-1")
    store.transition(run.run_id, RunStatus.RUNNING)
    for i in range(4):
        store.checkpoint(run.run_id, step=i, state={"i": i})
    fetched = store.get(run.run_id)
    assert len(fetched.checkpoint_log) == 4


def test_checkpoint_unknown_run_raises(store):
    with pytest.raises(KeyError):
        store.checkpoint("bad-id", step=0, state={})


def test_retry_transitions_to_running(store):
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
    pending_ids = [r.run_id for r in pending]
    running_ids = [r.run_id for r in running]
    assert r2.run_id in pending_ids
    assert r1.run_id in running_ids


def test_list_all_returns_all(store):
    r1 = store.create("agent-1")
    r2 = store.create("agent-2")
    all_runs = store.list_all()
    ids = [r.run_id for r in all_runs]
    assert r1.run_id in ids
    assert r2.run_id in ids


def test_delete_removes_run(store):
    run = store.create("agent-1")
    store.delete(run.run_id)
    assert store.get(run.run_id) is None


def test_delete_nonexistent_is_noop(store):
    store.delete("nonexistent-id")


def test_list_all_empty_initially(store):
    assert store.list_all() == []


def test_updated_at_changes_on_transition(store):
    run = store.create("agent-1")
    before = run.updated_at
    time.sleep(0.01)
    updated = store.transition(run.run_id, RunStatus.RUNNING)
    assert updated.updated_at > before


def test_durability_across_get_calls(store):
    run = store.create("durable-agent")
    store.transition(run.run_id, RunStatus.RUNNING)
    store.transition(run.run_id, RunStatus.FAILED)
    fetched = store.get(run.run_id)
    assert fetched.status == RunStatus.FAILED
    assert fetched.agent_id == "durable-agent"
