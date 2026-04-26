import json

import pytest

from src.workflow.event_sourcing import (
    WORKFLOW_EVENT_STORE,
    EventType,
    WorkflowEvent,
    WorkflowEventStore,
)


@pytest.fixture()
def store() -> WorkflowEventStore:
    return WorkflowEventStore()


def test_append_returns_workflow_event(store):
    event = store.append("wf-1", EventType.WORKFLOW_STARTED)
    assert isinstance(event, WorkflowEvent)


def test_append_assigns_unique_event_id(store):
    e1 = store.append("wf-1", EventType.WORKFLOW_STARTED)
    e2 = store.append("wf-1", EventType.STEP_STARTED, step_id="s1")
    assert e1.event_id != e2.event_id
    assert len(e1.event_id) == 36


def test_append_stores_correct_fields(store):
    event = store.append("wf-2", EventType.STEP_STARTED, step_id="step-a", payload={"x": 1})
    assert event.workflow_id == "wf-2"
    assert event.event_type == EventType.STEP_STARTED
    assert event.step_id == "step-a"
    assert event.payload == {"x": 1}
    assert event.timestamp > 0


def test_append_default_payload_is_empty_dict(store):
    event = store.append("wf-1", EventType.WORKFLOW_STARTED)
    assert event.payload == {}


def test_get_history_empty_for_unknown(store):
    assert store.get_history("nonexistent") == []


def test_get_history_returns_all_events_in_order(store):
    store.append("wf-3", EventType.WORKFLOW_STARTED)
    store.append("wf-3", EventType.STEP_STARTED, step_id="s1")
    store.append("wf-3", EventType.STEP_COMPLETED, step_id="s1")
    history = store.get_history("wf-3")
    assert len(history) == 3
    assert history[0].event_type == EventType.WORKFLOW_STARTED
    assert history[1].event_type == EventType.STEP_STARTED
    assert history[2].event_type == EventType.STEP_COMPLETED


def test_get_history_isolates_workflows(store):
    store.append("wf-a", EventType.WORKFLOW_STARTED)
    store.append("wf-b", EventType.WORKFLOW_STARTED)
    assert len(store.get_history("wf-a")) == 1
    assert len(store.get_history("wf-b")) == 1


def test_replay_unknown_workflow_returns_unknown_status(store):
    result = store.replay("no-such-wf")
    assert result["status"] == "unknown"
    assert result["completed_steps"] == []
    assert result["failed_steps"] == []
    assert result["current_step"] is None


def test_replay_workflow_started_sets_running(store):
    store.append("wf-r", EventType.WORKFLOW_STARTED)
    result = store.replay("wf-r")
    assert result["status"] == "running"


def test_replay_step_started_sets_current_step(store):
    store.append("wf-r", EventType.WORKFLOW_STARTED)
    store.append("wf-r", EventType.STEP_STARTED, step_id="s1")
    result = store.replay("wf-r")
    assert result["current_step"] == "s1"


def test_replay_step_completed_adds_to_completed(store):
    store.append("wf-r", EventType.WORKFLOW_STARTED)
    store.append("wf-r", EventType.STEP_STARTED, step_id="s1")
    store.append("wf-r", EventType.STEP_COMPLETED, step_id="s1")
    result = store.replay("wf-r")
    assert "s1" in result["completed_steps"]
    assert result["current_step"] is None


def test_replay_step_failed_adds_to_failed(store):
    store.append("wf-r", EventType.WORKFLOW_STARTED)
    store.append("wf-r", EventType.STEP_STARTED, step_id="s2")
    store.append("wf-r", EventType.STEP_FAILED, step_id="s2")
    result = store.replay("wf-r")
    assert "s2" in result["failed_steps"]
    assert result["current_step"] is None


def test_replay_workflow_completed_sets_completed(store):
    store.append("wf-r", EventType.WORKFLOW_STARTED)
    store.append("wf-r", EventType.WORKFLOW_COMPLETED)
    result = store.replay("wf-r")
    assert result["status"] == "completed"


def test_replay_workflow_failed_sets_failed(store):
    store.append("wf-r", EventType.WORKFLOW_STARTED)
    store.append("wf-r", EventType.WORKFLOW_FAILED)
    result = store.replay("wf-r")
    assert result["status"] == "failed"


def test_list_workflows_empty_initially(store):
    assert store.list_workflows() == []


def test_list_workflows_returns_known_ids(store):
    store.append("wf-x", EventType.WORKFLOW_STARTED)
    store.append("wf-y", EventType.WORKFLOW_STARTED)
    wfs = store.list_workflows()
    assert "wf-x" in wfs
    assert "wf-y" in wfs


def test_prune_removes_events_and_returns_count(store):
    store.append("wf-p", EventType.WORKFLOW_STARTED)
    store.append("wf-p", EventType.STEP_STARTED, step_id="s1")
    count = store.prune("wf-p")
    assert count == 2
    assert store.get_history("wf-p") == []


def test_prune_nonexistent_returns_zero(store):
    assert store.prune("no-such-wf") == 0


def test_export_json_is_valid_json(store):
    store.append("wf-e", EventType.WORKFLOW_STARTED)
    store.append("wf-e", EventType.STEP_STARTED, step_id="s1")
    raw = store.export_json("wf-e")
    data = json.loads(raw)
    assert isinstance(data, list)
    assert len(data) == 2


def test_export_json_contains_correct_fields(store):
    store.append("wf-e2", EventType.WORKFLOW_STARTED, payload={"key": "val"})
    data = json.loads(store.export_json("wf-e2"))
    record = data[0]
    assert "event_id" in record
    assert "workflow_id" in record
    assert record["event_type"] == "workflow_started"
    assert record["payload"] == {"key": "val"}
    assert "timestamp" in record


def test_export_json_empty_workflow(store):
    raw = store.export_json("nonexistent")
    assert json.loads(raw) == []


def test_module_singleton_is_workflow_event_store():
    assert isinstance(WORKFLOW_EVENT_STORE, WorkflowEventStore)


def test_event_type_values():
    assert EventType.WORKFLOW_STARTED == "workflow_started"
    assert EventType.CHECKPOINT == "checkpoint"
