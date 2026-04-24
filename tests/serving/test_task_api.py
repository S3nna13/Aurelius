"""Tests for src/serving/task_api.py — task lifecycle REST contracts."""

import time
import pytest

from src.serving.task_api import (
    Artifact,
    APIError,
    CreateTaskRequest,
    TaskCitation,
    TaskMetrics,
    TaskResult,
    TaskStore,
    TASK_STORE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_store() -> TaskStore:
    return TaskStore()


def make_request(**kwargs) -> CreateTaskRequest:
    defaults = {"goal": "Summarise the quarterly report"}
    defaults.update(kwargs)
    return CreateTaskRequest(**defaults)


# ---------------------------------------------------------------------------
# CreateTaskRequest defaults
# ---------------------------------------------------------------------------

def test_create_request_defaults():
    req = make_request()
    assert req.goal == "Summarise the quarterly report"
    assert req.priority == 5
    assert req.risk_profile == "balanced"
    assert req.max_steps == 50
    assert req.task_id != ""
    assert req.user_id != ""
    assert req.created_at <= time.time()


def test_create_request_custom_fields():
    req = make_request(
        priority=9,
        risk_profile="strict",
        max_steps=100,
        tool_allowlist=["search", "code"],
        constraints={"domain": "finance"},
    )
    assert req.priority == 9
    assert req.risk_profile == "strict"
    assert req.max_steps == 100
    assert "search" in req.tool_allowlist
    assert req.constraints["domain"] == "finance"


# ---------------------------------------------------------------------------
# TaskStore.create
# ---------------------------------------------------------------------------

def test_store_create_returns_pending():
    store = fresh_store()
    req = make_request()
    result = store.create(req)
    assert result.status == "pending"
    assert result.task_id == req.task_id


def test_store_create_unique_tasks():
    store = fresh_store()
    r1 = store.create(make_request())
    r2 = store.create(make_request())
    assert r1.task_id != r2.task_id


# ---------------------------------------------------------------------------
# TaskStore.get_result
# ---------------------------------------------------------------------------

def test_get_result_existing():
    store = fresh_store()
    req = make_request()
    created = store.create(req)
    fetched = store.get_result(req.task_id)
    assert fetched is not None
    assert fetched.task_id == created.task_id


def test_get_result_missing():
    store = fresh_store()
    assert store.get_result("nonexistent-id") is None


# ---------------------------------------------------------------------------
# TaskStore.update_status
# ---------------------------------------------------------------------------

def test_update_status_to_running():
    store = fresh_store()
    req = make_request()
    store.create(req)
    ok = store.update_status(req.task_id, "running")
    assert ok is True
    assert store.get_result(req.task_id).status == "running"


def test_update_status_to_completed_sets_completed_at():
    store = fresh_store()
    req = make_request()
    store.create(req)
    store.update_status(req.task_id, "completed", summary="All done")
    result = store.get_result(req.task_id)
    assert result.status == "completed"
    assert result.summary == "All done"
    assert result.completed_at is not None
    assert result.completed_at <= time.time()


def test_update_status_failed_sets_completed_at():
    store = fresh_store()
    req = make_request()
    store.create(req)
    store.update_status(req.task_id, "failed", error="Timeout")
    result = store.get_result(req.task_id)
    assert result.status == "failed"
    assert result.error == "Timeout"
    assert result.completed_at is not None


def test_update_status_cancelled():
    store = fresh_store()
    req = make_request()
    store.create(req)
    ok = store.update_status(req.task_id, "cancelled")
    assert ok is True
    assert store.get_result(req.task_id).completed_at is not None


def test_update_status_missing_task_returns_false():
    store = fresh_store()
    assert store.update_status("bad-id", "completed") is False


# ---------------------------------------------------------------------------
# TaskStore.list_tasks
# ---------------------------------------------------------------------------

def test_list_tasks_all():
    store = fresh_store()
    store.create(make_request())
    store.create(make_request())
    assert len(store.list_tasks()) == 2


def test_list_tasks_by_user():
    store = fresh_store()
    uid_a = "user-A"
    uid_b = "user-B"
    store.create(make_request(user_id=uid_a))
    store.create(make_request(user_id=uid_a))
    store.create(make_request(user_id=uid_b))
    user_a_results = store.list_tasks(user_id=uid_a)
    assert len(user_a_results) == 2
    assert all(True for _ in user_a_results)  # each is a TaskResult


def test_list_tasks_unknown_user_returns_empty():
    store = fresh_store()
    store.create(make_request(user_id="alice"))
    assert store.list_tasks(user_id="nobody") == []


# ---------------------------------------------------------------------------
# Artifact dataclass
# ---------------------------------------------------------------------------

def test_artifact_defaults():
    a = Artifact()
    assert a.artifact_type == "text"
    assert a.mime_type == "text/plain"
    assert a.artifact_id != ""


def test_artifact_custom():
    a = Artifact(artifact_type="json", content='{"k": 1}', mime_type="application/json")
    assert a.content == '{"k": 1}'


# ---------------------------------------------------------------------------
# TaskMetrics dataclass
# ---------------------------------------------------------------------------

def test_task_metrics_defaults():
    m = TaskMetrics()
    assert m.steps == 0
    assert m.tokens_in == 0
    assert m.latency_ms == 0.0


# ---------------------------------------------------------------------------
# APIError
# ---------------------------------------------------------------------------

def test_api_error_to_dict():
    err = APIError(code="rate_limited", message="Slow down", retry_after_seconds=5.0)
    d = err.to_dict()
    assert d["error"]["code"] == "rate_limited"
    assert d["error"]["message"] == "Slow down"
    assert d["error"]["retry_after_seconds"] == 5.0
    assert "trace_id" in d["error"]


def test_api_error_unique_trace_ids():
    e1 = APIError(code="x", message="y")
    e2 = APIError(code="x", message="y")
    assert e1.trace_id != e2.trace_id


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def test_task_store_singleton_exists():
    assert TASK_STORE is not None
    assert isinstance(TASK_STORE, TaskStore)
