"""Tests for src.data.usage_pipeline.

Requires the ``aurelius_usage_store`` Rust extension to be built::

    cd crates/usage-store && maturin develop
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC

import pytest

pytest.importorskip("aurelius_usage_store")

from src.data.usage_pipeline import UsagePipeline, get_pipeline


@pytest.fixture
def pipe():
    with tempfile.TemporaryDirectory() as td:
        yield UsagePipeline(os.path.join(td, "usage.redb"))


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_log_returns_ulid(pipe):
    eid = pipe.log("u1", "test", {"x": 1})
    assert isinstance(eid, str)
    assert len(eid) == 26


def test_count_and_query(pipe):
    pipe.log("u1", "chat_completion", {"model": "m1"})
    pipe.log("u1", "chat_completion", {"model": "m2"})
    pipe.log("u2", "image_gen", {"model": "sd"})

    assert pipe.count("u1") == 2
    assert pipe.count("u2") == 1
    assert pipe.count("u3") == 0

    events = pipe.query("u1")
    assert len(events) == 2
    assert events[0]["event_type"] == "chat_completion"


def test_query_with_event_type_filter(pipe):
    pipe.log("u1", "chat_completion", {"m": 1})
    pipe.log("u1", "tool_call", {"t": 1})
    pipe.log("u1", "chat_completion", {"m": 2})

    chats = pipe.query("u1", event_type="chat_completion")
    assert len(chats) == 2
    assert all(e["event_type"] == "chat_completion" for e in chats)


def test_query_limit(pipe):
    for i in range(5):
        pipe.log("u1", "chat_completion", {"i": i})
    events = pipe.query("u1", limit=2)
    assert len(events) == 2


def test_query_since_filter(pipe):
    from datetime import datetime

    pipe.log("u1", "old", {"v": 1})
    cutoff = datetime.now(UTC).isoformat()
    pipe.log("u1", "new", {"v": 2})

    events = pipe.query("u1", since=cutoff)
    assert len(events) == 1
    assert events[0]["payload"]["v"] == 2


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def test_log_chat(pipe):
    eid = pipe.log_chat("u1", "aurelius-v1", 10, 20, latency_ms=45.2)
    assert isinstance(eid, str)
    events = pipe.recent_chat_events("u1")
    assert len(events) == 1
    p = events[0]["payload"]
    assert p["model"] == "aurelius-v1"
    assert p["tokens_in"] == 10
    assert p["tokens_out"] == 20
    assert p["latency_ms"] == 45.2


def test_log_tool_call(pipe):
    eid = pipe.log_tool_call("u1", "shell", True, duration_ms=12.0)
    assert isinstance(eid, str)
    events = pipe.query("u1", event_type="tool_call")
    assert events[0]["payload"]["tool"] == "shell"
    assert events[0]["payload"]["success"] is True


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def test_get_pipeline_singleton():
    with tempfile.TemporaryDirectory() as td:
        p1 = get_pipeline(os.path.join(td, "singleton.redb"))
        p2 = get_pipeline(os.path.join(td, "singleton.redb"))
        assert p1 is p2
