"""Tests for AuditLogger and AuditLogEntry."""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from src.observability.audit_logger import AuditLogEntry, AuditLogger


class TestAuditLogEntry:
    def test_to_dict(self) -> None:
        entry = AuditLogEntry(
            timestamp=1.0,
            actor="user:42",
            action="read",
            resource="doc/1",
            status="success",
            metadata={"ip": "127.0.0.1"},
            trace_id="abc123",
        )
        d = entry.to_dict()
        assert d["actor"] == "user:42"
        assert d["action"] == "read"
        assert d["resource"] == "doc/1"
        assert d["status"] == "success"
        assert d["metadata"] == {"ip": "127.0.0.1"}
        assert d["trace_id"] == "abc123"

    def test_to_json_roundtrip(self) -> None:
        entry = AuditLogEntry(
            timestamp=1.5,
            actor="a",
            action="b",
            resource="c",
            status="d",
            metadata={"x": 1},
            trace_id="t1",
        )
        raw = entry.to_json()
        parsed = json.loads(raw)
        assert parsed["actor"] == "a"
        assert parsed["metadata"] == {"x": 1}

    def test_immutable(self) -> None:
        entry = AuditLogEntry(
            timestamp=1.0, actor="a", action="b", resource="c", status="d"
        )
        with pytest.raises(AttributeError):
            entry.actor = "x"  # type: ignore[misc]


class TestAuditLogger:
    def test_log_and_get_entries(self) -> None:
        logger = AuditLogger()
        logger.log("alice", "create", "file/1", "success")
        logger.log("bob", "delete", "file/2", "failure", metadata={"reason": "perms"})
        entries = logger.get_entries()
        assert len(entries) == 2
        assert entries[0].actor == "bob"
        assert entries[1].actor == "alice"

    def test_filter_by_actor(self) -> None:
        logger = AuditLogger()
        logger.log("alice", "create", "r1", "success")
        logger.log("bob", "create", "r2", "success")
        assert len(logger.get_entries(actor="alice")) == 1
        assert logger.get_entries(actor="alice")[0].actor == "alice"

    def test_filter_by_action(self) -> None:
        logger = AuditLogger()
        logger.log("a", "read", "r1", "success")
        logger.log("a", "write", "r2", "success")
        assert len(logger.get_entries(action="write")) == 1

    def test_filter_by_resource(self) -> None:
        logger = AuditLogger()
        logger.log("a", "x", "r1", "success")
        logger.log("a", "x", "r2", "success")
        assert len(logger.get_entries(resource="r1")) == 1

    def test_filter_by_status(self) -> None:
        logger = AuditLogger()
        logger.log("a", "x", "r1", "ok")
        logger.log("a", "x", "r2", "fail")
        assert len(logger.get_entries(status="ok")) == 1

    def test_filter_by_trace_id(self) -> None:
        logger = AuditLogger()
        logger.log("a", "x", "r1", "ok", trace_id="t1")
        logger.log("a", "x", "r2", "ok", trace_id="t2")
        assert len(logger.get_entries(trace_id="t2")) == 1

    def test_limit(self) -> None:
        logger = AuditLogger()
        for i in range(5):
            logger.log("a", "x", f"r{i}", "ok")
        assert len(logger.get_entries(limit=2)) == 2

    def test_buffer_size_eviction(self) -> None:
        logger = AuditLogger(buffer_size=3)
        for i in range(5):
            logger.log("a", "x", f"r{i}", "ok")
        assert len(logger) == 3
        assert logger.get_entries(limit=1)[0].resource == "r4"

    def test_flush(self) -> None:
        logger = AuditLogger()
        logger.log("a", "x", "r1", "ok")
        flushed = logger.flush()
        assert len(flushed) == 1
        assert len(logger) == 0

    def test_sink_called(self) -> None:
        calls: list[AuditLogEntry] = []

        def sink(entry: AuditLogEntry) -> None:
            calls.append(entry)

        logger = AuditLogger(sinks=[sink])
        logger.log("a", "x", "r1", "ok")
        assert len(calls) == 1
        assert calls[0].actor == "a"

    def test_sink_exception_ignored(self) -> None:
        def bad_sink(_entry: AuditLogEntry) -> None:
            raise RuntimeError("boom")

        logger = AuditLogger(sinks=[bad_sink])
        entry = logger.log("a", "x", "r1", "ok")
        assert entry.actor == "a"

    def test_add_remove_sink(self) -> None:
        logger = AuditLogger()

        def sink(_entry: AuditLogEntry) -> None:
            pass

        logger.add_sink(sink)
        assert len(logger._sinks) == 1
        logger.remove_sink(sink)
        assert len(logger._sinks) == 0

    def test_thread_safety(self) -> None:
        logger = AuditLogger(buffer_size=10_000)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    logger.log("a", "x", "r", "ok")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(logger) == 1_000  # buffer cycles: 10000 entries logged, buffer_size=1000

    def test_default_metadata(self) -> None:
        logger = AuditLogger()
        entry = logger.log("a", "x", "r", "ok")
        assert entry.metadata == {}

    def test_metadata_mutation_safe(self) -> None:
        logger = AuditLogger()
        meta: dict[str, Any] = {"key": [1, 2, 3]}
        entry = logger.log("a", "x", "r", "ok", metadata=meta)
        meta["key"].append(4)
        # entry should have captured a copy
        assert entry.metadata == {"key": [1, 2, 3]}
