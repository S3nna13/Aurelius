"""Tests for src/security/audit_trail.py — ≥28 tests."""

from __future__ import annotations

import json
import os
import time

import pytest

from src.security.audit_trail import (
    AuditEvent,
    AUDIT_TRAIL_REGISTRY,
    AuditTrail,
)


# ---------------------------------------------------------------------------
# AuditEvent
# ---------------------------------------------------------------------------

class TestAuditEvent:
    def test_event_has_auto_event_id(self):
        ev = AuditEvent(
            actor="alice",
            action="login",
            resource="portal",
            outcome="success",
            timestamp=1.0,
            metadata={},
        )
        assert ev.event_id
        assert isinstance(ev.event_id, str)
        assert len(ev.event_id) == 12

    def test_event_id_unique(self):
        ev1 = AuditEvent("a", "b", "c", "success", 1.0, {})
        ev2 = AuditEvent("a", "b", "c", "success", 1.0, {})
        assert ev1.event_id != ev2.event_id

    def test_frozen_actor(self):
        ev = AuditEvent("alice", "login", "portal", "success", 1.0, {})
        with pytest.raises((AttributeError, TypeError)):
            ev.actor = "bob"  # type: ignore[misc]

    def test_frozen_metadata(self):
        ev = AuditEvent("alice", "login", "portal", "success", 1.0, {})
        with pytest.raises((AttributeError, TypeError)):
            ev.metadata = {"key": "value"}  # type: ignore[misc]

    def test_to_dict_keys(self):
        ev = AuditEvent("alice", "login", "portal", "success", 1.0, {})
        d = ev.to_dict()
        for key in ("event_id", "actor", "action", "resource", "outcome", "timestamp", "metadata"):
            assert key in d


# ---------------------------------------------------------------------------
# AuditTrail — log
# ---------------------------------------------------------------------------

class TestAuditTrailLog:
    def test_log_creates_event(self):
        trail = AuditTrail()
        ev = trail.log("alice", "delete", "/files/x.txt")
        assert isinstance(ev, AuditEvent)

    def test_log_increments_len(self):
        trail = AuditTrail()
        assert len(trail) == 0
        trail.log("a", "b", "c")
        assert len(trail) == 1
        trail.log("a", "b", "c")
        assert len(trail) == 2

    def test_log_default_outcome_success(self):
        trail = AuditTrail()
        ev = trail.log("a", "b", "c")
        assert ev.outcome == "success"

    def test_log_custom_outcome(self):
        trail = AuditTrail()
        ev = trail.log("a", "b", "c", outcome="failure")
        assert ev.outcome == "failure"

    def test_log_metadata_stored(self):
        trail = AuditTrail()
        ev = trail.log("a", "b", "c", metadata={"ip": "1.2.3.4"})
        assert ev.metadata == {"ip": "1.2.3.4"}

    def test_log_none_metadata_becomes_empty_dict(self):
        trail = AuditTrail()
        ev = trail.log("a", "b", "c", metadata=None)
        assert ev.metadata == {}

    def test_log_timestamp_recent(self):
        before = time.time()
        trail = AuditTrail()
        ev = trail.log("a", "b", "c")
        after = time.time()
        assert before <= ev.timestamp <= after


# ---------------------------------------------------------------------------
# AuditTrail — max_events enforcement
# ---------------------------------------------------------------------------

class TestMaxEvents:
    def test_max_events_enforced(self):
        trail = AuditTrail(max_events=5)
        for i in range(10):
            trail.log("a", f"action_{i}", "res")
        assert len(trail) == 5

    def test_oldest_event_evicted(self):
        trail = AuditTrail(max_events=3)
        ev0 = trail.log("a", "first", "res")
        trail.log("a", "second", "res")
        trail.log("a", "third", "res")
        trail.log("a", "fourth", "res")
        events = trail.query()
        ids = [e.event_id for e in events]
        assert ev0.event_id not in ids


# ---------------------------------------------------------------------------
# AuditTrail — query
# ---------------------------------------------------------------------------

class TestAuditTrailQuery:
    def setup_method(self):
        self.trail = AuditTrail()
        self.trail.log("alice", "read", "/data/a.txt", outcome="success")
        self.trail.log("bob", "write", "/data/b.txt", outcome="success")
        self.trail.log("alice", "delete", "/data/c.txt", outcome="failure")
        self.trail.log("carol", "login", "/auth", outcome="success")

    def test_query_none_returns_all(self):
        results = self.trail.query()
        assert len(results) == 4

    def test_query_by_actor(self):
        results = self.trail.query(actor="alice")
        assert all(e.actor == "alice" for e in results)
        assert len(results) == 2

    def test_query_by_action(self):
        results = self.trail.query(action="delete")
        assert all(e.action == "delete" for e in results)
        assert len(results) == 1

    def test_query_by_outcome_failure(self):
        results = self.trail.query(outcome="failure")
        assert len(results) == 1
        assert results[0].actor == "alice"

    def test_query_by_actor_and_outcome(self):
        results = self.trail.query(actor="alice", outcome="success")
        assert len(results) == 1
        assert results[0].action == "read"

    def test_query_no_match_returns_empty(self):
        results = self.trail.query(actor="nobody")
        assert results == []

    def test_query_by_outcome_success_count(self):
        results = self.trail.query(outcome="success")
        assert len(results) == 3


# ---------------------------------------------------------------------------
# AuditTrail — since
# ---------------------------------------------------------------------------

class TestAuditTrailSince:
    def test_since_returns_events_at_or_after(self):
        trail = AuditTrail()
        t0 = time.time()
        trail.log("a", "b", "c")
        trail.log("a", "b", "c")
        results = trail.since(t0)
        assert len(results) == 2

    def test_since_future_returns_empty(self):
        trail = AuditTrail()
        trail.log("a", "b", "c")
        far_future = time.time() + 9999
        assert trail.since(far_future) == []

    def test_since_filters_old_events(self):
        # Inject events with known timestamps directly to avoid timing flakiness
        from src.security.audit_trail import AuditEvent
        trail = AuditTrail()
        old_ev = AuditEvent("a", "b", "c", "success", 1000.0, {})
        new_ev = AuditEvent("a", "b", "c", "success", 2000.0, {})
        trail._events.extend([old_ev, new_ev])
        results = trail.since(1500.0)
        assert len(results) == 1
        assert results[0].timestamp == 2000.0


# ---------------------------------------------------------------------------
# AuditTrail — export_jsonl
# ---------------------------------------------------------------------------

class TestExportJsonl:
    def test_export_returns_count(self, tmp_path):
        trail = AuditTrail()
        trail.log("a", "b", "c")
        trail.log("d", "e", "f")
        out = tmp_path / "events.jsonl"
        count = trail.export_jsonl(str(out))
        assert count == 2

    def test_export_creates_file(self, tmp_path):
        trail = AuditTrail()
        trail.log("a", "b", "c")
        out = tmp_path / "out.jsonl"
        trail.export_jsonl(str(out))
        assert out.exists()

    def test_export_valid_json_lines(self, tmp_path):
        trail = AuditTrail()
        trail.log("alice", "login", "portal", metadata={"ip": "1.1.1.1"})
        trail.log("bob", "logout", "portal")
        out = tmp_path / "audit.jsonl"
        trail.export_jsonl(str(out))
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "event_id" in obj
            assert "actor" in obj

    def test_export_empty_trail(self, tmp_path):
        trail = AuditTrail()
        out = tmp_path / "empty.jsonl"
        count = trail.export_jsonl(str(out))
        assert count == 0
        assert out.read_text() == ""


# ---------------------------------------------------------------------------
# AuditTrail — integrity_hash
# ---------------------------------------------------------------------------

class TestIntegrityHash:
    def test_hash_is_hex_string(self):
        trail = AuditTrail()
        trail.log("a", "b", "c")
        h = trail.integrity_hash()
        assert isinstance(h, str)
        int(h, 16)  # raises if not valid hex

    def test_hash_length_is_64(self):
        trail = AuditTrail()
        trail.log("a", "b", "c")
        assert len(trail.integrity_hash()) == 64

    def test_hash_changes_on_new_event(self):
        trail = AuditTrail()
        trail.log("a", "b", "c")
        h1 = trail.integrity_hash()
        trail.log("d", "e", "f")
        h2 = trail.integrity_hash()
        assert h1 != h2

    def test_hash_deterministic_same_events(self):
        # Build two trails with the same sequence — hashes must match
        trail1 = AuditTrail()
        trail2 = AuditTrail()
        # Inject events with fixed event_ids to guarantee determinism
        from src.security.audit_trail import AuditEvent
        ev = AuditEvent("alice", "login", "portal", "success", 1000.0, {}, event_id="aabbccddeeff")
        trail1._events.append(ev)
        trail2._events.append(ev)
        assert trail1.integrity_hash() == trail2.integrity_hash()

    def test_empty_trail_hash_is_deterministic(self):
        t1 = AuditTrail()
        t2 = AuditTrail()
        assert t1.integrity_hash() == t2.integrity_hash()


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in AUDIT_TRAIL_REGISTRY

    def test_registry_default_instantiates(self):
        cls = AUDIT_TRAIL_REGISTRY["default"]
        trail = cls()
        assert isinstance(trail, AuditTrail)
