"""Tests for audit export."""
from __future__ import annotations

import json
import tempfile

import pytest

from src.security.audit_export import AuditExporter, AuditEntry


class TestAuditExporter:
    def test_record_and_export_json(self):
        ae = AuditExporter()
        ae.record(AuditEntry("2026-01-01T00:00:00Z", "alice", "read", "/secrets", "success"))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        ae.export_json(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["actor"] == "alice"

    def test_filter(self):
        ae = AuditExporter()
        ae.record(AuditEntry("t1", "alice", "read", "/a", "success"))
        ae.record(AuditEntry("t2", "bob", "write", "/b", "denied"))
        filtered = ae.filter(actor="alice")
        assert len(filtered) == 1
        filtered = ae.filter(result="denied")
        assert len(filtered) == 1

    def test_clear(self):
        ae = AuditExporter()
        ae.record(AuditEntry("t", "x", "r", "/", "ok"))
        ae.clear()
        assert ae.entries == []