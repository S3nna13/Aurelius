"""Tests for trust boundary validator."""
from __future__ import annotations

import pytest

from src.security.trust_boundary import TrustBoundary, TrustBoundaryValidator


class TestTrustBoundaryValidator:
    def test_allows_authorized_caller(self):
        tbv = TrustBoundaryValidator()
        tbv.register(TrustBoundary(
            name="mcp", allowed_caller_prefixes=["tool:"], allowed_methods=["call"]
        ))
        ok, _ = tbv.check("mcp", "tool:retriever", "call")
        assert ok is True

    def test_rejects_unknown_boundary(self):
        tbv = TrustBoundaryValidator()
        ok, msg = tbv.check("unknown", "caller", "method")
        assert ok is False
        assert "unknown" in msg

    def test_rejects_unauthorized_caller(self):
        tbv = TrustBoundaryValidator()
        tbv.register(TrustBoundary(
            name="mcp", allowed_caller_prefixes=["tool:"], allowed_methods=["call"]
        ))
        ok, _ = tbv.check("mcp", "agent:bad", "call")
        assert ok is False

    def test_rejects_unauthorized_method(self):
        tbv = TrustBoundaryValidator()
        tbv.register(TrustBoundary(
            name="mcp", allowed_caller_prefixes=["tool:"], allowed_methods=["call"]
        ))
        ok, _ = tbv.check("mcp", "tool:retriever", "exec")
        assert ok is False

    def test_unregister(self):
        tbv = TrustBoundaryValidator()
        tbv.register(TrustBoundary(name="x"))
        tbv.unregister("x")
        ok, _ = tbv.check("x", "", "")
        assert ok is False