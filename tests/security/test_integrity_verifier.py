"""Tests for integrity verifier."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.security.integrity_verifier import IntegrityVerifier


class TestIntegrityVerifier:
    def test_snapshot_and_verify(self):
        iv = IntegrityVerifier()
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("hello world")
            path = f.name
        iv.snapshot(path)
        ok, msg = iv.verify(path)
        assert ok is True
        assert msg == "ok"

    def test_verify_no_snapshot(self):
        iv = IntegrityVerifier()
        ok, msg = iv.verify("/nonexistent/file")
        assert ok is False
        assert "no snapshot" in msg

    def test_verify_modified(self):
        iv = IntegrityVerifier()
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("original")
            path = f.name
        iv.snapshot(path)
        with open(path, "w") as f:
            f.write("modified")
        ok, _ = iv.verify(path)
        assert ok is False

    def test_verify_missing(self):
        iv = IntegrityVerifier()
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("data")
            path = f.name
        iv.snapshot(path)
        os.remove(path)
        ok, msg = iv.verify(path)
        assert ok is False
        assert "missing" in msg