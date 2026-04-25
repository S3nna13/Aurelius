"""Tests for session manager."""
from __future__ import annotations

import time

import pytest

from src.security.session_manager import SessionManager


class TestSessionManager:
    def test_create_and_validate(self):
        sm = SessionManager(duration_seconds=60)
        session = sm.create("alice")
        assert sm.validate(session.token) is not None

    def test_expired_session(self):
        sm = SessionManager(duration_seconds=0)
        session = sm.create("bob")
        time.sleep(0.01)
        assert sm.validate(session.token) is None

    def test_revoke_session(self):
        sm = SessionManager()
        session = sm.create("carol")
        sm.revoke(session.token)
        assert sm.validate(session.token) is None

    def test_active_count(self):
        sm = SessionManager(duration_seconds=60)
        sm.create("a")
        sm.create("b")
        assert sm.active_count() == 2

    def test_cleanup(self):
        sm = SessionManager(duration_seconds=0)
        sm.create("expired")
        time.sleep(0.01)
        n = sm.cleanup()
        assert n == 1