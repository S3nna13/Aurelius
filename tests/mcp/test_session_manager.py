"""Tests for src/mcp/session_manager.py — 10+ unit tests, no GPU."""

from __future__ import annotations

import time

import pytest

from src.mcp.session_manager import (
    MCP_REGISTRY,
    MCPSession,
    MCPSessionManager,
    SessionState,
)


@pytest.fixture()
def manager() -> MCPSessionManager:
    """Fresh MCPSessionManager for each test."""
    return MCPSessionManager()


# ---------------------------------------------------------------------------
# create_session()
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_returns_mcp_session(self, manager):
        session = manager.create_session()
        assert isinstance(session, MCPSession)

    def test_session_is_active_after_creation(self, manager):
        session = manager.create_session()
        assert session.state == SessionState.ACTIVE

    def test_session_has_unique_id(self, manager):
        s1 = manager.create_session()
        s2 = manager.create_session()
        assert s1.session_id != s2.session_id

    def test_metadata_stored(self, manager):
        session = manager.create_session(metadata={"user": "alice"})
        assert session.metadata["user"] == "alice"

    def test_no_metadata_defaults_empty_dict(self, manager):
        session = manager.create_session()
        assert session.metadata == {}

    def test_created_at_recent(self, manager):
        before = time.time()
        session = manager.create_session()
        after = time.time()
        assert before <= session.created_at <= after

    def test_session_stored_in_manager(self, manager):
        session = manager.create_session()
        assert manager.get_session(session.session_id) is session


# ---------------------------------------------------------------------------
# get_session()
# ---------------------------------------------------------------------------


class TestGetSession:
    def test_returns_none_for_unknown_id(self, manager):
        assert manager.get_session("nonexistent") is None

    def test_returns_session_by_id(self, manager):
        session = manager.create_session()
        assert manager.get_session(session.session_id) is session


# ---------------------------------------------------------------------------
# update_activity()
# ---------------------------------------------------------------------------


class TestUpdateActivity:
    def test_bumps_last_activity(self, manager):
        session = manager.create_session()
        old_activity = session.last_activity
        time.sleep(0.01)
        manager.update_activity(session.session_id)
        assert session.last_activity > old_activity

    def test_idle_session_becomes_active(self, manager):
        session = manager.create_session()
        session.state = SessionState.IDLE
        manager.update_activity(session.session_id)
        assert session.state == SessionState.ACTIVE

    def test_update_nonexistent_session_no_error(self, manager):
        # Should silently do nothing
        manager.update_activity("ghost-id")


# ---------------------------------------------------------------------------
# close_session()
# ---------------------------------------------------------------------------


class TestCloseSession:
    def test_session_becomes_closed(self, manager):
        session = manager.create_session()
        manager.close_session(session.session_id)
        assert session.state == SessionState.CLOSED

    def test_close_nonexistent_no_error(self, manager):
        manager.close_session("ghost-id")

    def test_closed_session_not_in_list_active(self, manager):
        session = manager.create_session()
        manager.close_session(session.session_id)
        assert session not in manager.list_active()


# ---------------------------------------------------------------------------
# prune_idle()
# ---------------------------------------------------------------------------


class TestPruneIdle:
    def test_prunes_old_session(self, manager):
        session = manager.create_session()
        # Backdate last_activity far into the past
        session.last_activity = time.time() - 1000.0
        pruned = manager.prune_idle(idle_timeout=300.0)
        assert pruned == 1
        assert session.state == SessionState.CLOSED

    def test_does_not_prune_recent_session(self, manager):
        manager.create_session()
        pruned = manager.prune_idle(idle_timeout=300.0)
        assert pruned == 0

    def test_does_not_prune_already_closed(self, manager):
        session = manager.create_session()
        manager.close_session(session.session_id)
        session.last_activity = time.time() - 1000.0
        pruned = manager.prune_idle(idle_timeout=300.0)
        assert pruned == 0

    def test_returns_count_pruned(self, manager):
        for _ in range(3):
            s = manager.create_session()
            s.last_activity = time.time() - 600.0
        pruned = manager.prune_idle(idle_timeout=300.0)
        assert pruned == 3


# ---------------------------------------------------------------------------
# list_active()
# ---------------------------------------------------------------------------


class TestListActive:
    def test_empty_initially(self, manager):
        assert manager.list_active() == []

    def test_active_sessions_listed(self, manager):
        s1 = manager.create_session()
        s2 = manager.create_session()
        active = manager.list_active()
        assert s1 in active and s2 in active

    def test_closed_not_listed(self, manager):
        session = manager.create_session()
        manager.close_session(session.session_id)
        assert session not in manager.list_active()

    def test_idle_sessions_listed(self, manager):
        session = manager.create_session()
        session.state = SessionState.IDLE
        assert session in manager.list_active()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_has_session_manager(self):
        assert "session_manager" in MCP_REGISTRY

    def test_registry_instance_is_manager(self):
        assert isinstance(MCP_REGISTRY["session_manager"], MCPSessionManager)
