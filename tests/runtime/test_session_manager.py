"""Tests for src/runtime/session_manager.py — 28+ tests."""

from __future__ import annotations

import time

import pytest

from src.runtime.session_manager import (
    SESSION_MANAGER_REGISTRY,
    Session,
    SessionManager,
    SessionState,
)


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default_key(self):
        assert "default" in SESSION_MANAGER_REGISTRY

    def test_registry_default_is_manager_class(self):
        assert SESSION_MANAGER_REGISTRY["default"] is SessionManager


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

class TestSessionDataclass:
    def test_session_default_state(self):
        s = Session("abc", "model1", 0.0, 0.0)
        assert s.state is SessionState.ACTIVE

    def test_session_default_metadata(self):
        s = Session("abc", "model1", 0.0, 0.0)
        assert s.metadata == {}

    def test_session_stores_fields(self):
        s = Session("sid", "mod", 1.0, 2.0, SessionState.IDLE, {"k": "v"})
        assert s.session_id == "sid"
        assert s.model_id == "mod"
        assert s.created_at == 1.0
        assert s.last_active == 2.0
        assert s.metadata == {"k": "v"}


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

class TestCreate:
    def test_create_returns_session(self):
        mgr = SessionManager()
        s = mgr.create("gpt2")
        assert isinstance(s, Session)

    def test_create_sets_model_id(self):
        mgr = SessionManager()
        s = mgr.create("gpt2")
        assert s.model_id == "gpt2"

    def test_create_auto_session_id_length(self):
        mgr = SessionManager()
        s = mgr.create("gpt2")
        assert len(s.session_id) == 12

    def test_create_session_id_is_hex(self):
        mgr = SessionManager()
        s = mgr.create("gpt2")
        int(s.session_id, 16)  # raises if not valid hex

    def test_create_unique_session_ids(self):
        mgr = SessionManager()
        ids = {mgr.create("m").session_id for _ in range(20)}
        assert len(ids) == 20

    def test_create_state_is_active(self):
        mgr = SessionManager()
        s = mgr.create("gpt2")
        assert s.state is SessionState.ACTIVE

    def test_create_with_metadata(self):
        mgr = SessionManager()
        s = mgr.create("gpt2", metadata={"user": "alice"})
        assert s.metadata["user"] == "alice"

    def test_create_metadata_none_defaults_to_empty(self):
        mgr = SessionManager()
        s = mgr.create("gpt2", metadata=None)
        assert s.metadata == {}

    def test_create_raises_at_max_sessions(self):
        mgr = SessionManager(max_sessions=2)
        mgr.create("m")
        mgr.create("m")
        with pytest.raises(ValueError, match="max_sessions"):
            mgr.create("m")

    def test_create_timestamps_monotonic(self):
        mgr = SessionManager()
        before = time.monotonic()
        s = mgr.create("m")
        after = time.monotonic()
        assert before <= s.created_at <= after
        assert s.created_at == s.last_active


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_existing_session(self):
        mgr = SessionManager()
        s = mgr.create("m")
        assert mgr.get(s.session_id) is s

    def test_get_missing_returns_none(self):
        mgr = SessionManager()
        assert mgr.get("nonexistent") is None


# ---------------------------------------------------------------------------
# touch
# ---------------------------------------------------------------------------

class TestTouch:
    def test_touch_returns_true_for_existing(self):
        mgr = SessionManager()
        s = mgr.create("m")
        assert mgr.touch(s.session_id) is True

    def test_touch_returns_false_for_missing(self):
        mgr = SessionManager()
        assert mgr.touch("ghost") is False

    def test_touch_updates_last_active(self):
        mgr = SessionManager()
        s = mgr.create("m")
        old_ts = s.last_active
        time.sleep(0.01)
        mgr.touch(s.session_id)
        assert s.last_active > old_ts

    def test_touch_sets_state_active(self):
        mgr = SessionManager()
        s = mgr.create("m")
        s.state = SessionState.IDLE
        mgr.touch(s.session_id)
        assert s.state is SessionState.ACTIVE


# ---------------------------------------------------------------------------
# expire_idle
# ---------------------------------------------------------------------------

class TestExpireIdle:
    def test_expire_idle_marks_expired(self):
        mgr = SessionManager(idle_timeout_s=0.0)
        s = mgr.create("m")
        far_future = s.last_active + 1000.0
        mgr.expire_idle(now=far_future)
        assert s.state is SessionState.EXPIRED

    def test_expire_idle_returns_ids(self):
        mgr = SessionManager(idle_timeout_s=0.0)
        s1 = mgr.create("m")
        s2 = mgr.create("m")
        far_future = max(s1.last_active, s2.last_active) + 1000.0
        expired = mgr.expire_idle(now=far_future)
        assert set(expired) == {s1.session_id, s2.session_id}

    def test_expire_idle_skips_recent_sessions(self):
        mgr = SessionManager(idle_timeout_s=9999.0)
        s = mgr.create("m")
        expired = mgr.expire_idle(now=time.monotonic())
        assert s.session_id not in expired
        assert s.state is SessionState.ACTIVE

    def test_expire_idle_uses_monotonic_when_now_none(self):
        mgr = SessionManager(idle_timeout_s=0.001)
        s = mgr.create("m")
        time.sleep(0.05)
        expired = mgr.expire_idle()
        assert s.session_id in expired

    def test_expire_idle_skips_terminated(self):
        mgr = SessionManager(idle_timeout_s=0.0)
        s = mgr.create("m")
        s.state = SessionState.TERMINATED
        far_future = s.last_active + 1000.0
        expired = mgr.expire_idle(now=far_future)
        assert s.session_id not in expired
        assert s.state is SessionState.TERMINATED


# ---------------------------------------------------------------------------
# terminate
# ---------------------------------------------------------------------------

class TestTerminate:
    def test_terminate_returns_true(self):
        mgr = SessionManager()
        s = mgr.create("m")
        assert mgr.terminate(s.session_id) is True

    def test_terminate_sets_state(self):
        mgr = SessionManager()
        s = mgr.create("m")
        mgr.terminate(s.session_id)
        assert s.state is SessionState.TERMINATED

    def test_terminate_returns_false_for_missing(self):
        mgr = SessionManager()
        assert mgr.terminate("ghost") is False


# ---------------------------------------------------------------------------
# active_count
# ---------------------------------------------------------------------------

class TestActiveCount:
    def test_active_count_initial(self):
        mgr = SessionManager()
        mgr.create("m")
        mgr.create("m")
        assert mgr.active_count() == 2

    def test_active_count_excludes_expired(self):
        mgr = SessionManager(idle_timeout_s=0.0)
        s1 = mgr.create("m")
        s2 = mgr.create("m")
        far_future = max(s1.last_active, s2.last_active) + 1000.0
        mgr.expire_idle(now=far_future)
        assert mgr.active_count() == 0

    def test_active_count_excludes_terminated(self):
        mgr = SessionManager()
        s = mgr.create("m")
        mgr.terminate(s.session_id)
        assert mgr.active_count() == 0


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_cleanup_removes_terminated(self):
        mgr = SessionManager()
        s = mgr.create("m")
        mgr.terminate(s.session_id)
        mgr.cleanup()
        assert mgr.get(s.session_id) is None

    def test_cleanup_removes_expired(self):
        mgr = SessionManager(idle_timeout_s=0.0)
        s = mgr.create("m")
        mgr.expire_idle(now=s.last_active + 1000.0)
        mgr.cleanup()
        assert mgr.get(s.session_id) is None

    def test_cleanup_returns_count(self):
        mgr = SessionManager(idle_timeout_s=0.0)
        s1 = mgr.create("m")
        s2 = mgr.create("m")
        s3 = mgr.create("m")
        mgr.terminate(s1.session_id)
        far_future = max(s2.last_active, s3.last_active) + 1000.0
        mgr.expire_idle(now=far_future)
        count = mgr.cleanup()
        assert count == 3

    def test_cleanup_keeps_active(self):
        mgr = SessionManager()
        s_active = mgr.create("m")
        s_dead = mgr.create("m")
        mgr.terminate(s_dead.session_id)
        mgr.cleanup()
        assert mgr.get(s_active.session_id) is s_active

    def test_cleanup_allows_new_sessions_after(self):
        mgr = SessionManager(max_sessions=1)
        s = mgr.create("m")
        mgr.terminate(s.session_id)
        mgr.cleanup()
        new_s = mgr.create("m")  # should not raise
        assert new_s is not None
