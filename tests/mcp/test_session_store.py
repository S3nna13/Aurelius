"""Tests for src/mcp/session_store.py — ≥28 test cases."""

from __future__ import annotations

import time

import pytest

from src.mcp.session_store import SESSION_STORE_REGISTRY, MCPSession, SessionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_store(**kwargs) -> SessionStore:
    return SessionStore(**kwargs)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_returns_mcp_session(self):
        store = make_store()
        s = store.create("client-1")
        assert isinstance(s, MCPSession)

    def test_session_id_auto_assigned(self):
        store = make_store()
        s = store.create("client-1")
        assert s.session_id is not None
        assert len(s.session_id) == 16

    def test_session_id_is_hex(self):
        store = make_store()
        s = store.create("client-1")
        int(s.session_id, 16)  # should not raise

    def test_client_id_stored(self):
        store = make_store()
        s = store.create("myClient")
        assert s.client_id == "myClient"

    def test_created_at_is_recent(self):
        store = make_store()
        before = time.monotonic()
        s = store.create("c")
        after = time.monotonic()
        assert before <= s.created_at <= after

    def test_expires_at_uses_default_ttl(self):
        store = make_store(default_ttl_s=600.0)
        s = store.create("c")
        assert s.expires_at is not None
        assert abs((s.expires_at - s.created_at) - 600.0) < 1.0

    def test_custom_ttl_applied(self):
        store = make_store(default_ttl_s=3600.0)
        s = store.create("c", ttl_s=120.0)
        assert abs((s.expires_at - s.created_at) - 120.0) < 1.0

    def test_data_defaults_empty(self):
        store = make_store()
        s = store.create("c")
        assert s.data == {}

    def test_data_passed_in(self):
        store = make_store()
        s = store.create("c", data={"key": "val"})
        assert s.data == {"key": "val"}

    def test_unique_session_ids(self):
        store = make_store()
        ids = {store.create("c").session_id for _ in range(50)}
        assert len(ids) == 50

    def test_max_sessions_raises(self):
        store = make_store(max_sessions=2)
        store.create("c")
        store.create("c")
        with pytest.raises(ValueError):
            store.create("c")

    def test_max_sessions_error_message(self):
        store = make_store(max_sessions=1)
        store.create("c")
        with pytest.raises(ValueError, match="capacity"):
            store.create("c")


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_existing_session(self):
        store = make_store()
        s = store.create("c")
        retrieved = store.get(s.session_id)
        assert retrieved is s

    def test_get_missing_returns_none(self):
        store = make_store()
        assert store.get("nonexistent") is None

    def test_get_expired_returns_none(self):
        store = make_store(default_ttl_s=1000.0)
        s = store.create("c")
        # Force expiry by passing a future "now"
        future = s.expires_at + 1.0
        assert store.get(s.session_id, now=future) is None

    def test_get_not_yet_expired(self):
        store = make_store(default_ttl_s=1000.0)
        s = store.create("c")
        past = s.expires_at - 1.0
        assert store.get(s.session_id, now=past) is s


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_merges_data(self):
        store = make_store()
        s = store.create("c", data={"a": 1})
        store.update(s.session_id, {"b": 2})
        assert s.data == {"a": 1, "b": 2}

    def test_update_overwrites_key(self):
        store = make_store()
        s = store.create("c", data={"a": 1})
        store.update(s.session_id, {"a": 99})
        assert s.data["a"] == 99

    def test_update_returns_true_on_success(self):
        store = make_store()
        s = store.create("c")
        assert store.update(s.session_id, {"x": 1}) is True

    def test_update_missing_returns_false(self):
        store = make_store()
        assert store.update("nope", {"x": 1}) is False

    def test_update_expired_returns_false(self):
        store = make_store(default_ttl_s=1000.0)
        s = store.create("c")
        future = s.expires_at + 1.0
        assert store.update(s.session_id, {"x": 1}, now=future) is False


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_existing_returns_true(self):
        store = make_store()
        s = store.create("c")
        assert store.delete(s.session_id) is True

    def test_delete_removes_session(self):
        store = make_store()
        s = store.create("c")
        store.delete(s.session_id)
        assert store.get(s.session_id) is None

    def test_delete_missing_returns_false(self):
        store = make_store()
        assert store.delete("no-such-id") is False

    def test_delete_allows_new_create(self):
        store = make_store(max_sessions=1)
        s = store.create("c")
        store.delete(s.session_id)
        # Now capacity is freed
        s2 = store.create("c2")
        assert s2 is not None


# ---------------------------------------------------------------------------
# expire_all
# ---------------------------------------------------------------------------


class TestExpireAll:
    def test_expire_all_returns_count(self):
        store = make_store(default_ttl_s=1000.0)
        s1 = store.create("c")
        store.create("c")
        store.create("c")
        future = s1.expires_at + 1.0
        count = store.expire_all(now=future)
        assert count == 3

    def test_expire_all_leaves_non_expired(self):
        store = make_store(default_ttl_s=1000.0)
        s = store.create("c")
        # Use a "now" before expiry
        past = s.expires_at - 1.0
        count = store.expire_all(now=past)
        assert count == 0

    def test_expire_all_partial(self):
        store = make_store()
        s1 = store.create("c", ttl_s=1.0)
        store.create("c", ttl_s=9999.0)
        future = s1.expires_at + 0.1
        count = store.expire_all(now=future)
        assert count == 1


# ---------------------------------------------------------------------------
# active_count
# ---------------------------------------------------------------------------


class TestActiveCount:
    def test_active_count_after_creates(self):
        store = make_store()
        store.create("c")
        store.create("c")
        assert store.active_count() == 2

    def test_active_count_excludes_expired(self):
        store = make_store(default_ttl_s=1000.0)
        s = store.create("c")
        store.create("c")
        future = s.expires_at + 1.0
        assert store.active_count(now=future) == 0

    def test_active_count_zero_on_empty(self):
        store = make_store()
        assert store.active_count() == 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in SESSION_STORE_REGISTRY

    def test_default_is_session_store(self):
        assert SESSION_STORE_REGISTRY["default"] is SessionStore

    def test_default_instantiable(self):
        cls = SESSION_STORE_REGISTRY["default"]
        store = cls()
        assert isinstance(store, SessionStore)
