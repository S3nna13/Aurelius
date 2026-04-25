"""Tests for src/runtime/hot_reload.py — 28+ tests."""

from __future__ import annotations

import dataclasses
import time

import pytest

from src.runtime.hot_reload import (
    HOT_RELOAD_REGISTRY,
    HotReloader,
    ReloadConfig,
    ReloadEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_loader(path: str) -> dict:
    return {"loaded_from": path, "value": 42}


def _bad_loader(path: str) -> None:
    raise RuntimeError(f"Cannot load '{path}'")


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default_key(self):
        assert "default" in HOT_RELOAD_REGISTRY

    def test_registry_default_is_class(self):
        assert HOT_RELOAD_REGISTRY["default"] is HotReloader


# ---------------------------------------------------------------------------
# ReloadEvent frozen dataclass
# ---------------------------------------------------------------------------

class TestReloadEvent:
    def test_event_is_frozen(self):
        evt = ReloadEvent(
            resource_type="cfg",
            path="/tmp/f",
            triggered_at=1.0,
            success=True,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            evt.success = False  # type: ignore[misc]

    def test_event_auto_event_id(self):
        evt = ReloadEvent(resource_type="x", path="p", triggered_at=0.0, success=True)
        assert isinstance(evt.event_id, str)
        assert len(evt.event_id) == 8

    def test_event_id_is_unique(self):
        ids = {
            ReloadEvent(resource_type="x", path="p", triggered_at=0.0, success=True).event_id
            for _ in range(20)
        }
        assert len(ids) == 20

    def test_event_default_error_empty(self):
        evt = ReloadEvent(resource_type="x", path="p", triggered_at=0.0, success=True)
        assert evt.error == ""

    def test_event_stores_all_fields(self):
        evt = ReloadEvent(
            resource_type="weights",
            path="/models/w.bin",
            triggered_at=5.5,
            success=False,
            error="disk error",
        )
        assert evt.resource_type == "weights"
        assert evt.path == "/models/w.bin"
        assert evt.triggered_at == 5.5
        assert evt.success is False
        assert evt.error == "disk error"


# ---------------------------------------------------------------------------
# ReloadConfig frozen dataclass
# ---------------------------------------------------------------------------

class TestReloadConfig:
    def test_config_is_frozen(self):
        cfg = ReloadConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.max_retries = 99  # type: ignore[misc]

    def test_config_defaults(self):
        cfg = ReloadConfig()
        assert cfg.watch_interval_s == 5.0
        assert cfg.max_retries == 3
        assert cfg.rollback_on_failure is True

    def test_config_custom_values(self):
        cfg = ReloadConfig(watch_interval_s=1.0, max_retries=1, rollback_on_failure=False)
        assert cfg.watch_interval_s == 1.0
        assert cfg.max_retries == 1
        assert cfg.rollback_on_failure is False


# ---------------------------------------------------------------------------
# register_resource
# ---------------------------------------------------------------------------

class TestRegisterResource:
    def test_register_increments_resource_count(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        assert r.resource_count() == 1

    def test_register_multiple(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        r.register_resource("weights", "/tmp/w.bin", _good_loader)
        assert r.resource_count() == 2

    def test_register_overwrites_same_type(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/v1.json", _good_loader)
        r.register_resource("cfg", "/tmp/v2.json", _good_loader)
        assert r.resource_count() == 1


# ---------------------------------------------------------------------------
# reload — success path
# ---------------------------------------------------------------------------

class TestReloadSuccess:
    def test_reload_returns_reload_event(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        evt = r.reload("cfg")
        assert isinstance(evt, ReloadEvent)

    def test_reload_success_true(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        evt = r.reload("cfg")
        assert evt.success is True

    def test_reload_stores_in_cache(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        r.reload("cfg")
        assert r.get("cfg") == {"loaded_from": "/tmp/cfg.json", "value": 42}

    def test_reload_event_resource_type(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        evt = r.reload("cfg")
        assert evt.resource_type == "cfg"

    def test_reload_event_path(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        evt = r.reload("cfg")
        assert evt.path == "/tmp/cfg.json"

    def test_reload_event_triggered_at_monotonic(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        before = time.monotonic()
        evt = r.reload("cfg")
        after = time.monotonic()
        assert before <= evt.triggered_at <= after

    def test_reload_error_empty_on_success(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        evt = r.reload("cfg")
        assert evt.error == ""


# ---------------------------------------------------------------------------
# reload — failure path
# ---------------------------------------------------------------------------

class TestReloadFailure:
    def test_reload_failure_success_false(self):
        r = HotReloader()
        r.register_resource("cfg", "/bad/path", _bad_loader)
        evt = r.reload("cfg")
        assert evt.success is False

    def test_reload_failure_records_error(self):
        r = HotReloader()
        r.register_resource("cfg", "/bad/path", _bad_loader)
        evt = r.reload("cfg")
        assert "bad/path" in evt.error or "Cannot load" in evt.error

    def test_rollback_on_failure_keeps_old_value(self):
        r = HotReloader(ReloadConfig(rollback_on_failure=True))
        old_value = {"version": 1}
        r.register_resource("cfg", "/good", lambda p: old_value)
        r.reload("cfg")
        # Now swap to a bad loader
        r.register_resource("cfg", "/bad", _bad_loader)
        r.reload("cfg")
        assert r.get("cfg") == old_value

    def test_no_rollback_clears_old_value_on_first_fail(self):
        """With rollback_on_failure=False and no prior value, cache stays None."""
        r = HotReloader(ReloadConfig(rollback_on_failure=False))
        r.register_resource("cfg", "/bad", _bad_loader)
        r.reload("cfg")
        assert r.get("cfg") is None

    def test_reload_unregistered_resource_returns_failure_event(self):
        r = HotReloader()
        evt = r.reload("unknown")
        assert evt.success is False
        assert evt.resource_type == "unknown"


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_none_before_reload(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        assert r.get("cfg") is None

    def test_get_after_reload(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        r.reload("cfg")
        assert r.get("cfg") is not None

    def test_get_unregistered_none(self):
        r = HotReloader()
        assert r.get("anything") is None


# ---------------------------------------------------------------------------
# reload_history
# ---------------------------------------------------------------------------

class TestReloadHistory:
    def test_history_empty_initially(self):
        r = HotReloader()
        assert r.reload_history() == []

    def test_history_grows_with_reloads(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        r.reload("cfg")
        r.reload("cfg")
        assert len(r.reload_history()) == 2

    def test_history_includes_failures(self):
        r = HotReloader()
        r.register_resource("cfg", "/bad", _bad_loader)
        r.reload("cfg")
        history = r.reload_history()
        assert len(history) == 1
        assert history[0].success is False

    def test_history_returns_copy(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        r.reload("cfg")
        h1 = r.reload_history()
        h1.clear()
        assert len(r.reload_history()) == 1

    def test_history_event_order(self):
        r = HotReloader()
        r.register_resource("cfg", "/tmp/cfg.json", _good_loader)
        r.register_resource("weights", "/bad", _bad_loader)
        r.reload("cfg")
        r.reload("weights")
        history = r.reload_history()
        assert history[0].resource_type == "cfg"
        assert history[1].resource_type == "weights"
