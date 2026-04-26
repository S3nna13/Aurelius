"""Tests for src/runtime/model_gc.py — 28+ tests."""

from __future__ import annotations

import time

from src.runtime.model_gc import (
    MODEL_GC_REGISTRY,
    GCPolicy,
    ModelGCManager,
    ModelHandle,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_handle(
    model_id: str = "m1",
    size_bytes: int = 1024,
    last_used: float | None = None,
    ref_count: int = 0,
) -> ModelHandle:
    return ModelHandle(
        model_id=model_id,
        size_bytes=size_bytes,
        last_used=last_used if last_used is not None else time.monotonic(),
        ref_count=ref_count,
    )


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_has_default_key(self):
        assert "default" in MODEL_GC_REGISTRY

    def test_registry_default_is_manager_class(self):
        assert MODEL_GC_REGISTRY["default"] is ModelGCManager


# ---------------------------------------------------------------------------
# ModelHandle dataclass
# ---------------------------------------------------------------------------


class TestModelHandle:
    def test_handle_stores_fields(self):
        h = ModelHandle(model_id="abc", size_bytes=512, last_used=1.0, ref_count=2)
        assert h.model_id == "abc"
        assert h.size_bytes == 512
        assert h.last_used == 1.0
        assert h.ref_count == 2

    def test_handle_default_ref_count(self):
        h = ModelHandle(model_id="x", size_bytes=100, last_used=0.0)
        assert h.ref_count == 0


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_adds_handle(self):
        mgr = ModelGCManager()
        h = make_handle("m1")
        mgr.register(h)
        assert len(mgr.status()) == 1

    def test_register_multiple(self):
        mgr = ModelGCManager()
        mgr.register(make_handle("a"))
        mgr.register(make_handle("b"))
        ids = {s["id"] for s in mgr.status()}
        assert ids == {"a", "b"}

    def test_register_overwrites_same_id(self):
        mgr = ModelGCManager()
        h1 = make_handle("m1", size_bytes=100)
        h2 = make_handle("m1", size_bytes=200)
        mgr.register(h1)
        mgr.register(h2)
        assert mgr.status()[0]["size_bytes"] == 200


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------


class TestAcquire:
    def test_acquire_returns_true_for_registered(self):
        mgr = ModelGCManager()
        mgr.register(make_handle("m1"))
        assert mgr.acquire("m1") is True

    def test_acquire_returns_false_for_unregistered(self):
        mgr = ModelGCManager()
        assert mgr.acquire("ghost") is False

    def test_acquire_increments_ref_count(self):
        mgr = ModelGCManager()
        mgr.register(make_handle("m1", ref_count=0))
        mgr.acquire("m1")
        status = {s["id"]: s for s in mgr.status()}
        assert status["m1"]["ref_count"] == 1

    def test_acquire_multiple_increments(self):
        mgr = ModelGCManager()
        mgr.register(make_handle("m1"))
        mgr.acquire("m1")
        mgr.acquire("m1")
        status = {s["id"]: s for s in mgr.status()}
        assert status["m1"]["ref_count"] == 2

    def test_acquire_updates_last_used(self):
        mgr = ModelGCManager()
        h = make_handle("m1", last_used=0.0)
        mgr.register(h)
        before = time.monotonic()
        mgr.acquire("m1")
        after = time.monotonic()
        status = {s["id"]: s for s in mgr.status()}
        assert before <= status["m1"]["last_used"] <= after


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------


class TestRelease:
    def test_release_returns_true_for_registered(self):
        mgr = ModelGCManager()
        mgr.register(make_handle("m1", ref_count=1))
        assert mgr.release("m1") is True

    def test_release_returns_false_for_unregistered(self):
        mgr = ModelGCManager()
        assert mgr.release("ghost") is False

    def test_release_decrements_ref_count(self):
        mgr = ModelGCManager()
        mgr.register(make_handle("m1", ref_count=3))
        mgr.release("m1")
        status = {s["id"]: s for s in mgr.status()}
        assert status["m1"]["ref_count"] == 2

    def test_release_floors_at_zero(self):
        mgr = ModelGCManager()
        mgr.register(make_handle("m1", ref_count=0))
        mgr.release("m1")
        status = {s["id"]: s for s in mgr.status()}
        assert status["m1"]["ref_count"] == 0


# ---------------------------------------------------------------------------
# evict_candidates
# ---------------------------------------------------------------------------


class TestEvictCandidates:
    def _manager_with_handles(self, policy: GCPolicy) -> ModelGCManager:
        mgr = ModelGCManager(policy=policy)
        mgr.register(ModelHandle("oldest", 100, last_used=1.0, ref_count=0))
        mgr.register(ModelHandle("middle", 100, last_used=2.0, ref_count=0))
        mgr.register(ModelHandle("newest", 100, last_used=3.0, ref_count=0))
        mgr.register(ModelHandle("held", 100, last_used=0.5, ref_count=1))
        return mgr

    def test_lru_sorted_oldest_first(self):
        mgr = self._manager_with_handles(GCPolicy.LRU)
        candidates = mgr.evict_candidates()
        assert candidates == ["oldest", "middle", "newest"]

    def test_lru_excludes_ref_count_nonzero(self):
        mgr = self._manager_with_handles(GCPolicy.LRU)
        assert "held" not in mgr.evict_candidates()

    def test_eager_returns_all_free(self):
        mgr = self._manager_with_handles(GCPolicy.EAGER)
        candidates = set(mgr.evict_candidates())
        assert candidates == {"oldest", "middle", "newest"}

    def test_eager_excludes_held(self):
        mgr = self._manager_with_handles(GCPolicy.EAGER)
        assert "held" not in mgr.evict_candidates()

    def test_deferred_returns_empty(self):
        mgr = self._manager_with_handles(GCPolicy.DEFERRED)
        assert mgr.evict_candidates() == []

    def test_none_returns_empty(self):
        mgr = self._manager_with_handles(GCPolicy.NONE)
        assert mgr.evict_candidates() == []

    def test_ref_count_nonzero_excluded_from_lru(self):
        mgr = ModelGCManager(policy=GCPolicy.LRU)
        mgr.register(ModelHandle("busy", 100, last_used=0.0, ref_count=5))
        assert mgr.evict_candidates() == []

    def test_lru_empty_when_all_held(self):
        mgr = ModelGCManager(policy=GCPolicy.LRU)
        mgr.register(ModelHandle("a", 100, last_used=1.0, ref_count=1))
        mgr.register(ModelHandle("b", 100, last_used=2.0, ref_count=2))
        assert mgr.evict_candidates() == []


# ---------------------------------------------------------------------------
# evict
# ---------------------------------------------------------------------------


class TestEvict:
    def test_evict_returns_true_if_existed(self):
        mgr = ModelGCManager()
        mgr.register(make_handle("m1"))
        assert mgr.evict("m1") is True

    def test_evict_removes_from_registry(self):
        mgr = ModelGCManager()
        mgr.register(make_handle("m1"))
        mgr.evict("m1")
        assert mgr.status() == []

    def test_evict_returns_false_if_not_existed(self):
        mgr = ModelGCManager()
        assert mgr.evict("ghost") is False


# ---------------------------------------------------------------------------
# total_memory
# ---------------------------------------------------------------------------


class TestTotalMemory:
    def test_total_memory_empty(self):
        assert ModelGCManager().total_memory() == 0

    def test_total_memory_sum(self):
        mgr = ModelGCManager()
        mgr.register(make_handle("a", size_bytes=100))
        mgr.register(make_handle("b", size_bytes=200))
        mgr.register(make_handle("c", size_bytes=300))
        assert mgr.total_memory() == 600

    def test_total_memory_after_evict(self):
        mgr = ModelGCManager()
        mgr.register(make_handle("a", size_bytes=400))
        mgr.register(make_handle("b", size_bytes=600))
        mgr.evict("a")
        assert mgr.total_memory() == 600


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_returns_list(self):
        mgr = ModelGCManager()
        assert isinstance(mgr.status(), list)

    def test_status_dict_keys(self):
        mgr = ModelGCManager()
        mgr.register(make_handle("m1", size_bytes=512, ref_count=1))
        entry = mgr.status()[0]
        assert set(entry.keys()) == {"id", "size_bytes", "last_used", "ref_count"}

    def test_status_values_correct(self):
        mgr = ModelGCManager()
        h = ModelHandle("m1", 1024, 99.0, 2)
        mgr.register(h)
        entry = mgr.status()[0]
        assert entry["id"] == "m1"
        assert entry["size_bytes"] == 1024
        assert entry["last_used"] == 99.0
        assert entry["ref_count"] == 2

    def test_status_empty_when_no_handles(self):
        assert ModelGCManager().status() == []
