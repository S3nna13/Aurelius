"""Unit tests for :mod:`src.inference.kv_cache_eviction`."""

from __future__ import annotations

import json

import pytest

from src.inference.kv_cache_eviction import (
    EVICTION_POLICY_REGISTRY,
    CacheEntry,
    EvictionDecision,
    EvictionEngine,
    EvictionError,
    EvictionPolicy,
    select_victims,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class FakeClock:
    """Deterministic, strictly-increasing ns clock for LRU/LFU assertions."""

    def __init__(self, start: int = 1_000) -> None:
        self.value = start

    def __call__(self) -> int:
        v = self.value
        self.value += 1
        return v


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

def test_construct_valid_defaults(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=1024, clock_ns=clock)
    assert engine.policy is EvictionPolicy.LRU
    assert engine.capacity_bytes == 1024
    assert engine.sink_count == 0
    assert len(engine) == 0
    assert engine.total_bytes() == 0
    assert engine.keys() == ()


def test_construct_rejects_zero_capacity():
    with pytest.raises(EvictionError, match="invalid_capacity"):
        EvictionEngine(EvictionPolicy.LRU, capacity_bytes=0)


def test_construct_rejects_negative_capacity():
    with pytest.raises(EvictionError, match="invalid_capacity"):
        EvictionEngine(EvictionPolicy.LRU, capacity_bytes=-1)


def test_construct_rejects_negative_sink_count():
    with pytest.raises(EvictionError, match="invalid_sink_count"):
        EvictionEngine(EvictionPolicy.LRU, capacity_bytes=100, sink_count=-1)


def test_construct_rejects_non_enum_policy():
    with pytest.raises(EvictionError, match="invalid_policy"):
        EvictionEngine("lru", capacity_bytes=100)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# admit — validation
# ---------------------------------------------------------------------------

def test_admit_rejects_negative_size(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=100, clock_ns=clock)
    with pytest.raises(EvictionError, match="size_bytes"):
        engine.admit(CacheEntry(key="a", size_bytes=-1))


def test_admit_rejects_empty_key(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=100, clock_ns=clock)
    with pytest.raises(EvictionError, match="non-empty"):
        engine.admit(CacheEntry(key="", size_bytes=10))


def test_admit_rejects_non_cache_entry(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=100, clock_ns=clock)
    with pytest.raises(EvictionError, match="invalid_entry"):
        engine.admit("not_an_entry")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# admit — basic behavior
# ---------------------------------------------------------------------------

def test_admit_empty_cache_returns_empty_decision(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=100, clock_ns=clock)
    decision = engine.admit(CacheEntry(key="a", size_bytes=40))
    assert isinstance(decision, EvictionDecision)
    assert decision.evicted == ()
    assert decision.freed_bytes == 0
    assert decision.policy is EvictionPolicy.LRU
    assert decision.reason == "admit"
    assert "a" in engine
    assert engine.total_bytes() == 40


def test_admit_up_to_capacity_no_evict(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=100, clock_ns=clock)
    engine.admit(CacheEntry("a", 40))
    engine.admit(CacheEntry("b", 40))
    decision = engine.admit(CacheEntry("c", 20))
    assert decision.evicted == ()
    assert engine.total_bytes() == 100
    assert set(engine.keys()) == {"a", "b", "c"}


def test_re_admit_refreshes_existing(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=100, clock_ns=clock)
    engine.admit(CacheEntry("a", 30))
    decision = engine.admit(CacheEntry("a", 50, weight=2.0))
    assert decision.reason == "refresh"
    assert decision.evicted == ()
    assert engine.total_bytes() == 50


def test_admit_zero_byte_entry_allowed(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=100, clock_ns=clock)
    decision = engine.admit(CacheEntry("a", 0))
    assert decision.reason == "admit"
    assert "a" in engine
    assert engine.total_bytes() == 0


def test_admit_larger_than_capacity_raises(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=100, clock_ns=clock)
    with pytest.raises(EvictionError, match="insufficient_capacity"):
        engine.admit(CacheEntry("giant", 200))
    # Transactional: cache state unchanged.
    assert len(engine) == 0
    assert engine.total_bytes() == 0


# ---------------------------------------------------------------------------
# Policy-specific scenarios
# ---------------------------------------------------------------------------

def test_lru_evicts_least_recently_touched(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=30, clock_ns=clock)
    engine.admit(CacheEntry("A", 10))
    engine.admit(CacheEntry("B", 10))
    engine.admit(CacheEntry("C", 10))
    engine.touch("A")  # A becomes MRU; B is now LRU
    decision = engine.admit(CacheEntry("D", 10))
    assert decision.evicted == ("B",)
    assert "B" not in engine
    assert "A" in engine
    assert "C" in engine
    assert "D" in engine


def test_lfu_evicts_least_used(clock):
    engine = EvictionEngine(EvictionPolicy.LFU, capacity_bytes=30, clock_ns=clock)
    engine.admit(CacheEntry("A", 10))
    engine.admit(CacheEntry("B", 10))
    engine.admit(CacheEntry("C", 10))
    engine.touch("A"); engine.touch("A")
    engine.touch("B")
    decision = engine.admit(CacheEntry("D", 10))
    assert decision.evicted == ("C",)
    assert set(engine.keys()) == {"A", "B", "D"}


def test_fifo_evicts_oldest_insertion(clock):
    engine = EvictionEngine(EvictionPolicy.FIFO, capacity_bytes=30, clock_ns=clock)
    engine.admit(CacheEntry("A", 10))
    engine.admit(CacheEntry("B", 10))
    engine.admit(CacheEntry("C", 10))
    for _ in range(5):
        engine.touch("C")
    decision = engine.admit(CacheEntry("D", 10))
    assert decision.evicted == ("A",)


def test_weighted_evicts_lowest_density(clock):
    engine = EvictionEngine(EvictionPolicy.WEIGHTED, capacity_bytes=200, clock_ns=clock)
    engine.admit(CacheEntry("A", 100, weight=10.0))
    engine.admit(CacheEntry("B", 100, weight=1.0))
    decision = engine.admit(CacheEntry("C", 100, weight=5.0))
    assert decision.evicted == ("B",)
    assert "B" not in engine and "A" in engine and "C" in engine


def test_sink_preserving_pins_first_n_insertions(clock):
    engine = EvictionEngine(
        EvictionPolicy.SINK_PRESERVING, capacity_bytes=40, sink_count=2, clock_ns=clock,
    )
    engine.admit(CacheEntry("S1", 10))
    engine.admit(CacheEntry("S2", 10))
    engine.admit(CacheEntry("C", 10))
    engine.admit(CacheEntry("D", 10))
    # Touch sinks to make them the "most recent" — SINK_PRESERVING must
    # still refuse to evict them. Touch D to keep C as LRU candidate.
    engine.touch("D")
    decision = engine.admit(CacheEntry("E", 10))
    assert "S1" in engine and "S2" in engine
    assert decision.evicted == ("C",)


def test_sink_preserving_never_evicts_even_when_they_are_lru(clock):
    engine = EvictionEngine(
        EvictionPolicy.SINK_PRESERVING, capacity_bytes=20, sink_count=1, clock_ns=clock,
    )
    engine.admit(CacheEntry("SINK", 10))  # never touched again
    engine.admit(CacheEntry("X", 10))
    engine.touch("X")  # X is newer
    decision = engine.admit(CacheEntry("Y", 10))
    # SINK is the oldest/LRU but is pinned; X must go.
    assert decision.evicted == ("X",)
    assert "SINK" in engine


def test_sink_preserving_with_sink_count_zero_behaves_like_lru(clock):
    sp = EvictionEngine(
        EvictionPolicy.SINK_PRESERVING, capacity_bytes=30, sink_count=0, clock_ns=FakeClock(),
    )
    lru = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=30, clock_ns=FakeClock())
    for engine in (sp, lru):
        engine.admit(CacheEntry("A", 10))
        engine.admit(CacheEntry("B", 10))
        engine.admit(CacheEntry("C", 10))
        engine.touch("A")
        engine.admit(CacheEntry("D", 10))
    assert set(sp.keys()) == set(lru.keys())


def test_lru_honors_per_entry_is_sink_flag(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=30, clock_ns=clock)
    engine.admit(CacheEntry("A", 10, is_sink=True))  # pinned via flag
    engine.admit(CacheEntry("B", 10))
    engine.admit(CacheEntry("C", 10))
    engine.touch("B"); engine.touch("C")  # A is still "oldest"
    decision = engine.admit(CacheEntry("D", 10))
    assert "A" in engine
    assert decision.evicted == ("B",)


def test_re_admitting_sink_preserves_pin(clock):
    engine = EvictionEngine(
        EvictionPolicy.SINK_PRESERVING, capacity_bytes=50, sink_count=1, clock_ns=clock,
    )
    engine.admit(CacheEntry("SINK", 10))
    engine.admit(CacheEntry("X", 10))
    engine.admit(CacheEntry("Y", 10))
    engine.admit(CacheEntry("Z", 10))
    # Refresh the sink — size grows, but SINK stays pinned.
    refresh = engine.admit(CacheEntry("SINK", 20))
    assert refresh.reason == "refresh"
    assert "SINK" in engine
    # Touching survivors must not touch SINK off.
    engine.touch("X"); engine.touch("Y"); engine.touch("Z")
    decision = engine.admit(CacheEntry("W", 10))
    assert "SINK" in engine
    assert decision.evicted  # a non-sink had to go


# ---------------------------------------------------------------------------
# touch / evict
# ---------------------------------------------------------------------------

def test_touch_unknown_key_raises(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=100, clock_ns=clock)
    with pytest.raises(EvictionError, match="unknown_key"):
        engine.touch("ghost")


def test_evict_known_returns_entry(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=100, clock_ns=clock)
    engine.admit(CacheEntry("a", 40, weight=2.5))
    removed = engine.evict("a")
    assert removed.key == "a"
    assert removed.size_bytes == 40
    assert removed.weight == 2.5
    assert "a" not in engine
    assert engine.total_bytes() == 0


def test_evict_unknown_raises(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=100, clock_ns=clock)
    with pytest.raises(EvictionError, match="unknown_key"):
        engine.evict("nope")


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def test_total_bytes_equals_sum_of_sizes(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=1000, clock_ns=clock)
    engine.admit(CacheEntry("a", 100))
    engine.admit(CacheEntry("b", 250))
    engine.admit(CacheEntry("c", 50))
    assert engine.total_bytes() == 400


def test_contains_len_and_keys(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=100, clock_ns=clock)
    assert len(engine) == 0
    assert "a" not in engine
    engine.admit(CacheEntry("b", 10))
    engine.admit(CacheEntry("a", 10))
    engine.admit(CacheEntry("c", 10))
    assert len(engine) == 3
    assert "a" in engine and "b" in engine and "c" in engine
    assert engine.keys() == ("a", "b", "c")  # sorted stable order
    assert "missing" not in engine
    assert 123 not in engine  # non-string safely returns False


def test_snapshot_is_json_serializable(clock):
    engine = EvictionEngine(
        EvictionPolicy.WEIGHTED, capacity_bytes=100, sink_count=1, clock_ns=clock,
    )
    engine.admit(CacheEntry("a", 20, weight=3.0, is_sink=True))
    engine.admit(CacheEntry("b", 10, weight=1.5))
    engine.touch("b")
    snap = engine.snapshot()
    payload = json.dumps(snap)
    restored = json.loads(payload)
    assert restored["policy"] == "weighted"
    assert restored["capacity_bytes"] == 100
    assert restored["sink_count"] == 1
    assert restored["total_bytes"] == 30
    keys = {e["key"] for e in restored["entries"]}
    assert keys == {"a", "b"}


# ---------------------------------------------------------------------------
# select_victims (pure function)
# ---------------------------------------------------------------------------

def _build_ctx():
    """Fixture context: A, B, C inserted in that order."""

    entries = (
        CacheEntry("A", 10, weight=5.0),
        CacheEntry("B", 10, weight=1.0),
        CacheEntry("C", 10, weight=3.0),
    )
    uses = {"A": 3, "B": 1, "C": 2}
    last_touch = {"A": 30, "B": 10, "C": 20}
    order = ("A", "B", "C")
    return entries, uses, last_touch, order


def test_select_victims_noop_when_nothing_needed():
    entries, uses, last, order = _build_ctx()
    assert select_victims(EvictionPolicy.LRU, entries, uses, last, order, 0, 0) == ()
    assert select_victims(EvictionPolicy.LRU, entries, uses, last, order, 0, -5) == ()


def test_select_victims_lru():
    entries, uses, last, order = _build_ctx()
    chosen = select_victims(EvictionPolicy.LRU, entries, uses, last, order, 0, 10)
    assert chosen == ("B",)


def test_select_victims_lfu():
    entries, uses, last, order = _build_ctx()
    chosen = select_victims(EvictionPolicy.LFU, entries, uses, last, order, 0, 10)
    assert chosen == ("B",)


def test_select_victims_fifo():
    entries, uses, last, order = _build_ctx()
    chosen = select_victims(EvictionPolicy.FIFO, entries, uses, last, order, 0, 10)
    assert chosen == ("A",)


def test_select_victims_weighted_picks_lowest_density():
    entries, uses, last, order = _build_ctx()
    chosen = select_victims(EvictionPolicy.WEIGHTED, entries, uses, last, order, 0, 10)
    assert chosen == ("B",)  # B has lowest weight/size


def test_select_victims_sink_preserving_pins_first_n():
    entries, uses, last, order = _build_ctx()
    chosen = select_victims(
        EvictionPolicy.SINK_PRESERVING, entries, uses, last, order, sink_count=2,
        bytes_needed=10,
    )
    # Only C is evictable; A and B are pinned even though B has the
    # oldest last_touch.
    assert chosen == ("C",)


def test_select_victims_raises_when_infeasible():
    entries, uses, last, order = _build_ctx()
    with pytest.raises(EvictionError, match="insufficient_capacity"):
        select_victims(
            EvictionPolicy.SINK_PRESERVING, entries, uses, last, order,
            sink_count=3, bytes_needed=10,
        )


def test_select_victims_takes_multiple_keys_if_needed():
    entries, uses, last, order = _build_ctx()
    chosen = select_victims(EvictionPolicy.LRU, entries, uses, last, order, 0, 25)
    assert chosen == ("B", "C", "A")  # must take all three, LRU ascending


# ---------------------------------------------------------------------------
# Determinism / tie-breaks
# ---------------------------------------------------------------------------

def test_deterministic_clock_drives_lru_choice(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=20, clock_ns=clock)
    engine.admit(CacheEntry("A", 10))  # touch_ns baseline 1000
    engine.admit(CacheEntry("B", 10))  # touch_ns baseline 1001
    engine.touch("A")                   # A now newest
    decision = engine.admit(CacheEntry("C", 10))  # must evict B
    assert decision.evicted == ("B",)


def test_lru_tie_break_is_insertion_order():
    entries = (CacheEntry("A", 10), CacheEntry("B", 10), CacheEntry("C", 10))
    uses = {"A": 0, "B": 0, "C": 0}
    last_touch = {"A": 100, "B": 100, "C": 100}  # identical timestamps
    order = ("A", "B", "C")
    chosen = select_victims(EvictionPolicy.LRU, entries, uses, last_touch, order, 0, 10)
    assert chosen == ("A",)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_eviction_policy_registry_is_complete():
    assert set(EVICTION_POLICY_REGISTRY.keys()) == {p.value for p in EvictionPolicy}
    for cls in EVICTION_POLICY_REGISTRY.values():
        assert cls is EvictionEngine


# ---------------------------------------------------------------------------
# Large-scale sanity
# ---------------------------------------------------------------------------

def test_large_scale_stable_size(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=500, clock_ns=clock)
    for i in range(1000):
        engine.admit(CacheEntry(f"k{i:04d}", 10))
        assert engine.total_bytes() <= 500
    assert len(engine) == 50
    # The last 50 admissions should be resident.
    resident = set(engine.keys())
    assert resident == {f"k{i:04d}" for i in range(950, 1000)}


# ---------------------------------------------------------------------------
# Transactional property
# ---------------------------------------------------------------------------

def test_admit_is_transactional_on_failure(clock):
    engine = EvictionEngine(
        EvictionPolicy.SINK_PRESERVING, capacity_bytes=20, sink_count=2, clock_ns=clock,
    )
    engine.admit(CacheEntry("S1", 10))
    engine.admit(CacheEntry("S2", 10))
    pre_keys = engine.keys()
    pre_bytes = engine.total_bytes()
    # Only sinks resident; no evictable entries -> must raise.
    with pytest.raises(EvictionError, match="insufficient_capacity"):
        engine.admit(CacheEntry("X", 10))
    assert engine.keys() == pre_keys
    assert engine.total_bytes() == pre_bytes
    assert "X" not in engine


def test_refresh_with_eviction(clock):
    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=30, clock_ns=clock)
    engine.admit(CacheEntry("A", 10))
    engine.admit(CacheEntry("B", 10))
    engine.admit(CacheEntry("C", 10))
    engine.touch("A"); engine.touch("C")  # B is LRU
    decision = engine.admit(CacheEntry("A", 20))  # A grows; need 10 freed
    assert decision.reason == "refresh_with_eviction"
    assert decision.evicted == ("B",)
    assert engine.total_bytes() == 30
    assert set(engine.keys()) == {"A", "C"}
