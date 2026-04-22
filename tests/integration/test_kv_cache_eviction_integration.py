"""Integration tests for the KV cache eviction primitives.

Verifies additive integration into ``src.inference`` (no existing
decoder exports were disturbed) and end-to-end scenarios on the two
most safety-relevant policies: LRU and SINK_PRESERVING.
"""

from __future__ import annotations

import pytest


def test_public_surface_re_exported_from_src_inference():
    from src.inference import (
        EVICTION_POLICY_REGISTRY,
        CacheEntry,
        EvictionDecision,
        EvictionEngine,
        EvictionError,
        EvictionPolicy,
        select_victims,
    )

    assert EvictionPolicy.LRU.value == "lru"
    assert issubclass(EvictionError, Exception)
    assert EvictionDecision.__dataclass_fields__
    assert CacheEntry(key="x", size_bytes=1).key == "x"
    assert callable(select_victims)
    assert EvictionEngine  # sanity
    assert EVICTION_POLICY_REGISTRY  # sanity


def test_end_to_end_lru_scenario():
    from src.inference import CacheEntry, EvictionEngine, EvictionPolicy

    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=30)
    engine.admit(CacheEntry("a", 10))
    engine.admit(CacheEntry("b", 10))
    engine.admit(CacheEntry("c", 10))
    engine.touch("a"); engine.touch("c")  # b becomes LRU
    decision = engine.admit(CacheEntry("d", 10))
    assert decision.evicted == ("b",)
    assert decision.policy is EvictionPolicy.LRU
    assert engine.total_bytes() == 30
    assert set(engine.keys()) == {"a", "c", "d"}


def test_end_to_end_sink_preserving_scenario():
    from src.inference import CacheEntry, EvictionEngine, EvictionPolicy

    engine = EvictionEngine(
        EvictionPolicy.SINK_PRESERVING, capacity_bytes=40, sink_count=2,
    )
    engine.admit(CacheEntry("SINK_A", 10))
    engine.admit(CacheEntry("SINK_B", 10))
    engine.admit(CacheEntry("body_1", 10))
    engine.admit(CacheEntry("body_2", 10))
    engine.touch("body_2")  # body_1 is the LRU non-sink
    decision = engine.admit(CacheEntry("body_3", 10))
    assert "SINK_A" in engine and "SINK_B" in engine
    assert decision.evicted == ("body_1",)
    assert set(engine.keys()) == {"SINK_A", "SINK_B", "body_2", "body_3"}


def test_eviction_policy_registry_maps_all_five():
    from src.inference import EVICTION_POLICY_REGISTRY, EvictionEngine, EvictionPolicy

    assert set(EVICTION_POLICY_REGISTRY.keys()) == {p.value for p in EvictionPolicy}
    assert len(EVICTION_POLICY_REGISTRY) == 5
    for cls in EVICTION_POLICY_REGISTRY.values():
        assert cls is EvictionEngine


def test_decoder_registry_additive_integration_untouched():
    from src.inference import DECODER_REGISTRY

    assert isinstance(DECODER_REGISTRY, dict)
    assert len(DECODER_REGISTRY) > 0
    # Spot-check a couple of well-known decoders shipped by Aurelius.
    for expected in ("reasoning_level", "coconut", "eagle3"):
        assert expected in DECODER_REGISTRY


def test_transactional_admission_when_over_capacity():
    """Failing admit() leaves engine state completely intact."""

    from src.inference import CacheEntry, EvictionEngine, EvictionError, EvictionPolicy

    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=20)
    engine.admit(CacheEntry("a", 10, is_sink=True))
    engine.admit(CacheEntry("b", 10, is_sink=True))
    snap_before = engine.snapshot()

    with pytest.raises(EvictionError):
        engine.admit(CacheEntry("c", 10))  # nothing evictable

    snap_after = engine.snapshot()
    assert snap_before == snap_after
