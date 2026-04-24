"""Tests for src/serving/model_router.py"""

import pytest
from src.serving.model_router import (
    ModelEndpoint,
    ModelRouter,
    RoutingPolicy,
    SERVING_REGISTRY,
)


def _ep(eid: str, port: int = 8000, **kwargs) -> ModelEndpoint:
    return ModelEndpoint(
        endpoint_id=eid,
        model_name="test-model",
        host="localhost",
        port=port,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_and_list():
    r = ModelRouter()
    r.register(_ep("a"))
    assert len(r.endpoints) == 1
    assert r.endpoints[0].endpoint_id == "a"


def test_deregister():
    r = ModelRouter()
    r.register(_ep("a"))
    r.deregister("a")
    assert r.endpoints == []


def test_deregister_missing_is_noop():
    r = ModelRouter()
    r.deregister("does-not-exist")  # should not raise


def test_register_overwrites():
    r = ModelRouter()
    r.register(_ep("a", port=8000))
    r.register(_ep("a", port=9000))
    assert len(r.endpoints) == 1
    assert r.endpoints[0].port == 9000


# ---------------------------------------------------------------------------
# select — no endpoints
# ---------------------------------------------------------------------------


def test_select_empty_returns_none():
    r = ModelRouter()
    assert r.select("key") is None


# ---------------------------------------------------------------------------
# ROUND_ROBIN
# ---------------------------------------------------------------------------


def test_round_robin_cycles():
    r = ModelRouter()
    r.register(_ep("a"))
    r.register(_ep("b"))
    r.register(_ep("c"))
    ids = [r.select("x", RoutingPolicy.ROUND_ROBIN).endpoint_id for _ in range(6)]
    # Must cycle through all three
    assert set(ids[:3]) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# LEAST_LOADED
# ---------------------------------------------------------------------------


def test_least_loaded_picks_minimum():
    r = ModelRouter()
    r.register(_ep("heavy", active_requests=10))
    r.register(_ep("light", active_requests=1))
    r.register(_ep("medium", active_requests=5))
    assert r.select("k", RoutingPolicy.LEAST_LOADED).endpoint_id == "light"


# ---------------------------------------------------------------------------
# HASH_CONSISTENT
# ---------------------------------------------------------------------------


def test_hash_consistent_deterministic():
    r = ModelRouter()
    r.register(_ep("a"))
    r.register(_ep("b"))
    r.register(_ep("c"))
    first = r.select("stable-key", RoutingPolicy.HASH_CONSISTENT).endpoint_id
    second = r.select("stable-key", RoutingPolicy.HASH_CONSISTENT).endpoint_id
    assert first == second


# ---------------------------------------------------------------------------
# LATENCY_AWARE
# ---------------------------------------------------------------------------


def test_latency_aware_picks_best_score():
    r = ModelRouter()
    # weight/latency score: a=1/10=0.1, b=1/5=0.2 → b wins
    r.register(_ep("a", latency_ms=10.0))
    r.register(_ep("b", latency_ms=5.0))
    assert r.select("k", RoutingPolicy.LATENCY_AWARE).endpoint_id == "b"


def test_latency_aware_skips_zero_latency():
    r = ModelRouter()
    r.register(_ep("zero", latency_ms=0.0))
    r.register(_ep("good", latency_ms=20.0))
    assert r.select("k", RoutingPolicy.LATENCY_AWARE).endpoint_id == "good"


def test_latency_aware_fallback_when_all_zero():
    r = ModelRouter()
    r.register(_ep("a", latency_ms=0.0))
    # Should not raise; falls back to round-robin
    ep = r.select("k", RoutingPolicy.LATENCY_AWARE)
    assert ep is not None


# ---------------------------------------------------------------------------
# Load tracking
# ---------------------------------------------------------------------------


def test_increment_and_decrement_load():
    r = ModelRouter()
    r.register(_ep("a"))
    r.increment_load("a")
    r.increment_load("a")
    assert r.endpoints[0].active_requests == 2
    r.decrement_load("a")
    assert r.endpoints[0].active_requests == 1


def test_decrement_does_not_go_below_zero():
    r = ModelRouter()
    r.register(_ep("a"))
    r.decrement_load("a")
    assert r.endpoints[0].active_requests == 0


def test_load_tracking_missing_id_is_noop():
    r = ModelRouter()
    r.increment_load("ghost")
    r.decrement_load("ghost")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_serving_registry_contains_model_router():
    assert "model_router" in SERVING_REGISTRY
    assert isinstance(SERVING_REGISTRY["model_router"], ModelRouter)
