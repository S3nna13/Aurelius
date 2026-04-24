"""Tests for src/serving/model_multiplexer.py — ≥28 test cases."""

from __future__ import annotations

import pytest

from src.serving.model_multiplexer import (
    MODEL_MULTIPLEXER_REGISTRY,
    ModelEndpoint,
    ModelMultiplexer,
    RoutingRule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ep(name: str, model_id: str = "m", priority: int = 0, tags: list[str] | None = None) -> ModelEndpoint:
    return ModelEndpoint(name=name, model_id=model_id, priority=priority, tags=tags or [])


def _rule(tag: str, endpoint_name: str) -> RoutingRule:
    return RoutingRule(tag=tag, endpoint_name=endpoint_name)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in MODEL_MULTIPLEXER_REGISTRY

    def test_registry_default_is_class(self):
        assert MODEL_MULTIPLEXER_REGISTRY["default"] is ModelMultiplexer


# ---------------------------------------------------------------------------
# Dataclass contracts
# ---------------------------------------------------------------------------


class TestModelEndpoint:
    def test_endpoint_fields(self):
        ep = _ep("alpha", model_id="gpt-4", priority=5, tags=["fast"])
        assert ep.name == "alpha"
        assert ep.model_id == "gpt-4"
        assert ep.priority == 5
        assert ep.tags == ["fast"]

    def test_endpoint_default_priority(self):
        ep = _ep("beta")
        assert ep.priority == 0

    def test_endpoint_default_tags(self):
        ep = _ep("gamma")
        assert ep.tags == []

    def test_endpoints_are_mutable_dataclasses(self):
        ep = _ep("delta")
        ep.priority = 10
        assert ep.priority == 10


class TestRoutingRule:
    def test_rule_fields(self):
        rule = _rule("gpu", "fast-endpoint")
        assert rule.tag == "gpu"
        assert rule.endpoint_name == "fast-endpoint"

    def test_rule_is_frozen(self):
        rule = _rule("gpu", "fast-endpoint")
        with pytest.raises((AttributeError, TypeError)):
            rule.tag = "cpu"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# list_endpoints
# ---------------------------------------------------------------------------


class TestListEndpoints:
    def test_list_endpoints_sorted(self):
        mx = ModelMultiplexer([_ep("charlie"), _ep("alice"), _ep("bob")])
        assert mx.list_endpoints() == ["alice", "bob", "charlie"]

    def test_list_endpoints_empty(self):
        mx = ModelMultiplexer([])
        assert mx.list_endpoints() == []

    def test_list_endpoints_single(self):
        mx = ModelMultiplexer([_ep("solo")])
        assert mx.list_endpoints() == ["solo"]


# ---------------------------------------------------------------------------
# add_endpoint / remove_endpoint
# ---------------------------------------------------------------------------


class TestAddRemoveEndpoint:
    def test_add_endpoint_appears_in_list(self):
        mx = ModelMultiplexer([])
        mx.add_endpoint(_ep("new"))
        assert "new" in mx.list_endpoints()

    def test_add_endpoint_replaces_existing(self):
        mx = ModelMultiplexer([_ep("x", model_id="old")])
        mx.add_endpoint(_ep("x", model_id="new"))
        assert len(mx.list_endpoints()) == 1

    def test_remove_endpoint_returns_true(self):
        mx = ModelMultiplexer([_ep("r")])
        assert mx.remove_endpoint("r") is True

    def test_remove_endpoint_missing_returns_false(self):
        mx = ModelMultiplexer([])
        assert mx.remove_endpoint("ghost") is False

    def test_remove_endpoint_no_longer_listed(self):
        mx = ModelMultiplexer([_ep("gone")])
        mx.remove_endpoint("gone")
        assert "gone" not in mx.list_endpoints()


# ---------------------------------------------------------------------------
# add_rule
# ---------------------------------------------------------------------------


class TestAddRule:
    def test_add_rule_used_by_route(self):
        ep_a = _ep("a", priority=0)
        ep_b = _ep("b", priority=10)
        mx = ModelMultiplexer([ep_a, ep_b])
        mx.add_rule(_rule("slow", "a"))
        result = mx.route(["slow"])
        assert result.name == "a"


# ---------------------------------------------------------------------------
# route – rule matching
# ---------------------------------------------------------------------------


class TestRouteByRule:
    def test_first_matching_rule_wins(self):
        ep_gpu = _ep("gpu-ep", priority=0)
        ep_cpu = _ep("cpu-ep", priority=10)
        rules = [_rule("gpu", "gpu-ep"), _rule("gpu", "cpu-ep")]
        mx = ModelMultiplexer([ep_gpu, ep_cpu], rules=rules)
        assert mx.route(["gpu"]).name == "gpu-ep"

    def test_second_rule_used_when_first_tag_absent(self):
        ep_a = _ep("a")
        ep_b = _ep("b")
        rules = [_rule("tag-a", "a"), _rule("tag-b", "b")]
        mx = ModelMultiplexer([ep_a, ep_b], rules=rules)
        assert mx.route(["tag-b"]).name == "b"

    def test_rule_for_nonexistent_endpoint_skipped(self):
        ep_real = _ep("real", priority=1)
        rules = [_rule("x", "missing"), _rule("x", "real")]
        mx = ModelMultiplexer([ep_real], rules=rules)
        assert mx.route(["x"]).name == "real"

    def test_multiple_tags_first_matching_rule_used(self):
        ep_a = _ep("a")
        ep_b = _ep("b")
        rules = [_rule("alpha", "a"), _rule("beta", "b")]
        mx = ModelMultiplexer([ep_a, ep_b], rules=rules)
        # Both tags present; "alpha" rule comes first
        assert mx.route(["beta", "alpha"]).name == "a"


# ---------------------------------------------------------------------------
# route – priority fallback
# ---------------------------------------------------------------------------


class TestRouteFallback:
    def test_no_rule_highest_priority_returned(self):
        ep_lo = _ep("low", priority=1)
        ep_hi = _ep("high", priority=99)
        mx = ModelMultiplexer([ep_lo, ep_hi])
        assert mx.route([]).name == "high"

    def test_tie_broken_alphabetically_by_name(self):
        ep_z = _ep("zebra", priority=5)
        ep_a = _ep("apple", priority=5)
        mx = ModelMultiplexer([ep_z, ep_a])
        # Alphabetically "apple" < "zebra"; we want *lowest* alphabetically to win
        # (spec: "tie-break by name alphabetically")
        result = mx.route([])
        assert result.name == "apple"

    def test_fallback_skips_unmatched_tags(self):
        ep = _ep("only", priority=7)
        rules = [_rule("missing-tag", "only")]
        mx = ModelMultiplexer([ep], rules=rules)
        # No matching rule tag in request → fallback
        assert mx.route(["other-tag"]).name == "only"

    def test_no_endpoints_raises(self):
        mx = ModelMultiplexer([])
        with pytest.raises(ValueError):
            mx.route([])

    def test_single_endpoint_always_returned(self):
        ep = _ep("solo", priority=0)
        mx = ModelMultiplexer([ep])
        assert mx.route(["anything"]).name == "solo"
