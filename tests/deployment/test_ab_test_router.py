"""Tests for src/deployment/ab_test_router.py — ≥28 test cases."""

from __future__ import annotations

import dataclasses
import pytest

from src.deployment.ab_test_router import (
    AB_TEST_ROUTER_REGISTRY,
    ABTestRouter,
    Assignment,
    Variant,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_equal_router() -> ABTestRouter:
    return ABTestRouter([Variant("A", 1.0), Variant("B", 1.0)])


def _make_single_router() -> ABTestRouter:
    return ABTestRouter([Variant("only", 1.0)])


# ---------------------------------------------------------------------------
# Variant dataclass
# ---------------------------------------------------------------------------

class TestVariant:
    def test_name_and_weight(self):
        v = Variant("ctrl", 50.0)
        assert v.name == "ctrl"
        assert v.weight == 50.0

    def test_default_metadata_is_empty_dict(self):
        v = Variant("ctrl", 50.0)
        assert v.metadata == {}

    def test_metadata_not_shared_across_instances(self):
        v1 = Variant("a", 1.0)
        v2 = Variant("b", 1.0)
        v1.metadata["key"] = "val"
        assert "key" not in v2.metadata

    def test_custom_metadata(self):
        v = Variant("x", 1.0, metadata={"color": "blue"})
        assert v.metadata["color"] == "blue"


# ---------------------------------------------------------------------------
# Assignment frozen dataclass
# ---------------------------------------------------------------------------

class TestAssignment:
    def test_fields(self):
        a = Assignment("req-1", "A", 42)
        assert a.request_id == "req-1"
        assert a.variant_name == "A"
        assert a.bucket == 42

    def test_frozen_request_id(self):
        a = Assignment("req-1", "A", 42)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            a.request_id = "other"  # type: ignore[misc]

    def test_frozen_variant_name(self):
        a = Assignment("req-1", "A", 42)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            a.variant_name = "B"  # type: ignore[misc]

    def test_frozen_bucket(self):
        a = Assignment("req-1", "A", 42)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            a.bucket = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ABTestRouter construction
# ---------------------------------------------------------------------------

class TestABTestRouterInit:
    def test_empty_variants_raises(self):
        with pytest.raises(ValueError):
            ABTestRouter([])

    def test_all_zero_weights_raises(self):
        with pytest.raises(ValueError):
            ABTestRouter([Variant("A", 0.0), Variant("B", 0.0)])

    def test_weights_normalised_to_100(self):
        router = ABTestRouter([Variant("A", 1.0), Variant("B", 1.0)])
        total = sum(v.weight for v in router._variants)
        assert abs(total - 100.0) < 1e-9

    def test_single_variant_weight_becomes_100(self):
        router = _make_single_router()
        assert abs(router._variants[0].weight - 100.0) < 1e-9


# ---------------------------------------------------------------------------
# ABTestRouter.assign — determinism & bucket range
# ---------------------------------------------------------------------------

class TestABTestRouterAssign:
    def test_single_variant_always_assigned(self):
        router = _make_single_router()
        for i in range(50):
            a = router.assign(f"req-{i}")
            assert a.variant_name == "only"

    def test_deterministic_same_id_same_variant(self):
        router = _make_equal_router()
        first = router.assign("stable-id-xyz").variant_name
        for _ in range(10):
            assert router.assign("stable-id-xyz").variant_name == first

    def test_bucket_in_range_0_to_99(self):
        router = _make_equal_router()
        for i in range(200):
            a = router.assign(f"id-{i}")
            assert 0 <= a.bucket <= 99

    def test_assignment_request_id_preserved(self):
        router = _make_equal_router()
        a = router.assign("my-request")
        assert a.request_id == "my-request"

    def test_assignment_variant_is_known(self):
        router = _make_equal_router()
        known = {v.name for v in router._variants}
        for i in range(100):
            a = router.assign(f"r{i}")
            assert a.variant_name in known

    def test_two_equal_variants_roughly_50_50(self):
        """With 1 000 requests the split should be within 10% of 50/50."""
        router = _make_equal_router()
        counts: dict[str, int] = {"A": 0, "B": 0}
        for i in range(1000):
            a = router.assign(f"request-{i:06d}")
            counts[a.variant_name] += 1
        assert 400 <= counts["A"] <= 600, f"Unexpected split: {counts}"
        assert 400 <= counts["B"] <= 600, f"Unexpected split: {counts}"


# ---------------------------------------------------------------------------
# ABTestRouter.assignment_stats
# ---------------------------------------------------------------------------

class TestAssignmentStats:
    def test_totals_match_input_length(self):
        router = _make_equal_router()
        assignments = [router.assign(f"r{i}") for i in range(50)]
        stats = router.assignment_stats(assignments)
        assert stats["total"] == 50

    def test_by_variant_keys_match_variants(self):
        router = _make_equal_router()
        assignments = [router.assign(f"r{i}") for i in range(20)]
        stats = router.assignment_stats(assignments)
        assert set(stats["by_variant"].keys()) == {"A", "B"}

    def test_by_variant_counts_sum_to_total(self):
        router = _make_equal_router()
        assignments = [router.assign(f"r{i}") for i in range(100)]
        stats = router.assignment_stats(assignments)
        assert sum(stats["by_variant"].values()) == stats["total"]

    def test_empty_assignments(self):
        router = _make_equal_router()
        stats = router.assignment_stats([])
        assert stats["total"] == 0
        assert stats["by_variant"]["A"] == 0
        assert stats["by_variant"]["B"] == 0


# ---------------------------------------------------------------------------
# ABTestRouter.add_variant
# ---------------------------------------------------------------------------

class TestAddVariant:
    def test_add_variant_increases_count(self):
        router = _make_equal_router()
        router.add_variant(Variant("C", 1.0))
        assert len(router._variants) == 3

    def test_weights_renormalised_after_add(self):
        router = _make_equal_router()
        router.add_variant(Variant("C", 1.0))
        total = sum(v.weight for v in router._variants)
        assert abs(total - 100.0) < 1e-9

    def test_new_variant_is_assigned(self):
        router = _make_equal_router()
        router.add_variant(Variant("C", 1.0))
        known = {v.name for v in router._variants}
        for i in range(300):
            a = router.assign(f"req-{i}")
            assert a.variant_name in known


# ---------------------------------------------------------------------------
# ABTestRouter.remove_variant
# ---------------------------------------------------------------------------

class TestRemoveVariant:
    def test_remove_existing_returns_true(self):
        router = _make_equal_router()
        result = router.remove_variant("A")
        assert result is True

    def test_remove_nonexistent_returns_false(self):
        router = _make_equal_router()
        result = router.remove_variant("Z")
        assert result is False

    def test_remove_decreases_count(self):
        router = _make_equal_router()
        router.remove_variant("A")
        assert len(router._variants) == 1

    def test_remove_renormalises_weights(self):
        router = ABTestRouter([Variant("A", 1.0), Variant("B", 2.0), Variant("C", 1.0)])
        router.remove_variant("C")
        total = sum(v.weight for v in router._variants)
        assert abs(total - 100.0) < 1e-9

    def test_removed_variant_never_assigned(self):
        router = _make_equal_router()
        router.remove_variant("A")
        for i in range(100):
            a = router.assign(f"r{i}")
            assert a.variant_name != "A"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default_key(self):
        assert "default" in AB_TEST_ROUTER_REGISTRY

    def test_registry_default_is_ab_test_router(self):
        assert AB_TEST_ROUTER_REGISTRY["default"] is ABTestRouter

    def test_registry_default_is_instantiable(self):
        cls = AB_TEST_ROUTER_REGISTRY["default"]
        instance = cls([Variant("v1", 1.0)])
        assert isinstance(instance, ABTestRouter)
