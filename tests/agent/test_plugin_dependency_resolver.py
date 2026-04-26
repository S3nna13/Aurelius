"""Tests for src.agent.plugin_dependency_resolver — topological dependency ordering."""

from __future__ import annotations

import pytest

from src.agent.plugin_dependency_resolver import (
    DEFAULT_DEPENDENCY_RESOLVER,
    DEPENDENCY_RESOLVER_REGISTRY,
    DependencyCycleError,
    DependencyResolver,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh() -> DependencyResolver:
    """Return a brand-new resolver to avoid test pollution."""
    return DependencyResolver()


# ---------------------------------------------------------------------------
# 1. test_simple_chain
# ---------------------------------------------------------------------------


def test_simple_chain():
    r = _fresh()
    r.register("B", ["C"])
    r.register("A", ["B"])
    assert r.resolve("A") == ["C", "B", "A"]


# ---------------------------------------------------------------------------
# 2. test_diamond
# ---------------------------------------------------------------------------


def test_diamond():
    r = _fresh()
    r.register("D", [])
    r.register("B", ["D"])
    r.register("C", ["D"])
    r.register("A", ["B", "C"])
    result = r.resolve("A")
    assert result[0] == "D"
    assert result[-1] == "A"
    assert set(result[1:3]) == {"B", "C"}


# ---------------------------------------------------------------------------
# 3. test_self_dependency_raises
# ---------------------------------------------------------------------------


def test_self_dependency_raises():
    r = _fresh()
    with pytest.raises(DependencyCycleError, match="cannot depend on itself"):
        r.register("A", ["A"])


# ---------------------------------------------------------------------------
# 4. test_simple_cycle
# ---------------------------------------------------------------------------


def test_simple_cycle():
    r = _fresh()
    r.register("A", ["B"])
    r.register("B", ["A"])
    with pytest.raises(DependencyCycleError, match="Circular dependency"):
        r.resolve("A")


# ---------------------------------------------------------------------------
# 5. test_longer_cycle
# ---------------------------------------------------------------------------


def test_longer_cycle():
    r = _fresh()
    r.register("A", ["B"])
    r.register("B", ["C"])
    r.register("C", ["A"])
    with pytest.raises(DependencyCycleError, match="Circular dependency"):
        r.resolve("A")


# ---------------------------------------------------------------------------
# 6. test_missing_dependency_resolves
# ---------------------------------------------------------------------------


def test_missing_dependency_resolves():
    r = _fresh()
    r.register("A", ["B"])
    # B is not registered
    assert r.resolve("A") == ["B", "A"]


# ---------------------------------------------------------------------------
# 7. test_batch_resolution
# ---------------------------------------------------------------------------


def test_batch_resolution():
    r = _fresh()
    r.register("C", [])
    r.register("B", ["C"])
    r.register("A", ["B"])
    result = r.resolve_batch(["A", "C"])
    assert result.index("C") < result.index("B") < result.index("A")
    assert "C" in result


# ---------------------------------------------------------------------------
# 8. test_cache_hit
# ---------------------------------------------------------------------------


def test_cache_hit():
    r = _fresh()
    r.register("B", ["C"])
    r.register("A", ["B"])
    first = r.resolve("A")
    second = r.resolve("A")
    assert first is second  # same list object from cache


# ---------------------------------------------------------------------------
# 9. test_cache_invalidation_on_register
# ---------------------------------------------------------------------------


def test_cache_invalidation_on_register():
    r = _fresh()
    r.register("B", ["C"])
    r.register("A", ["B"])
    first = r.resolve("A")
    # Re-register B with new dependency D
    r.register("B", ["D"])
    second = r.resolve("A")
    assert "D" in second
    assert second != first


# ---------------------------------------------------------------------------
# 10. test_has_dependency_true
# ---------------------------------------------------------------------------


def test_has_dependency_true():
    r = _fresh()
    r.register("A", ["B"])
    assert r.has_dependency("A", "B") is True


# ---------------------------------------------------------------------------
# 11. test_has_dependency_false
# ---------------------------------------------------------------------------


def test_has_dependency_false():
    r = _fresh()
    r.register("A", ["B"])
    assert r.has_dependency("A", "C") is False


# ---------------------------------------------------------------------------
# 12. test_get_direct_dependencies
# ---------------------------------------------------------------------------


def test_get_direct_dependencies():
    r = _fresh()
    r.register("A", ["B", "C"])
    assert r.get_direct_dependencies("A") == ["B", "C"]


# ---------------------------------------------------------------------------
# 13. test_clear_resets_state
# ---------------------------------------------------------------------------


def test_clear_resets_state():
    r = _fresh()
    r.register("A", ["B"])
    r.resolve("A")
    r.clear()
    assert r.get_direct_dependencies("A") == []
    assert r._resolved == {}


# ---------------------------------------------------------------------------
# 14. test_empty_deps_returns_plugin_id
# ---------------------------------------------------------------------------


def test_empty_deps_returns_plugin_id():
    r = _fresh()
    r.register("A", [])
    assert r.resolve("A") == ["A"]


# ---------------------------------------------------------------------------
# 15. test_registry_singleton_exists
# ---------------------------------------------------------------------------


def test_registry_singleton_exists():
    assert isinstance(DEFAULT_DEPENDENCY_RESOLVER, DependencyResolver)
    assert isinstance(DEPENDENCY_RESOLVER_REGISTRY, dict)
    assert "default" in DEPENDENCY_RESOLVER_REGISTRY
    assert DEPENDENCY_RESOLVER_REGISTRY["default"] is DEFAULT_DEPENDENCY_RESOLVER


# ---------------------------------------------------------------------------
# 16. test_register_empty_plugin_id_raises
# ---------------------------------------------------------------------------


def test_register_empty_plugin_id_raises():
    r = _fresh()
    with pytest.raises(ValueError, match="non-empty"):
        r.register("", ["B"])


# ---------------------------------------------------------------------------
# 17. test_register_empty_dependency_raises
# ---------------------------------------------------------------------------


def test_register_empty_dependency_raises():
    r = _fresh()
    with pytest.raises(ValueError, match="non-empty"):
        r.register("A", [""])


# ---------------------------------------------------------------------------
# 18. test_batch_cycle_detection
# ---------------------------------------------------------------------------


def test_batch_cycle_detection():
    r = _fresh()
    r.register("A", ["B"])
    r.register("B", ["A"])
    with pytest.raises(DependencyCycleError, match="Circular dependency"):
        r.resolve_batch(["A"])


# ---------------------------------------------------------------------------
# 19. test_get_direct_dependencies_unregistered_empty
# ---------------------------------------------------------------------------


def test_get_direct_dependencies_unregistered_empty():
    r = _fresh()
    assert r.get_direct_dependencies("ghost") == []


# ---------------------------------------------------------------------------
# 20. test_has_dependency_unregistered_false
# ---------------------------------------------------------------------------


def test_has_dependency_unregistered_false():
    r = _fresh()
    assert r.has_dependency("ghost", "other") is False


# ---------------------------------------------------------------------------
# 21. test_resolve_batch_shared_dependencies
# ---------------------------------------------------------------------------


def test_resolve_batch_shared_dependencies():
    r = _fresh()
    r.register("D", [])
    r.register("B", ["D"])
    r.register("C", ["D"])
    r.register("A", ["B", "C"])
    result = r.resolve_batch(["A", "B"])
    assert result.index("D") < result.index("B")
    assert result.index("D") < result.index("C")
    assert result.index("B") < result.index("A")


# ---------------------------------------------------------------------------
# 22. test_resolve_batch_empty_list
# ---------------------------------------------------------------------------


def test_resolve_batch_empty_list():
    r = _fresh()
    assert r.resolve_batch([]) == []
