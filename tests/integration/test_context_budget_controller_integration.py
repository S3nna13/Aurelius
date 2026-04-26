"""Integration tests for ContextBudgetController registry wiring.

Verifies that:
- 'context_budget' key exists in LONGCONTEXT_STRATEGY_REGISTRY
- The class can be instantiated from the registry
- The full eviction scenario (spec integration test) works end-to-end
- Pre-existing registry entries are not disturbed (regression guards)
"""

from __future__ import annotations

from src.longcontext import LONGCONTEXT_STRATEGY_REGISTRY
from src.longcontext.context_budget_controller import (
    ContextBudgetConfig,
    ContextBudgetController,
    ContextSegment,
    SegmentPriority,
)

# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_context_budget_in_registry():
    """'context_budget' key must exist in LONGCONTEXT_STRATEGY_REGISTRY."""
    assert "context_budget" in LONGCONTEXT_STRATEGY_REGISTRY


def test_registry_maps_to_correct_class():
    """Registry value for 'context_budget' must be ContextBudgetController."""
    assert LONGCONTEXT_STRATEGY_REGISTRY["context_budget"] is ContextBudgetController


def test_construct_from_registry():
    """Controller instantiated via registry is fully functional."""
    cls = LONGCONTEXT_STRATEGY_REGISTRY["context_budget"]
    ctrl = cls()
    assert isinstance(ctrl, ContextBudgetController)
    assert ctrl.total_tokens() == 0


# ---------------------------------------------------------------------------
# Spec integration scenario
# ---------------------------------------------------------------------------


def test_integration_eviction_scenario():
    """Full spec scenario: max=1000, reserve=100; BACKGROUND evicted, survivors verified.

    Segments:
        CRITICAL  100 tokens  (evictable=False)
        HIGH      300 tokens
        MEDIUM    200 tokens
        BACKGROUND 350 tokens
        ─────────────────────
        total     950 tokens

    trigger = 0.85 * (1000 - 100) = 765
    950 > 765 → eviction needed.

    Expected: BACKGROUND (350) is evicted first (lowest priority, largest in its tier).
    After eviction: total = 600 < 765 → stop.
    """
    config = ContextBudgetConfig(max_tokens=1_000, reserve_tokens=100, trigger_ratio=0.85)
    ctrl = ContextBudgetController(config)

    ctrl.add_segment(ContextSegment("system", 100, SegmentPriority.CRITICAL, evictable=False))
    ctrl.add_segment(ContextSegment("history", 300, SegmentPriority.HIGH))
    ctrl.add_segment(ContextSegment("tools", 200, SegmentPriority.MEDIUM))
    ctrl.add_segment(ContextSegment("retrieved_docs", 350, SegmentPriority.BACKGROUND))

    assert ctrl.total_tokens() == 950
    assert ctrl.needs_eviction() is True

    evicted = ctrl.evict()

    # BACKGROUND should be the only evicted segment
    assert evicted == ["retrieved_docs"]

    # Remaining total should be below threshold
    assert ctrl.total_tokens() == 600
    assert ctrl.needs_eviction() is False

    # Verify survivors still present
    summary = ctrl.budget_summary()
    assert summary["by_priority"]["CRITICAL"] == 100
    assert summary["by_priority"]["HIGH"] == 300
    assert summary["by_priority"]["MEDIUM"] == 200
    assert summary["by_priority"]["BACKGROUND"] == 0


def test_integration_allocate_with_eviction():
    """allocate() removes overflow segments and returns survivors."""
    config = ContextBudgetConfig(max_tokens=1_000, reserve_tokens=100, trigger_ratio=0.85)
    ctrl = ContextBudgetController(config)

    segments = [
        ContextSegment("system", 100, SegmentPriority.CRITICAL, evictable=False),
        ContextSegment("history", 300, SegmentPriority.HIGH),
        ContextSegment("tools", 200, SegmentPriority.MEDIUM),
        ContextSegment("retrieved_docs", 350, SegmentPriority.BACKGROUND),
    ]

    survivors = ctrl.allocate(segments)
    survivor_names = {s.name for s in survivors}

    assert "retrieved_docs" not in survivor_names
    assert "system" in survivor_names
    assert "history" in survivor_names
    assert "tools" in survivor_names
    assert ctrl.needs_eviction() is False


# ---------------------------------------------------------------------------
# Regression guards — pre-existing registry entries must not be disturbed
# ---------------------------------------------------------------------------


def test_regression_kv_int8_still_present():
    """kv_int8 must remain in registry after additive change."""
    assert "kv_int8" in LONGCONTEXT_STRATEGY_REGISTRY


def test_regression_attention_sinks_still_present():
    """attention_sinks must remain in registry after additive change."""
    assert "attention_sinks" in LONGCONTEXT_STRATEGY_REGISTRY


def test_regression_hierarchical_context_still_present():
    """hierarchical_context must remain in registry after additive change."""
    assert "hierarchical_context" in LONGCONTEXT_STRATEGY_REGISTRY


def test_regression_compaction_trigger_still_present():
    """compaction_trigger must remain in registry after additive change."""
    assert "compaction_trigger" in LONGCONTEXT_STRATEGY_REGISTRY
