"""Unit tests for src.longcontext.context_budget_controller.

Kimi K2.6-inspired 256K adaptive token budget prioritisation.
"""

from __future__ import annotations

import pytest

from src.longcontext.context_budget_controller import (
    ContextBudgetConfig,
    ContextBudgetController,
    ContextSegment,
    SegmentPriority,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(
    name: str,
    tokens: int,
    priority: SegmentPriority = SegmentPriority.MEDIUM,
    evictable: bool = True,
) -> ContextSegment:
    return ContextSegment(name=name, tokens=tokens, priority=priority, evictable=evictable)


def _ctrl(
    max_tokens: int = 1_000,
    reserve_tokens: int = 100,
    trigger_ratio: float = 0.85,
) -> ContextBudgetController:
    return ContextBudgetController(
        ContextBudgetConfig(
            max_tokens=max_tokens,
            reserve_tokens=reserve_tokens,
            trigger_ratio=trigger_ratio,
        )
    )


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    """Default ContextBudgetConfig values match the spec."""
    cfg = ContextBudgetConfig()
    assert cfg.max_tokens == 262_144
    assert cfg.reserve_tokens == 4_096
    assert cfg.trigger_ratio == 0.85


# ---------------------------------------------------------------------------
# 2. test_add_segment
# ---------------------------------------------------------------------------


def test_add_segment():
    """add_segment increases total_tokens by the segment's token count."""
    ctrl = _ctrl()
    assert ctrl.total_tokens() == 0
    ctrl.add_segment(_seg("a", 200))
    assert ctrl.total_tokens() == 200
    ctrl.add_segment(_seg("b", 50))
    assert ctrl.total_tokens() == 250


# ---------------------------------------------------------------------------
# 3. test_add_duplicate_raises
# ---------------------------------------------------------------------------


def test_add_duplicate_raises():
    """Adding a segment with a duplicate name raises ValueError."""
    ctrl = _ctrl()
    ctrl.add_segment(_seg("x", 100))
    with pytest.raises(ValueError, match="x"):
        ctrl.add_segment(_seg("x", 50))


# ---------------------------------------------------------------------------
# 4. test_remove_segment
# ---------------------------------------------------------------------------


def test_remove_segment():
    """remove_segment returns True and decreases total_tokens."""
    ctrl = _ctrl()
    ctrl.add_segment(_seg("a", 300))
    assert ctrl.total_tokens() == 300
    result = ctrl.remove_segment("a")
    assert result is True
    assert ctrl.total_tokens() == 0


# ---------------------------------------------------------------------------
# 5. test_remove_nonexistent
# ---------------------------------------------------------------------------


def test_remove_nonexistent():
    """remove_segment on a missing name returns False."""
    ctrl = _ctrl()
    assert ctrl.remove_segment("ghost") is False


# ---------------------------------------------------------------------------
# 6. test_available_tokens
# ---------------------------------------------------------------------------


def test_available_tokens():
    """available_tokens = max_tokens - reserve_tokens - total_tokens."""
    ctrl = _ctrl(max_tokens=1_000, reserve_tokens=100)
    ctrl.add_segment(_seg("a", 300))
    # usable = 1000 - 100 = 900; available = 900 - 300 = 600
    assert ctrl.available_tokens() == 600


# ---------------------------------------------------------------------------
# 7. test_needs_eviction_false
# ---------------------------------------------------------------------------


def test_needs_eviction_false():
    """Below trigger threshold → needs_eviction() is False."""
    # trigger = 0.85 * (1000 - 100) = 765; total = 700 → no eviction
    ctrl = _ctrl(max_tokens=1_000, reserve_tokens=100, trigger_ratio=0.85)
    ctrl.add_segment(_seg("a", 700))
    assert ctrl.needs_eviction() is False


# ---------------------------------------------------------------------------
# 8. test_needs_eviction_true
# ---------------------------------------------------------------------------


def test_needs_eviction_true():
    """Above trigger threshold → needs_eviction() is True."""
    # trigger = 0.85 * 900 = 765; total = 800 → eviction needed
    ctrl = _ctrl(max_tokens=1_000, reserve_tokens=100, trigger_ratio=0.85)
    ctrl.add_segment(_seg("a", 800))
    assert ctrl.needs_eviction() is True


# ---------------------------------------------------------------------------
# 9. test_evict_low_priority_first
# ---------------------------------------------------------------------------


def test_evict_low_priority_first():
    """BACKGROUND is evicted before MEDIUM when budget is exceeded."""
    ctrl = _ctrl(max_tokens=1_000, reserve_tokens=100, trigger_ratio=0.85)
    # total = 800 > 765 → needs eviction
    ctrl.add_segment(_seg("bg", 400, SegmentPriority.BACKGROUND))
    ctrl.add_segment(_seg("med", 400, SegmentPriority.MEDIUM))
    evicted = ctrl.evict()
    # BACKGROUND (priority=4) must be evicted before MEDIUM (priority=2)
    assert "bg" in evicted
    assert "med" not in evicted


# ---------------------------------------------------------------------------
# 10. test_evict_largest_first_within_priority
# ---------------------------------------------------------------------------


def test_evict_largest_first_within_priority():
    """Within same priority, the larger segment is evicted first."""
    ctrl = _ctrl(max_tokens=1_000, reserve_tokens=100, trigger_ratio=0.85)
    # total = 800 > 765; both BACKGROUND
    ctrl.add_segment(_seg("small_bg", 150, SegmentPriority.BACKGROUND))
    ctrl.add_segment(_seg("large_bg", 650, SegmentPriority.BACKGROUND))
    evicted = ctrl.evict()
    # Evicting large_bg (650) reduces total to 150 < 765 — stops there
    assert evicted[0] == "large_bg"
    assert "small_bg" not in evicted


# ---------------------------------------------------------------------------
# 11. test_evict_critical_never
# ---------------------------------------------------------------------------


def test_evict_critical_never():
    """CRITICAL segments (evictable=False) are never evicted."""
    ctrl = _ctrl(max_tokens=1_000, reserve_tokens=100, trigger_ratio=0.85)
    # Only segment is non-evictable → no eviction possible
    ctrl.add_segment(ContextSegment("crit", 900, SegmentPriority.CRITICAL, evictable=False))
    assert ctrl.needs_eviction() is True
    evicted = ctrl.evict()
    assert evicted == []
    # Critical segment must still be there
    assert ctrl.total_tokens() == 900


# ---------------------------------------------------------------------------
# 12. test_evict_returns_names
# ---------------------------------------------------------------------------


def test_evict_returns_names():
    """evict() returns a list of the names of evicted segments."""
    ctrl = _ctrl(max_tokens=1_000, reserve_tokens=100, trigger_ratio=0.85)
    ctrl.add_segment(_seg("bg1", 500, SegmentPriority.BACKGROUND))
    ctrl.add_segment(_seg("bg2", 500, SegmentPriority.BACKGROUND))
    evicted = ctrl.evict()
    assert isinstance(evicted, list)
    assert all(isinstance(n, str) for n in evicted)
    for name in evicted:
        assert name in {"bg1", "bg2"}


# ---------------------------------------------------------------------------
# 13. test_allocate_fits_all
# ---------------------------------------------------------------------------


def test_allocate_fits_all():
    """allocate() returns all segments when they fit within budget."""
    ctrl = _ctrl(max_tokens=1_000, reserve_tokens=100, trigger_ratio=0.85)
    segments = [
        _seg("s1", 100, SegmentPriority.HIGH),
        _seg("s2", 100, SegmentPriority.MEDIUM),
    ]
    # total = 200 < 765 → nothing evicted
    survivors = ctrl.allocate(segments)
    assert len(survivors) == 2
    survivor_names = {s.name for s in survivors}
    assert survivor_names == {"s1", "s2"}


# ---------------------------------------------------------------------------
# 14. test_allocate_evicts_overflow
# ---------------------------------------------------------------------------


def test_allocate_evicts_overflow():
    """allocate() evicts lower-priority segments when budget overflows."""
    ctrl = _ctrl(max_tokens=1_000, reserve_tokens=100, trigger_ratio=0.85)
    segments = [
        _seg("crit", 200, SegmentPriority.CRITICAL, evictable=False),
        _seg("high", 200, SegmentPriority.HIGH),
        _seg("bg", 500, SegmentPriority.BACKGROUND),
    ]
    # total = 900 > 765; BACKGROUND (500) should be evicted → total = 400 < 765
    survivors = ctrl.allocate(segments)
    survivor_names = {s.name for s in survivors}
    assert "bg" not in survivor_names
    assert "crit" in survivor_names
    assert "high" in survivor_names


# ---------------------------------------------------------------------------
# 15. test_budget_summary
# ---------------------------------------------------------------------------


def test_budget_summary():
    """budget_summary() returns correct keys and aggregated values."""
    ctrl = _ctrl(max_tokens=1_000, reserve_tokens=100, trigger_ratio=0.85)
    ctrl.add_segment(_seg("a", 100, SegmentPriority.CRITICAL, evictable=False))
    ctrl.add_segment(_seg("b", 200, SegmentPriority.HIGH))
    ctrl.add_segment(_seg("c", 50, SegmentPriority.LOW))
    summary = ctrl.budget_summary()

    assert "total" in summary
    assert "available" in summary
    assert "by_priority" in summary

    assert summary["total"] == 350
    # available = 900 - 350 = 550
    assert summary["available"] == 550

    bp = summary["by_priority"]
    assert set(bp.keys()) == {"CRITICAL", "HIGH", "MEDIUM", "LOW", "BACKGROUND"}
    assert bp["CRITICAL"] == 100
    assert bp["HIGH"] == 200
    assert bp["MEDIUM"] == 0
    assert bp["LOW"] == 50
    assert bp["BACKGROUND"] == 0


# ---------------------------------------------------------------------------
# 16. test_evict_until_ok
# ---------------------------------------------------------------------------


def test_evict_until_ok():
    """eviction loop continues until needs_eviction() returns False."""
    ctrl = _ctrl(max_tokens=1_000, reserve_tokens=100, trigger_ratio=0.85)
    # Add many small BACKGROUND segments; each alone may not stop eviction
    for i in range(10):
        ctrl.add_segment(_seg(f"bg{i}", 80, SegmentPriority.BACKGROUND))
    # total = 800 > 765
    assert ctrl.needs_eviction() is True
    evicted = ctrl.evict()
    # After eviction the controller must be below threshold
    assert ctrl.needs_eviction() is False
    # At least some segments must have been evicted
    assert len(evicted) > 0
    # All evicted names should have been bg* segments
    for name in evicted:
        assert name.startswith("bg")
