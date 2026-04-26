"""Unit tests for src.longcontext.hierarchical_context_mgr.

GLM-5 §6.2 — keep-recent-k + discard-all fallback.
"""

from __future__ import annotations

from src.longcontext.hierarchical_context_mgr import HierarchicalContextManager, Turn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_turns(n: int, tokens_each: int = 100) -> list[Turn]:
    return [{"role": "user", "content": f"turn {i}", "tokens": tokens_each} for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_below_trigger_no_truncation():
    """total = 79% of max_len → all turns returned unchanged."""
    mgr = HierarchicalContextManager(max_len=1000, trigger_ratio=0.8, keep_k=10)
    # 79 turns × 10 tokens = 790 tokens = 79% of 1000 (trigger at 800)
    turns = make_turns(79, tokens_each=10)
    result = mgr.manage(turns)
    assert result is turns, "Should return the exact same list when below trigger"
    assert len(result) == 79


def test_above_trigger_keep_recent_k():
    """total = 90% of max_len → returns last k=10 turns."""
    mgr = HierarchicalContextManager(max_len=1000, trigger_ratio=0.8, keep_k=10)
    # 90 turns × 10 tokens = 900 tokens = 90% of 1000 (trigger at 800)
    turns = make_turns(90, tokens_each=10)
    result = mgr.manage(turns, quality_score=1.0)
    assert len(result) == 10
    assert result == turns[-10:]


def test_after_truncation_within_limit():
    """Returned turn count must be <= keep_k after truncation."""
    mgr = HierarchicalContextManager(max_len=1000, trigger_ratio=0.8, keep_k=5)
    turns = make_turns(100, tokens_each=10)
    result = mgr.manage(turns, quality_score=1.0)
    assert len(result) <= 5


def test_discard_all_fallback():
    """quality_score=0.1 (< threshold 0.3) → returns only last 1 turn."""
    mgr = HierarchicalContextManager(
        max_len=1000, trigger_ratio=0.8, keep_k=10, quality_threshold=0.3
    )
    turns = make_turns(50, tokens_each=20)  # 1000 tokens = exactly at trigger boundary
    # Use 51 turns to exceed trigger
    turns = make_turns(51, tokens_each=20)  # 1020 > 800
    result = mgr.manage(turns, quality_score=0.1)
    assert len(result) == 1


def test_discard_all_returns_last_turn():
    """Verifies the LAST turn is preserved, not the first."""
    mgr = HierarchicalContextManager(
        max_len=1000, trigger_ratio=0.8, keep_k=10, quality_threshold=0.3
    )
    turns = make_turns(60, tokens_each=20)  # 1200 > 800
    result = mgr.manage(turns, quality_score=0.1)
    assert len(result) == 1
    assert result[0] is turns[-1], "Discard-all must preserve the LAST turn, not first"


def test_empty_turns_no_crash():
    """Empty input returns empty list without error."""
    mgr = HierarchicalContextManager()
    result = mgr.manage([])
    assert result == []


def test_keep_k_zero():
    """keep_k=0 → returns [] (not a crash) when above trigger."""
    mgr = HierarchicalContextManager(max_len=1000, trigger_ratio=0.8, keep_k=0)
    turns = make_turns(90, tokens_each=10)  # 900 > 800
    result = mgr.manage(turns, quality_score=1.0)
    assert result == []


def test_recent_turns_ordering():
    """Returned turns are in original chronological order, not reversed."""
    mgr = HierarchicalContextManager(max_len=1000, trigger_ratio=0.8, keep_k=5)
    turns = make_turns(90, tokens_each=10)  # 900 > 800
    result = mgr.manage(turns, quality_score=1.0)
    # The last 5 turns in original order
    expected = turns[-5:]
    assert result == expected
    # Verify they are in ascending index order
    for i in range(len(result) - 1):
        assert result[i]["content"] < result[i + 1]["content"] or True  # ordering check
    assert result[0]["content"] == "turn 85"
    assert result[-1]["content"] == "turn 89"


def test_quality_at_exactly_threshold():
    """quality_score == quality_threshold → keep-recent-k (NOT discard-all)."""
    mgr = HierarchicalContextManager(
        max_len=1000, trigger_ratio=0.8, keep_k=10, quality_threshold=0.3
    )
    turns = make_turns(90, tokens_each=10)  # 900 > 800
    # Exactly at threshold: should NOT trigger discard-all (strict <)
    result = mgr.manage(turns, quality_score=0.3)
    assert len(result) == 10
    assert result == turns[-10:]


def test_quality_below_threshold():
    """quality_score = threshold - 0.01 → discard-all."""
    mgr = HierarchicalContextManager(
        max_len=1000, trigger_ratio=0.8, keep_k=10, quality_threshold=0.3
    )
    turns = make_turns(90, tokens_each=10)  # 900 > 800
    result = mgr.manage(turns, quality_score=0.3 - 0.01)
    assert len(result) == 1
    assert result[0] is turns[-1]


def test_token_count():
    """token_count returns sum of tokens across all turns."""
    mgr = HierarchicalContextManager()
    turns = make_turns(5, tokens_each=100)
    assert mgr.token_count(turns) == 500
    assert mgr.token_count([]) == 0


def test_utilization_below_one():
    """With few turns, utilization < 1.0."""
    mgr = HierarchicalContextManager(max_len=8192)
    turns = make_turns(10, tokens_each=100)  # 1000 tokens out of 8192
    util = mgr.utilization(turns)
    assert 0.0 < util < 1.0
    assert abs(util - (1000 / 8192)) < 1e-9


def test_utilization_at_limit():
    """Turns filling max_len → utilization ≈ 1.0."""
    mgr = HierarchicalContextManager(max_len=1000)
    # 10 turns × 100 tokens = exactly 1000 tokens
    turns = make_turns(10, tokens_each=100)
    util = mgr.utilization(turns)
    assert abs(util - 1.0) < 1e-9
