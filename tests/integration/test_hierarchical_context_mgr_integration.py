"""Integration tests for hierarchical_context_mgr registration in longcontext.

GLM-5 §6.2 — verifies LONGCONTEXT_STRATEGY_REGISTRY wiring.
"""
from __future__ import annotations

from src.longcontext import LONGCONTEXT_STRATEGY_REGISTRY, HierarchicalContextManager
from src.longcontext.hierarchical_context_mgr import Turn


def make_turns(n: int, tokens_each: int = 100) -> list[Turn]:
    return [
        {"role": "user", "content": f"turn {i}", "tokens": tokens_each}
        for i in range(n)
    ]


def test_hierarchical_context_in_registry():
    """'hierarchical_context' key must exist in LONGCONTEXT_STRATEGY_REGISTRY."""
    assert "hierarchical_context" in LONGCONTEXT_STRATEGY_REGISTRY


def test_registry_maps_to_correct_class():
    """Registry value must be HierarchicalContextManager."""
    assert LONGCONTEXT_STRATEGY_REGISTRY["hierarchical_context"] is HierarchicalContextManager


def test_construct_from_registry_and_manage():
    """Construct from registry, call manage() on 20 turns, result within bounds."""
    cls = LONGCONTEXT_STRATEGY_REGISTRY["hierarchical_context"]
    # 20 turns × 500 tokens = 10000 tokens > 8192 * 0.8 = 6553.6 → triggers truncation
    mgr = cls(max_len=8192, trigger_ratio=0.8, keep_k=10)
    turns = make_turns(20, tokens_each=500)
    result = mgr.manage(turns, quality_score=1.0)
    assert len(result) <= 10
    assert len(result) > 0


def test_regression_kv_int8_still_present():
    """Regression guard: kv_int8 must remain in registry after additive change."""
    assert "kv_int8" in LONGCONTEXT_STRATEGY_REGISTRY


def test_regression_attention_sinks_still_present():
    """Regression guard: attention_sinks must remain in registry."""
    assert "attention_sinks" in LONGCONTEXT_STRATEGY_REGISTRY


def test_regression_context_compaction_still_present():
    """Regression guard: context_compaction must remain in registry."""
    assert "context_compaction" in LONGCONTEXT_STRATEGY_REGISTRY
