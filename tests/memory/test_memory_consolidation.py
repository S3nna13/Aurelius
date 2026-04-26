"""Tests for src/memory/memory_consolidation.py — ~45 tests."""

import pytest

from src.memory import MEMORY_REGISTRY
from src.memory.episodic_memory import MemoryEntry
from src.memory.memory_consolidation import (
    ConsolidationPolicy,
    ConsolidationResult,
    MemoryConsolidator,
)
from src.memory.semantic_memory import SemanticMemory

# ---------------------------------------------------------------------------
# ConsolidationPolicy enum
# ---------------------------------------------------------------------------


def test_policy_recency():
    assert ConsolidationPolicy.RECENCY == "recency"


def test_policy_importance():
    assert ConsolidationPolicy.IMPORTANCE == "importance"


def test_policy_frequency():
    assert ConsolidationPolicy.FREQUENCY == "frequency"


# ---------------------------------------------------------------------------
# ConsolidationResult dataclass
# ---------------------------------------------------------------------------


def test_consolidation_result_fields():
    cr = ConsolidationResult(consolidated_count=3, summary="hello", dropped_ids=["a", "b"])
    assert cr.consolidated_count == 3
    assert cr.summary == "hello"
    assert cr.dropped_ids == ["a", "b"]


def test_consolidation_result_zero():
    cr = ConsolidationResult(consolidated_count=0, summary="", dropped_ids=[])
    assert cr.consolidated_count == 0
    assert cr.dropped_ids == []


# ---------------------------------------------------------------------------
# MemoryConsolidator — init
# ---------------------------------------------------------------------------


def test_consolidator_default_policy():
    mc = MemoryConsolidator()
    assert mc.policy == ConsolidationPolicy.IMPORTANCE


def test_consolidator_custom_policy():
    mc = MemoryConsolidator(policy=ConsolidationPolicy.RECENCY)
    assert mc.policy == ConsolidationPolicy.RECENCY


def test_consolidator_default_decay_factor():
    mc = MemoryConsolidator()
    assert mc.decay_factor == pytest.approx(0.95)


def test_consolidator_custom_decay_factor():
    mc = MemoryConsolidator(decay_factor=0.8)
    assert mc.decay_factor == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# decay_importance
# ---------------------------------------------------------------------------


def _make_entries(*importances):
    return [
        MemoryEntry(role="user", content=f"memory {i}", importance=imp)
        for i, imp in enumerate(importances)
    ]


def test_decay_importance_multiplies_by_factor():
    mc = MemoryConsolidator(decay_factor=0.95)
    entries = _make_entries(1.0, 0.8)
    decayed = mc.decay_importance(entries, steps=1)
    assert decayed[0].importance == pytest.approx(0.95)
    assert decayed[1].importance == pytest.approx(0.76)


def test_decay_importance_non_destructive():
    mc = MemoryConsolidator(decay_factor=0.9)
    entries = _make_entries(1.0)
    mc.decay_importance(entries, steps=1)
    assert entries[0].importance == pytest.approx(1.0)


def test_decay_importance_steps_2():
    mc = MemoryConsolidator(decay_factor=0.9)
    entries = _make_entries(1.0)
    decayed = mc.decay_importance(entries, steps=2)
    assert decayed[0].importance == pytest.approx(0.81)


def test_decay_importance_steps_0():
    mc = MemoryConsolidator(decay_factor=0.9)
    entries = _make_entries(0.5)
    decayed = mc.decay_importance(entries, steps=0)
    assert decayed[0].importance == pytest.approx(0.5)


def test_decay_importance_returns_list():
    mc = MemoryConsolidator()
    decayed = mc.decay_importance(_make_entries(1.0))
    assert isinstance(decayed, list)


def test_decay_importance_length_preserved():
    mc = MemoryConsolidator()
    entries = _make_entries(1.0, 0.5, 0.2)
    assert len(mc.decay_importance(entries)) == 3


def test_decay_importance_returns_copies():
    mc = MemoryConsolidator(decay_factor=0.9)
    entries = _make_entries(1.0)
    decayed = mc.decay_importance(entries)
    assert decayed[0] is not entries[0]


def test_decay_importance_large_steps():
    mc = MemoryConsolidator(decay_factor=0.5)
    entries = _make_entries(1.0)
    decayed = mc.decay_importance(entries, steps=10)
    assert decayed[0].importance == pytest.approx(0.5**10)


# ---------------------------------------------------------------------------
# select_for_consolidation
# ---------------------------------------------------------------------------


def test_select_returns_below_threshold():
    mc = MemoryConsolidator()
    entries = _make_entries(0.1, 0.5, 0.9)
    selected = mc.select_for_consolidation(entries, threshold=0.3)
    assert len(selected) == 1
    assert selected[0].importance == pytest.approx(0.1)


def test_select_excludes_at_threshold():
    mc = MemoryConsolidator()
    entries = _make_entries(0.3, 0.1)
    selected = mc.select_for_consolidation(entries, threshold=0.3)
    # 0.3 is NOT < 0.3, so only 0.1 entry
    assert len(selected) == 1
    assert selected[0].importance == pytest.approx(0.1)


def test_select_capped_at_max_count():
    mc = MemoryConsolidator()
    entries = _make_entries(*[0.1] * 15)
    selected = mc.select_for_consolidation(entries, threshold=0.3, max_count=5)
    assert len(selected) == 5


def test_select_sorted_importance_asc():
    mc = MemoryConsolidator()
    entries = _make_entries(0.25, 0.05, 0.15)
    selected = mc.select_for_consolidation(entries, threshold=0.3)
    importances = [e.importance for e in selected]
    assert importances == sorted(importances)


def test_select_empty_when_all_above_threshold():
    mc = MemoryConsolidator()
    entries = _make_entries(0.5, 0.8, 1.0)
    assert mc.select_for_consolidation(entries, threshold=0.3) == []


def test_select_all_when_all_below_threshold():
    mc = MemoryConsolidator()
    entries = _make_entries(0.1, 0.2, 0.25)
    selected = mc.select_for_consolidation(entries, threshold=0.3, max_count=10)
    assert len(selected) == 3


def test_select_returns_list():
    mc = MemoryConsolidator()
    result = mc.select_for_consolidation([])
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# consolidate
# ---------------------------------------------------------------------------


def test_consolidate_dropped_ids_match():
    mc = MemoryConsolidator()
    entries = _make_entries(0.1, 0.2, 0.8)
    result = mc.consolidate(entries)
    low_ids = {e.id for e in entries if e.importance < 0.3}
    assert set(result.dropped_ids) == low_ids


def test_consolidate_count_correct():
    mc = MemoryConsolidator()
    entries = _make_entries(0.1, 0.2, 0.9)
    result = mc.consolidate(entries)
    assert result.consolidated_count == 2


def test_consolidate_count_zero_when_none_below():
    mc = MemoryConsolidator()
    entries = _make_entries(0.5, 0.8)
    result = mc.consolidate(entries)
    assert result.consolidated_count == 0


def test_consolidate_summary_is_string():
    mc = MemoryConsolidator()
    entries = _make_entries(0.1)
    result = mc.consolidate(entries)
    assert isinstance(result.summary, str)


def test_consolidate_summary_non_empty():
    mc = MemoryConsolidator()
    entries = _make_entries(0.1)
    result = mc.consolidate(entries)
    assert len(result.summary) > 0


def test_consolidate_summary_contains_count():
    mc = MemoryConsolidator()
    entries = _make_entries(0.1, 0.2)
    result = mc.consolidate(entries)
    assert "2" in result.summary


def test_consolidate_returns_consolidation_result():
    mc = MemoryConsolidator()
    result = mc.consolidate(_make_entries(0.1))
    assert isinstance(result, ConsolidationResult)


def test_consolidate_with_semantic_memory_adds_concepts():
    mc = MemoryConsolidator()
    sm = SemanticMemory()
    # MemoryEntry has role field which should be added as concept
    entries = [MemoryEntry(role="assistant", content="test", importance=0.1)]
    mc.consolidate(entries, semantic_memory=sm)
    # "assistant" role should be registered as a concept
    assert sm.get_concept("assistant") is not None


def test_consolidate_with_semantic_memory_no_duplicate_concepts():
    mc = MemoryConsolidator()
    sm = SemanticMemory()
    entries = [
        MemoryEntry(role="user", content="a", importance=0.1),
        MemoryEntry(role="user", content="b", importance=0.2),
    ]
    mc.consolidate(entries, semantic_memory=sm)
    # concept_count should be 1 ("user"), not 2
    assert sm.concept_count() == 1


def test_consolidate_without_semantic_memory():
    mc = MemoryConsolidator()
    entries = _make_entries(0.1, 0.5)
    result = mc.consolidate(entries, semantic_memory=None)
    assert result.consolidated_count == 1


def test_consolidate_dropped_ids_empty_when_nothing_selected():
    mc = MemoryConsolidator()
    entries = _make_entries(0.5, 0.9)
    result = mc.consolidate(entries)
    assert result.dropped_ids == []


# ---------------------------------------------------------------------------
# MEMORY_REGISTRY
# ---------------------------------------------------------------------------


def test_memory_registry_consolidator_exists():
    assert "consolidator" in MEMORY_REGISTRY


def test_memory_registry_consolidator_is_consolidator():
    assert isinstance(MEMORY_REGISTRY["consolidator"], MemoryConsolidator)
