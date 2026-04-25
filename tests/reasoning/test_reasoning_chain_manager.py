"""Tests for src/reasoning/reasoning_chain_manager.py — at least 20 tests."""
from __future__ import annotations
import pytest

from src.reasoning.reasoning_chain_manager import (
    ChainStep,
    ChainStrategy,
    ReasoningChainManager,
    CHAIN_MANAGER_REGISTRY,
    DEFAULT_CHAIN_MANAGER,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh() -> ReasoningChainManager:
    return ReasoningChainManager()


# ---------------------------------------------------------------------------
# 1. add_step with each strategy
# ---------------------------------------------------------------------------

def test_add_step_cot():
    mgr = fresh()
    step = mgr.add_step("think step", strategy=ChainStrategy.COT)
    assert step.strategy == ChainStrategy.COT
    assert step.content == "think step"


def test_add_step_tot():
    mgr = fresh()
    step = mgr.add_step("tree step", strategy=ChainStrategy.TOT)
    assert step.strategy == ChainStrategy.TOT


def test_add_step_mcts():
    mgr = fresh()
    step = mgr.add_step("mcts step", strategy=ChainStrategy.MCTS)
    assert step.strategy == ChainStrategy.MCTS


def test_add_step_scratchpad():
    mgr = fresh()
    step = mgr.add_step("scratch", strategy=ChainStrategy.SCRATCHPAD)
    assert step.strategy == ChainStrategy.SCRATCHPAD


# ---------------------------------------------------------------------------
# 2. Default strategy used when none provided
# ---------------------------------------------------------------------------

def test_add_step_default_strategy():
    mgr = ReasoningChainManager(default_strategy=ChainStrategy.TOT)
    step = mgr.add_step("no strategy given")
    assert step.strategy == ChainStrategy.TOT


# ---------------------------------------------------------------------------
# 3. Content truncation at max_step_length
# ---------------------------------------------------------------------------

def test_content_truncation():
    mgr = ReasoningChainManager(max_step_length=10)
    step = mgr.add_step("a" * 50)
    assert len(step.content) == 10


def test_content_not_truncated_when_short():
    mgr = ReasoningChainManager(max_step_length=100)
    step = mgr.add_step("short")
    assert step.content == "short"


# ---------------------------------------------------------------------------
# 4. max_steps overflow raises ValueError
# ---------------------------------------------------------------------------

def test_max_steps_overflow():
    mgr = ReasoningChainManager(max_steps=3)
    for i in range(3):
        mgr.add_step(f"step {i}")
    with pytest.raises(ValueError, match="max_steps"):
        mgr.add_step("overflow")


# ---------------------------------------------------------------------------
# 5. get_steps filtered by strategy
# ---------------------------------------------------------------------------

def test_get_steps_filtered():
    mgr = fresh()
    mgr.add_step("cot1", strategy=ChainStrategy.COT)
    mgr.add_step("tot1", strategy=ChainStrategy.TOT)
    mgr.add_step("cot2", strategy=ChainStrategy.COT)
    cot_steps = mgr.get_steps(strategy=ChainStrategy.COT)
    assert len(cot_steps) == 2
    assert all(s.strategy == ChainStrategy.COT for s in cot_steps)


def test_get_steps_no_filter_returns_all():
    mgr = fresh()
    mgr.add_step("a", strategy=ChainStrategy.COT)
    mgr.add_step("b", strategy=ChainStrategy.MCTS)
    assert len(mgr.get_steps()) == 2


def test_get_steps_empty_for_missing_strategy():
    mgr = fresh()
    mgr.add_step("cot", strategy=ChainStrategy.COT)
    assert mgr.get_steps(strategy=ChainStrategy.SCRATCHPAD) == []


# ---------------------------------------------------------------------------
# 6. summarize correctness
# ---------------------------------------------------------------------------

def test_summarize():
    mgr = fresh()
    mgr.add_step("line one", strategy=ChainStrategy.COT)
    mgr.add_step("line two", strategy=ChainStrategy.COT)
    assert mgr.summarize() == "line one\nline two"


def test_summarize_empty():
    mgr = fresh()
    assert mgr.summarize() == ""


# ---------------------------------------------------------------------------
# 7. export / from_export round-trip for all 4 strategies
# ---------------------------------------------------------------------------

def test_export_from_export_roundtrip():
    mgr = fresh()
    mgr.add_step("cot content", strategy=ChainStrategy.COT, score=0.9, metadata={"k": "v"})
    mgr.add_step("tot content", strategy=ChainStrategy.TOT, score=0.5)
    mgr.add_step("mcts content", strategy=ChainStrategy.MCTS)
    mgr.add_step("scratch content", strategy=ChainStrategy.SCRATCHPAD, score=-0.1)
    exported = mgr.export()
    restored = ReasoningChainManager.from_export(exported)
    assert len(restored.get_steps()) == 4
    orig_steps = mgr.get_steps()
    rest_steps = restored.get_steps()
    for o, r in zip(orig_steps, rest_steps):
        assert o.strategy == r.strategy
        assert o.content == r.content
        assert o.score == pytest.approx(r.score)
        assert o.metadata == r.metadata


# ---------------------------------------------------------------------------
# 8. from_export with missing 'strategy' key raises ValueError
# ---------------------------------------------------------------------------

def test_from_export_missing_strategy_key():
    with pytest.raises(ValueError, match="strategy"):
        ReasoningChainManager.from_export([{"content": "hello"}])


# ---------------------------------------------------------------------------
# 9. from_export with missing 'content' key raises ValueError
# ---------------------------------------------------------------------------

def test_from_export_missing_content_key():
    with pytest.raises(ValueError, match="content"):
        ReasoningChainManager.from_export([{"strategy": "cot"}])


# ---------------------------------------------------------------------------
# 10. clear() resets to empty
# ---------------------------------------------------------------------------

def test_clear():
    mgr = fresh()
    mgr.add_step("a", strategy=ChainStrategy.COT)
    mgr.add_step("b", strategy=ChainStrategy.MCTS)
    mgr.clear()
    assert mgr.get_steps() == []
    assert mgr.summarize() == ""


# ---------------------------------------------------------------------------
# 11. Adversarial: null bytes in content
# ---------------------------------------------------------------------------

def test_null_bytes_in_content():
    mgr = fresh()
    content = "hello\x00world\x00"
    step = mgr.add_step(content, strategy=ChainStrategy.COT)
    assert step.content == content


# ---------------------------------------------------------------------------
# 12. Adversarial: very long content (truncated correctly)
# ---------------------------------------------------------------------------

def test_very_long_content_truncated():
    mgr = ReasoningChainManager(max_step_length=4096)
    long_content = "z" * 10_000
    step = mgr.add_step(long_content)
    assert len(step.content) == 4096


# ---------------------------------------------------------------------------
# 13. Adversarial: all-whitespace content
# ---------------------------------------------------------------------------

def test_all_whitespace_content():
    mgr = fresh()
    step = mgr.add_step("   \t\n   ", strategy=ChainStrategy.SCRATCHPAD)
    assert step.content == "   \t\n   "


# ---------------------------------------------------------------------------
# 14. Empty data list → empty chain manager
# ---------------------------------------------------------------------------

def test_from_export_empty_list():
    mgr = ReasoningChainManager.from_export([])
    assert mgr.get_steps() == []
    assert mgr.summarize() == ""


# ---------------------------------------------------------------------------
# 15. Score stored correctly
# ---------------------------------------------------------------------------

def test_score_stored():
    mgr = fresh()
    step = mgr.add_step("s", strategy=ChainStrategy.COT, score=0.75)
    assert step.score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 16. Metadata stored correctly and isolated
# ---------------------------------------------------------------------------

def test_metadata_isolation():
    mgr = fresh()
    meta = {"key": "value"}
    step = mgr.add_step("s", strategy=ChainStrategy.COT, metadata=meta)
    meta["key"] = "mutated"
    assert step.metadata["key"] == "value"


# ---------------------------------------------------------------------------
# 17. CHAIN_MANAGER_REGISTRY contains the expected key
# ---------------------------------------------------------------------------

def test_registry_contains_chain_manager():
    assert "chain_manager" in CHAIN_MANAGER_REGISTRY
    assert CHAIN_MANAGER_REGISTRY["chain_manager"] is ReasoningChainManager


# ---------------------------------------------------------------------------
# 18. DEFAULT_CHAIN_MANAGER is a ReasoningChainManager instance
# ---------------------------------------------------------------------------

def test_default_chain_manager_singleton():
    assert isinstance(DEFAULT_CHAIN_MANAGER, ReasoningChainManager)


# ---------------------------------------------------------------------------
# 19. export produces correct dict structure
# ---------------------------------------------------------------------------

def test_export_structure():
    mgr = fresh()
    mgr.add_step("content here", strategy=ChainStrategy.MCTS, score=0.42, metadata={"a": 1})
    exported = mgr.export()
    assert len(exported) == 1
    d = exported[0]
    assert d["strategy"] == "mcts"
    assert d["content"] == "content here"
    assert d["score"] == pytest.approx(0.42)
    assert d["metadata"] == {"a": 1}


# ---------------------------------------------------------------------------
# 20. get_steps returns a copy (mutation does not affect internal state)
# ---------------------------------------------------------------------------

def test_get_steps_returns_copy():
    mgr = fresh()
    mgr.add_step("original", strategy=ChainStrategy.COT)
    steps = mgr.get_steps()
    steps.clear()
    assert len(mgr.get_steps()) == 1


# ---------------------------------------------------------------------------
# 21. add_step with no metadata defaults to empty dict
# ---------------------------------------------------------------------------

def test_add_step_no_metadata_defaults_to_empty():
    mgr = fresh()
    step = mgr.add_step("hello", strategy=ChainStrategy.COT)
    assert step.metadata == {}


# ---------------------------------------------------------------------------
# 22. Multiple from_export calls are independent instances
# ---------------------------------------------------------------------------

def test_from_export_independent_instances():
    data = [{"strategy": "cot", "content": "step1", "score": 0.0, "metadata": {}}]
    mgr1 = ReasoningChainManager.from_export(data)
    mgr2 = ReasoningChainManager.from_export(data)
    mgr1.clear()
    assert len(mgr2.get_steps()) == 1
