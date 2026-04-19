"""Integration tests for context_compaction registration in longcontext."""

from __future__ import annotations

from src import longcontext
from src.longcontext import LONGCONTEXT_STRATEGY_REGISTRY, ContextCompactor
from src.longcontext.context_compaction import Turn


def test_registry_contains_context_compaction():
    assert "context_compaction" in LONGCONTEXT_STRATEGY_REGISTRY
    assert LONGCONTEXT_STRATEGY_REGISTRY["context_compaction"] is ContextCompactor


def test_registered_class_constructible():
    cls = LONGCONTEXT_STRATEGY_REGISTRY["context_compaction"]
    inst = cls(
        summarize_fn=lambda turns: "S",
        token_counter=lambda s: len(s.split()),
        target_tokens=100,
    )
    assert inst.target_tokens == 100


def test_regression_kv_int8_still_present():
    assert "kv_int8" in LONGCONTEXT_STRATEGY_REGISTRY


def test_regression_attention_sinks_still_present():
    assert "attention_sinks" in LONGCONTEXT_STRATEGY_REGISTRY


def test_regression_ring_attention_still_present():
    assert "ring_attention" in LONGCONTEXT_STRATEGY_REGISTRY


def test_end_to_end_compaction():
    turns = [Turn(role="system", content="sys prompt", kind="system")]
    for i in range(20):
        turns.append(
            Turn(
                role="user" if i % 2 == 0 else "assistant",
                content=" ".join(f"tok{i}_{j}" for j in range(10)),
                kind="message",
            )
        )
    c = ContextCompactor(
        summarize_fn=lambda ts: f"summary of {len(ts)} turns",
        token_counter=lambda s: len(s.split()),
        target_tokens=40,
        keep_last_n=3,
    )
    out = c.compact(turns)
    assert len(out) < len(turns)
    # System preserved.
    assert any(t.kind == "system" and "sys prompt" in t.content for t in out)
    # Tail preserved.
    assert out[-1] == turns[-1]


def test_public_exports():
    assert hasattr(longcontext, "ContextCompactor")
    assert hasattr(longcontext, "CompactionTurn")
