"""Tests for QuantClaw precision security gating."""
from __future__ import annotations

import pytest

from src.safety.quantclaw_gate import (
    GatingContext,
    GateDecision,
    QuantClawGate,
    QuantLevel,
)


def create_gate(**kwargs) -> QuantClawGate:
    return QuantClawGate(**kwargs)


# ---------------------------------------------------------------------------
# Token-based gating
# ---------------------------------------------------------------------------


def test_small_input_triggers_fast() -> None:
    gate = create_gate(token_threshold_fast=2000, token_threshold_medium=8000)
    ctx = GatingContext(input_tokens=500, has_tool_calls=False)
    decision = gate.evaluate(ctx)
    assert decision.quant_level == QuantLevel.FAST
    assert "Small input" in decision.reasoning


def test_medium_input_triggers_medium() -> None:
    gate = create_gate(token_threshold_fast=2000, token_threshold_medium=8000)
    ctx = GatingContext(input_tokens=5000, has_tool_calls=False)
    decision = gate.evaluate(ctx)
    assert decision.quant_level == QuantLevel.MEDIUM
    assert "Medium input" in decision.reasoning


def test_large_input_triggers_thorough() -> None:
    gate = create_gate(token_threshold_fast=2000, token_threshold_medium=8000)
    ctx = GatingContext(input_tokens=10_000, has_tool_calls=False)
    decision = gate.evaluate(ctx)
    assert decision.quant_level == QuantLevel.THOROUGH
    assert "Large input" in decision.reasoning


def test_tool_calls_bump_level() -> None:
    gate = create_gate(token_threshold_fast=2000, token_threshold_medium=8000, tool_call_penalty=True)

    # Small input + tool calls → medium
    ctx = GatingContext(input_tokens=1500, has_tool_calls=True)
    decision = gate.evaluate(ctx)
    assert decision.quant_level == QuantLevel.MEDIUM
    assert "tool calls" in decision.reasoning.lower()

    # Medium input + tool calls → thorough
    ctx = GatingContext(input_tokens=5000, has_tool_calls=True)
    decision = gate.evaluate(ctx)
    assert decision.quant_level == QuantLevel.THOROUGH
    assert "tool calls" in decision.reasoning.lower()


def test_tool_call_penalty_disabled() -> None:
    gate = create_gate(tool_call_penalty=False)

    ctx = GatingContext(input_tokens=500, has_tool_calls=True)
    decision = gate.evaluate(ctx)
    assert decision.quant_level == QuantLevel.FAST  # stays fast


def test_historical_risk_adjustment() -> None:
    gate = create_gate(risk_threshold_fast=0.1, risk_threshold_medium=0.4)

    # No risk, small input → fast
    ctx = GatingContext(input_tokens=500, has_tool_calls=False, historical_risk=0.05)
    assert gate.evaluate(ctx).quant_level == QuantLevel.FAST

    # Low risk but > fast threshold → medium
    ctx = GatingContext(input_tokens=500, has_tool_calls=False, historical_risk=0.15)
    assert gate.evaluate(ctx).quant_level == QuantLevel.MEDIUM

    # High risk > medium threshold → thorough
    ctx = GatingContext(input_tokens=500, has_tool_calls=False, historical_risk=0.5)
    assert gate.evaluate(ctx).quant_level == QuantLevel.THOROUGH


def test_filter_sets() -> None:
    gate = create_gate()

    fast_decision = gate.evaluate(GatingContext(input_tokens=500, has_tool_calls=False))
    # Check that at least one filter with a keyword is present
    fast_names = [f.lower() for f in fast_decision.applied_filters]
    assert any("lexical" in f or "keyword" in f or "zero_width" in f for f in fast_names)
    assert len(fast_decision.skipped_filters) > 0  # thorough filters skipped
    # Ensure thorough-specific filters are NOT in fast
    assert "clawdrain_full" not in fast_decision.applied_filters

    medium_decision = gate.evaluate(GatingContext(input_tokens=5000, has_tool_calls=False))
    assert "clawdrain_full" not in medium_decision.applied_filters
    assert "prism_all_hooks" not in medium_decision.applied_filters

    thorough_decision = gate.evaluate(GatingContext(input_tokens=10_000, has_tool_calls=False))
    assert "clawdrain_full" in thorough_decision.applied_filters
    assert "prism_all_hooks" in thorough_decision.applied_filters
    assert len(thorough_decision.skipped_filters) == 0


def test_estimated_latency_ms_set_by_level() -> None:
    gate = create_gate(latency_budgets_ms={
        QuantLevel.FAST: 0.05,
        QuantLevel.MEDIUM: 1.5,
        QuantLevel.THOROUGH: 15.0,
    })

    f = gate.evaluate(GatingContext(input_tokens=500, has_tool_calls=False))
    assert 0.01 <= f.estimated_latency_ms <= 0.2

    m = gate.evaluate(GatingContext(input_tokens=5000, has_tool_calls=False))
    assert 0.5 <= m.estimated_latency_ms <= 3.0

    t = gate.evaluate(GatingContext(input_tokens=10_000, has_tool_calls=False))
    assert 5.0 <= t.estimated_latency_ms <= 60.0


def test_should_run_full_safety_boolean() -> None:
    gate = create_gate(token_threshold_medium=5000)

    assert not gate.should_run_full_safety(GatingContext(input_tokens=1000, has_tool_calls=False))
    assert not gate.should_run_full_safety(GatingContext(input_tokens=4000, has_tool_calls=False))
    assert gate.should_run_full_safety(GatingContext(input_tokens=6000, has_tool_calls=False))
    assert gate.should_run_full_safety(GatingContext(input_tokens=500, has_tool_calls=True, historical_risk=0.5))


def test_get_recommended_filters() -> None:
    gate = create_gate()

    fast_filters = gate.get_recommended_filters(GatingContext(input_tokens=500, has_tool_calls=False))
    assert "lexical_entropy_quick" in fast_filters or "keyword_blocklist" in fast_filters

    thorough_filters = gate.get_recommended_filters(GatingContext(input_tokens=10_000, has_tool_calls=False))
    assert "clawdrain_full" in thorough_filters
    assert "prism_all_hooks" in thorough_filters


def test_stats_collection() -> None:
    gate = create_gate()

    # Make several calls
    for _ in range(3):
        gate.evaluate(GatingContext(input_tokens=500, has_tool_calls=False))  # fast
    for _ in range(2):
        gate.evaluate(GatingContext(input_tokens=5000, has_tool_calls=False))  # medium
    for _ in range(1):
        gate.evaluate(GatingContext(input_tokens=10_000, has_tool_calls=False))  # thorough

    stats = gate.get_stats()
    assert stats["total_decisions"] == 6
    assert stats["fast"] == 3
    assert stats["medium"] == 2
    assert stats["thorough"] == 1
    assert 0 < stats["fast_pct"] < 1


def test_reset_stats() -> None:
    gate = create_gate()
    for _ in range(5):
        gate.evaluate(GatingContext(input_tokens=500, has_tool_calls=False))

    gate.reset_stats()
    stats = gate.get_stats()
    assert stats["total_decisions"] == 0
    assert stats["fast"] == 0


def test_custom_latency_budgets() -> None:
    gate = create_gate(latency_budgets_ms={
        QuantLevel.FAST: 0.2,
        QuantLevel.MEDIUM: 5.0,
        QuantLevel.THOROUGH: 50.0,
    })

    f = gate.evaluate(GatingContext(input_tokens=500, has_tool_calls=False))
    assert 0.05 <= f.estimated_latency_ms <= 0.5  # 20ms budget, allow some slack

    t = gate.evaluate(GatingContext(input_tokens=12_000, has_tool_calls=True))
    assert 10.0 <= t.estimated_latency_ms <= 60.0


def test_combined_triggers() -> None:
    gate = create_gate(token_threshold_fast=2000, token_threshold_medium=8000, risk_threshold_fast=0.1)

    # Medium tokens + high risk → thorough (risks bumps medium -> thorough)
    ctx = GatingContext(input_tokens=5000, has_tool_calls=False, historical_risk=0.5)
    decision = gate.evaluate(ctx)
    assert decision.quant_level == QuantLevel.THOROUGH
    assert "historical risk" in decision.reasoning.lower()


def test_empty_context_uses_defaults() -> None:
    gate = create_gate()
    # Just token info, other fields use defaults
    ctx = GatingContext(input_tokens=500, has_tool_calls=False)
    decision = gate.evaluate(ctx)
    assert decision.quant_level == QuantLevel.FAST
    assert isinstance(decision.applied_filters, list)
    assert isinstance(decision.skipped_filters, list)
