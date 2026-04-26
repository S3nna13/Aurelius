"""
Tests for src/eval/multi_agent_debate_eval.py
Covers: DebateConfig, AgentTurn, DebateEvalResult, DebateEvaluator
Cycle 137-D — 14 unit tests + 1 integration test
"""

from __future__ import annotations

import pytest

from src.eval import BENCHMARK_REGISTRY
from src.eval.multi_agent_debate_eval import (
    AgentTurn,
    DebateConfig,
    DebateEvalResult,
    DebateEvaluator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_turn(
    agent_id: int,
    round_idx: int,
    position: str,
    arguments: list[str] | None = None,
) -> AgentTurn:
    return AgentTurn(
        agent_id=agent_id,
        round_idx=round_idx,
        position=position,
        arguments=arguments or [],
    )


# ---------------------------------------------------------------------------
# Test 1 — DebateConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = DebateConfig()
    assert cfg.n_agents == 3
    assert cfg.n_rounds == 3
    assert cfg.convergence_threshold == 0.8
    assert cfg.ngram_n == 2


# ---------------------------------------------------------------------------
# Test 2 — Jaccard: identical texts → 1.0
# ---------------------------------------------------------------------------


def test_jaccard_identical():
    ev = DebateEvaluator()
    assert ev.jaccard_similarity(
        "the cat sat on the mat", "the cat sat on the mat"
    ) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 3 — Jaccard: disjoint texts → 0.0
# ---------------------------------------------------------------------------


def test_jaccard_disjoint():
    ev = DebateEvaluator()
    assert ev.jaccard_similarity("alpha beta gamma", "delta epsilon zeta") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 4 — Jaccard: partial overlap → correct fraction
# ---------------------------------------------------------------------------


def test_jaccard_partial():
    ev = DebateEvaluator()
    # A = {the, cat} B = {the, dog}  → intersection={the} union={the,cat,dog}
    result = ev.jaccard_similarity("the cat", "the dog")
    assert result == pytest.approx(1 / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 5 — position_drift: no change → drift ≈ 0
# ---------------------------------------------------------------------------


def test_position_drift_no_change():
    ev = DebateEvaluator()
    turns = [
        make_turn(0, 0, "the answer is yes"),
        make_turn(0, 1, "the answer is yes"),
        make_turn(1, 0, "climate change is real"),
        make_turn(1, 1, "climate change is real"),
    ]
    assert ev.position_drift(turns) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 6 — position_drift: full change → high drift
# ---------------------------------------------------------------------------


def test_position_drift_full_change():
    ev = DebateEvaluator()
    turns = [
        make_turn(0, 0, "alpha beta gamma"),
        make_turn(0, 1, "delta epsilon zeta"),
        make_turn(1, 0, "one two three"),
        make_turn(1, 1, "four five six"),
    ]
    drift = ev.position_drift(turns)
    # Both agents flipped completely, so drift should be 1.0
    assert drift == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 7 — consensus: all same → 1.0
# ---------------------------------------------------------------------------


def test_consensus_all_same():
    ev = DebateEvaluator()
    turns = [
        make_turn(0, 0, "the earth is round"),
        make_turn(1, 0, "the earth is round"),
        make_turn(2, 0, "the earth is round"),
    ]
    assert ev.consensus(turns, round_idx=0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 8 — consensus: all different → low score
# ---------------------------------------------------------------------------


def test_consensus_all_different():
    ev = DebateEvaluator()
    turns = [
        make_turn(0, 0, "alpha beta"),
        make_turn(1, 0, "gamma delta"),
        make_turn(2, 0, "epsilon zeta"),
    ]
    score = ev.consensus(turns, round_idx=0)
    assert score < 0.2


# ---------------------------------------------------------------------------
# Test 9 — argument_diversity: identical arguments → ≈ 0
# ---------------------------------------------------------------------------


def test_argument_diversity_identical():
    ev = DebateEvaluator()
    # Each agent makes the exact same single argument string — all pairs are
    # identical so pairwise diversity (1 - jaccard) should be 0.0.
    turns = [
        make_turn(0, 0, "pos", ["the sky is blue"]),
        make_turn(1, 0, "pos", ["the sky is blue"]),
        make_turn(2, 0, "pos", ["the sky is blue"]),
    ]
    diversity = ev.argument_diversity(turns)
    assert diversity == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 10 — argument_diversity: different arguments → > 0
# ---------------------------------------------------------------------------


def test_argument_diversity_different():
    ev = DebateEvaluator()
    turns = [
        make_turn(0, 0, "pos A", ["gravity causes apples to fall"]),
        make_turn(1, 0, "pos B", ["photosynthesis converts light to energy"]),
    ]
    diversity = ev.argument_diversity(turns)
    assert diversity > 0.0


# ---------------------------------------------------------------------------
# Test 11 — evaluate: returns DebateEvalResult
# ---------------------------------------------------------------------------


def test_evaluate_result_type():
    ev = DebateEvaluator()
    turns = [
        make_turn(0, 0, "yes", ["because a"]),
        make_turn(1, 0, "no", ["because b"]),
        make_turn(0, 1, "yes", ["still a"]),
        make_turn(1, 1, "yes", ["agree with a"]),
    ]
    result = ev.evaluate(turns)
    assert isinstance(result, DebateEvalResult)


# ---------------------------------------------------------------------------
# Test 12 — evaluate: overall in [0, 1]
# ---------------------------------------------------------------------------


def test_evaluate_overall_range():
    ev = DebateEvaluator()
    turns = [
        make_turn(0, 0, "position one", ["arg one"]),
        make_turn(1, 0, "position two", ["arg two"]),
        make_turn(0, 1, "position three", ["arg three"]),
        make_turn(1, 1, "position one", ["arg one"]),
    ]
    result = ev.evaluate(turns)
    assert 0.0 <= result.overall <= 1.0


# ---------------------------------------------------------------------------
# Test 13 — evaluate: round_convergence has correct length
# ---------------------------------------------------------------------------


def test_round_convergence_length():
    ev = DebateEvaluator()
    # 3 rounds (0, 1, 2)
    turns = [
        make_turn(0, 0, "a", ["x"]),
        make_turn(1, 0, "b", ["y"]),
        make_turn(0, 1, "a", ["x"]),
        make_turn(1, 1, "a", ["x"]),
        make_turn(0, 2, "a", ["x"]),
        make_turn(1, 2, "a", ["x"]),
    ]
    result = ev.evaluate(turns)
    # Data has rounds 0, 1, 2 → n_rounds = 3
    assert len(result.round_convergence) == 3


# ---------------------------------------------------------------------------
# Test 14 — aggregate: correct keys returned
# ---------------------------------------------------------------------------


def test_aggregate_keys():
    ev = DebateEvaluator()
    turns = [make_turn(0, 0, "pos", ["arg"])]
    r1 = ev.evaluate(turns)
    r2 = ev.evaluate(turns)
    agg = ev.aggregate([r1, r2])
    expected_keys = {"position_drift", "final_consensus", "argument_diversity", "overall"}
    assert set(agg.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test 15 — registry: DebateEvaluator registered under "multi_agent_debate"
# ---------------------------------------------------------------------------


def test_benchmark_registry():
    assert "multi_agent_debate" in BENCHMARK_REGISTRY
    assert BENCHMARK_REGISTRY["multi_agent_debate"] is DebateEvaluator


# ---------------------------------------------------------------------------
# Test 16 — evaluate_batch: returns list of same length
# ---------------------------------------------------------------------------


def test_evaluate_batch_length():
    ev = DebateEvaluator()
    debate_a = [make_turn(0, 0, "yes", ["arg a"]), make_turn(1, 0, "no", ["arg b"])]
    debate_b = [make_turn(0, 0, "maybe", ["arg c"])]
    results = ev.evaluate_batch([debate_a, debate_b])
    assert len(results) == 2
    for r in results:
        assert isinstance(r, DebateEvalResult)


# ---------------------------------------------------------------------------
# Integration test — 3 agents × 3 rounds with converging + drifting positions
# ---------------------------------------------------------------------------


def test_integration_3x3_debate():
    """
    3 agents, 3 rounds.
    - Agent 0: starts "the sky is blue", stays the same (no drift).
    - Agent 1: starts "the ground is red", then shifts to "the sky is blue" (drift).
    - Agent 2: starts "the ocean is vast", shifts to "the sky is blue" (drift).

    Final round: all three agents agree → high consensus.
    Arguments across agents vary → some diversity.
    """
    ev = DebateEvaluator(DebateConfig(n_agents=3, n_rounds=3))

    turns = [
        # Round 0 — differing positions
        make_turn(0, 0, "the sky is blue", ["light scatters at short wavelengths"]),
        make_turn(1, 0, "the ground is red", ["iron oxide in soil"]),
        make_turn(2, 0, "the ocean is vast", ["covers seventy percent of earth"]),
        # Round 1 — partial convergence
        make_turn(0, 1, "the sky is blue", ["light scatters at short wavelengths"]),
        make_turn(1, 1, "the sky is blue", ["i agree with agent zero"]),
        make_turn(2, 1, "the ocean is blue", ["both sky and ocean appear blue"]),
        # Round 2 — full convergence
        make_turn(0, 2, "the sky is blue", ["rayleigh scattering explains this"]),
        make_turn(1, 2, "the sky is blue", ["confirmed by physics"]),
        make_turn(2, 2, "the sky is blue", ["the ocean reflects sky color"]),
    ]

    result = ev.evaluate(turns)

    # Structural checks
    assert isinstance(result, DebateEvalResult)
    assert result.n_rounds == 3
    assert len(result.round_convergence) == 3

    # Range checks
    assert 0.0 <= result.position_drift <= 1.0
    assert 0.0 <= result.final_consensus <= 1.0
    assert 0.0 <= result.argument_diversity <= 1.0
    assert 0.0 <= result.overall <= 1.0

    # Semantic checks
    # Final round all agents say "the sky is blue" → consensus should be high
    assert result.final_consensus > 0.7

    # Agents 1 and 2 changed position → drift should be > 0
    assert result.position_drift > 0.0

    # Aggregate over two copies
    results = ev.evaluate_batch([turns, turns])
    agg = ev.aggregate(results)
    assert set(agg.keys()) == {"position_drift", "final_consensus", "argument_diversity", "overall"}
    # Aggregated overall should match individual (same debate twice)
    assert agg["overall"] == pytest.approx(result.overall, abs=1e-6)
