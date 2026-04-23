"""Tests for src/multiagent/consensus_engine.py"""
import pytest
from src.multiagent.consensus_engine import (
    ConsensusConfig,
    ConsensusEngine,
    ConsensusMethod,
    ConsensusResult,
    CONSENSUS_REGISTRY,
)


def test_majority_vote_picks_most_common():
    engine = ConsensusEngine()
    result = engine.aggregate(["yes", "no", "yes", "yes", "no"])
    assert result.winner == "yes"
    assert result.agreement == pytest.approx(3 / 5)
    assert result.method == ConsensusMethod.MAJORITY_VOTE.value


def test_majority_vote_agreement_perfect():
    engine = ConsensusEngine()
    result = engine.aggregate(["x", "x", "x"])
    assert result.winner == "x"
    assert result.agreement == pytest.approx(1.0)


def test_plurality_alias_same_as_majority():
    cfg = ConsensusConfig(method=ConsensusMethod.PLURALITY)
    engine = ConsensusEngine(cfg)
    result = engine.aggregate(["a", "b", "a"])
    assert result.winner == "a"
    assert result.method == ConsensusMethod.PLURALITY.value


def test_confidence_weighted_parses_format():
    cfg = ConsensusConfig(method=ConsensusMethod.CONFIDENCE_WEIGHTED)
    engine = ConsensusEngine(cfg)
    responses = [
        "CONFIDENCE:0.9:Paris",
        "CONFIDENCE:0.3:London",
        "CONFIDENCE:0.7:Paris",
    ]
    result = engine.aggregate(responses)
    assert result.winner == "Paris"
    assert result.method == ConsensusMethod.CONFIDENCE_WEIGHTED.value


def test_confidence_weighted_falls_back_to_weight_1():
    cfg = ConsensusConfig(method=ConsensusMethod.CONFIDENCE_WEIGHTED)
    engine = ConsensusEngine(cfg)
    result = engine.aggregate(["plain", "plain", "other"])
    assert result.winner == "plain"


def test_borda_count_first_rank_wins_on_unique():
    cfg = ConsensusConfig(method=ConsensusMethod.BORDA_COUNT)
    engine = ConsensusEngine(cfg)
    responses = ["alpha", "beta", "gamma"]
    result = engine.aggregate(responses)
    assert result.winner == "alpha"
    assert result.method == ConsensusMethod.BORDA_COUNT.value


def test_reached_consensus_true_above_threshold():
    engine = ConsensusEngine(ConsensusConfig(min_agreement=0.5))
    result = ConsensusResult(winner="yes", agreement=0.6, method="majority_vote")
    assert engine.reached_consensus(result) is True


def test_reached_consensus_false_below_threshold():
    engine = ConsensusEngine(ConsensusConfig(min_agreement=0.7))
    result = ConsensusResult(winner="yes", agreement=0.5, method="majority_vote")
    assert engine.reached_consensus(result) is False


def test_consensus_result_is_frozen():
    r = ConsensusResult(winner="x", agreement=1.0, method="majority_vote")
    with pytest.raises((AttributeError, TypeError)):
        r.winner = "y"  # type: ignore[misc]


def test_consensus_registry_has_default():
    assert "default" in CONSENSUS_REGISTRY
    assert CONSENSUS_REGISTRY["default"] is ConsensusEngine
