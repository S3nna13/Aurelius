"""Tests for src/multiagent/parallel_consensus.py"""
from __future__ import annotations

import asyncio

import pytest

from src.multiagent.parallel_consensus import (
    AdjudicationStrategy,
    ConsensusResult,
    ParallelConsensusEngine,
    ProviderResponse,
    _hash_confidence,
    _hash_latency,
    _hash_response,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_response(provider: str = "local", response: str = "hello", confidence: float = 0.8) -> ProviderResponse:
    return ProviderResponse(provider=provider, response=response, latency_ms=20.0, confidence=confidence)


# ---------------------------------------------------------------------------
# ProviderResponse dataclass
# ---------------------------------------------------------------------------

def test_provider_response_fields():
    pr = ProviderResponse(provider="openai", response="hi", latency_ms=42.5, confidence=0.9)
    assert pr.provider == "openai"
    assert pr.response == "hi"
    assert pr.latency_ms == 42.5
    assert pr.confidence == 0.9


def test_hash_confidence_range():
    for p in ["local", "openai", "anthropic"]:
        c = _hash_confidence(p, "test prompt")
        assert 0.0 <= c <= 1.0


def test_hash_confidence_deterministic():
    c1 = _hash_confidence("openai", "hello world")
    c2 = _hash_confidence("openai", "hello world")
    assert c1 == c2


def test_hash_latency_range():
    lat = _hash_latency("local", "my prompt")
    assert 10.0 <= lat <= 100.0


def test_hash_response_deterministic():
    r1 = _hash_response("anthropic", "foo")
    r2 = _hash_response("anthropic", "foo")
    assert r1 == r2


def test_hash_response_contains_provider():
    r = _hash_response("openai", "bar")
    assert "openai" in r


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------

def test_default_providers():
    engine = ParallelConsensusEngine()
    assert set(engine._providers) == {"local", "openai", "anthropic"}


def test_custom_providers():
    engine = ParallelConsensusEngine(providers=["p1", "p2"])
    assert engine._providers == ["p1", "p2"]


def test_add_provider():
    engine = ParallelConsensusEngine(providers=["a"])
    engine.add_provider("b")
    assert "b" in engine._providers


def test_add_provider_no_duplicate():
    engine = ParallelConsensusEngine(providers=["a"])
    engine.add_provider("a")
    assert engine._providers.count("a") == 1


def test_remove_provider():
    engine = ParallelConsensusEngine(providers=["a", "b", "c"])
    engine.remove_provider("b")
    assert "b" not in engine._providers
    assert "a" in engine._providers


def test_remove_nonexistent_provider_no_error():
    engine = ParallelConsensusEngine(providers=["a"])
    engine.remove_provider("z")
    assert engine._providers == ["a"]


# ---------------------------------------------------------------------------
# query_provider (async, deterministic stub)
# ---------------------------------------------------------------------------

def test_query_provider_returns_response():
    engine = ParallelConsensusEngine(providers=["local"])
    pr = asyncio.run(engine.query_provider("local", "test"))
    assert isinstance(pr, ProviderResponse)
    assert pr.provider == "local"
    assert 10.0 <= pr.latency_ms <= 100.0
    assert 0.0 <= pr.confidence <= 1.0


def test_query_provider_deterministic():
    engine = ParallelConsensusEngine(providers=["local"])
    pr1 = asyncio.run(engine.query_provider("local", "same prompt"))
    pr2 = asyncio.run(engine.query_provider("local", "same prompt"))
    assert pr1.response == pr2.response
    assert pr1.confidence == pr2.confidence


# ---------------------------------------------------------------------------
# gather_responses
# ---------------------------------------------------------------------------

def test_gather_responses_count():
    engine = ParallelConsensusEngine(providers=["a", "b", "c"])
    responses = asyncio.run(engine.gather_responses("prompt"))
    assert len(responses) == 3


def test_gather_responses_all_providers_represented():
    engine = ParallelConsensusEngine(providers=["local", "openai"])
    responses = asyncio.run(engine.gather_responses("hello"))
    providers = {r.provider for r in responses}
    assert providers == {"local", "openai"}


# ---------------------------------------------------------------------------
# adjudicate — each strategy
# ---------------------------------------------------------------------------

def test_adjudicate_highest_confidence():
    engine = ParallelConsensusEngine(strategy=AdjudicationStrategy.HIGHEST_CONFIDENCE)
    responses = [
        _make_response("a", "resp-a", 0.3),
        _make_response("b", "resp-b", 0.9),
        _make_response("c", "resp-c", 0.5),
    ]
    result = engine.adjudicate(responses)
    assert result.winner.provider == "b"
    assert result.strategy == AdjudicationStrategy.HIGHEST_CONFIDENCE


def test_adjudicate_llm_picks_longest():
    engine = ParallelConsensusEngine(strategy=AdjudicationStrategy.LLM_ADJUDICATOR)
    responses = [
        _make_response("a", "short", 0.5),
        _make_response("b", "much longer response here", 0.3),
        _make_response("c", "medium response", 0.7),
    ]
    result = engine.adjudicate(responses)
    assert result.winner.provider == "b"


def test_adjudicate_majority_picks_most_common():
    engine = ParallelConsensusEngine(strategy=AdjudicationStrategy.MAJORITY)
    responses = [
        _make_response("a", "apple", 0.5),
        _make_response("b", "apple", 0.5),
        _make_response("c", "banana", 0.9),
    ]
    result = engine.adjudicate(responses)
    assert result.winner.response == "apple"


def test_adjudicate_weighted_picks_highest_confidence():
    engine = ParallelConsensusEngine(strategy=AdjudicationStrategy.WEIGHTED)
    responses = [
        _make_response("a", "x", 0.2),
        _make_response("b", "y", 0.95),
    ]
    result = engine.adjudicate(responses)
    assert result.winner.provider == "b"


def test_adjudicate_agreement_score_all_same():
    engine = ParallelConsensusEngine(strategy=AdjudicationStrategy.MAJORITY)
    responses = [_make_response("a", "same"), _make_response("b", "same"), _make_response("c", "same")]
    result = engine.adjudicate(responses)
    assert result.agreement_score == pytest.approx(1.0)


def test_adjudicate_agreement_score_none_agree():
    engine = ParallelConsensusEngine(strategy=AdjudicationStrategy.HIGHEST_CONFIDENCE)
    responses = [
        _make_response("a", "r1", 0.9),
        _make_response("b", "r2", 0.1),
        _make_response("c", "r3", 0.1),
    ]
    result = engine.adjudicate(responses)
    assert result.agreement_score == pytest.approx(1 / 3)


def test_adjudicate_empty_raises():
    engine = ParallelConsensusEngine()
    with pytest.raises(ValueError):
        engine.adjudicate([])


def test_adjudicate_returns_consensus_result():
    engine = ParallelConsensusEngine()
    responses = [_make_response()]
    result = engine.adjudicate(responses)
    assert isinstance(result, ConsensusResult)
    assert result.all_responses == responses


# ---------------------------------------------------------------------------
# consensus (end-to-end async)
# ---------------------------------------------------------------------------

def test_consensus_end_to_end():
    engine = ParallelConsensusEngine(providers=["local", "openai"])
    result = asyncio.run(engine.consensus("end-to-end test"))
    assert isinstance(result, ConsensusResult)
    assert result.winner in result.all_responses
    assert 0.0 <= result.agreement_score <= 1.0
