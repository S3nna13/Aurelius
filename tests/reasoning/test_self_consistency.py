"""Tests for SelfConsistency."""
from __future__ import annotations

import pytest
from src.reasoning.self_consistency import (
    SelfConsistency,
    SelfConsistencyConfig,
    ConsistencyResult,
    SELF_CONSISTENCY_REGISTRY,
)


def test_aggregate_majority_vote_picks_most_common():
    sc = SelfConsistency(SelfConsistencyConfig(n_samples=5))
    samples = ["42", "42", "42", "7", "7"]
    result = sc.aggregate(samples)
    assert result.answer == "42"


def test_aggregate_majority_vote_confidence():
    sc = SelfConsistency(SelfConsistencyConfig(n_samples=5))
    samples = ["42", "42", "42", "7", "7"]
    result = sc.aggregate(samples)
    assert abs(result.confidence - 3 / 5) < 1e-9


def test_aggregate_vote_distribution():
    sc = SelfConsistency()
    samples = ["a", "b", "a"]
    result = sc.aggregate(samples)
    assert result.vote_distribution == {"a": 2, "b": 1}


def test_aggregate_n_samples_in_result():
    sc = SelfConsistency()
    samples = ["x"] * 6
    result = sc.aggregate(samples)
    assert result.n_samples == 6


def test_weighted_vote_parsing():
    sc = SelfConsistency(SelfConsistencyConfig(aggregation="weighted_vote"))
    samples = ["WEIGHT:0.9:yes", "WEIGHT:0.1:no"]
    result = sc.aggregate(samples)
    assert result.answer == "yes"


def test_weighted_vote_confidence():
    sc = SelfConsistency(SelfConsistencyConfig(aggregation="weighted_vote"))
    samples = ["WEIGHT:0.8:yes", "WEIGHT:0.2:no"]
    result = sc.aggregate(samples)
    assert abs(result.confidence - 0.8) < 1e-9


def test_extract_answer_hash_format():
    sc = SelfConsistency()
    assert sc.extract_answer("some reasoning #### 42") == "42"


def test_extract_answer_label_format():
    sc = SelfConsistency()
    assert sc.extract_answer("thinking...\nAnswer: Paris") == "Paris"


def test_extract_answer_equals_format():
    sc = SelfConsistency()
    assert sc.extract_answer("x = 7") == "7"


def test_extract_answer_the_answer_is():
    sc = SelfConsistency()
    assert sc.extract_answer("The answer is 99") == "99"


def test_extract_answer_fallback_last_line():
    sc = SelfConsistency()
    assert sc.extract_answer("line one\nfinal answer here") == "final answer here"


def test_run_calls_generate_fn_n_samples_times():
    call_count = 0

    def gen(q):
        nonlocal call_count
        call_count += 1
        return "#### 5"

    sc = SelfConsistency(SelfConsistencyConfig(n_samples=7))
    sc.run("what is 2+3?", gen)
    assert call_count == 7


def test_run_returns_consistency_result():
    sc = SelfConsistency(SelfConsistencyConfig(n_samples=3))
    result = sc.run("q", lambda q: "#### 5")
    assert isinstance(result, ConsistencyResult)
    assert result.answer == "5"


def test_registry_key():
    assert "default" in SELF_CONSISTENCY_REGISTRY
    assert SELF_CONSISTENCY_REGISTRY["default"] is SelfConsistency
