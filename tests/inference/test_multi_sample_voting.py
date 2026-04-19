"""Unit tests for ``src.inference.multi_sample_voting``."""

from __future__ import annotations

import re

import pytest

from src.inference.multi_sample_voting import MultiSampleVoter, VoteResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def last_integer(text: str) -> str:
    matches = re.findall(r"-?\d+", text)
    return matches[-1] if matches else ""


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------
def test_majority_vote_simple():
    voter = MultiSampleVoter(strategy="majority")
    result = voter.vote(["yes", "no", "yes"])
    assert isinstance(result, VoteResult)
    assert result.selected == "yes"
    assert result.strategy == "majority"
    assert result.votes["yes"] == 2.0
    assert result.votes["no"] == 1.0


def test_weighted_vote_minority_wins_with_weights():
    voter = MultiSampleVoter(strategy="weighted")
    # "no" is the minority but has a huge weight.
    result = voter.vote(["yes", "no", "yes"], weights=[0.1, 10.0, 0.1])
    assert result.selected == "no"
    assert result.strategy == "weighted"
    assert result.votes["no"] == pytest.approx(10.0)
    assert result.votes["yes"] == pytest.approx(0.2)


def test_usc_picks_medoid_sample():
    # Three "A-like" samples and one outlier; medoid should be one of the A's.
    samples = [
        "alpha beta gamma",
        "alpha beta delta",
        "alpha beta epsilon",
        "zzzzzzzz qqqq rrrr",
    ]
    voter = MultiSampleVoter(strategy="usc")
    result = voter.vote(samples)
    assert result.strategy == "usc"
    # Outlier must not be selected.
    assert result.selected != "zzzzzzzz qqqq rrrr"
    assert result.selected in samples[:3]


def test_answer_extractor_last_integer():
    voter = MultiSampleVoter(answer_extractor=last_integer, strategy="majority")
    samples = [
        "after lots of work the answer is 42",
        "I compute step by step and get 42 in the end",
        "final answer: 7",
    ]
    result = voter.vote(samples)
    assert result.selected == "42"
    assert result.votes["42"] == 2.0
    assert result.votes["7"] == 1.0


def test_ties_broken_by_first_seen():
    voter = MultiSampleVoter(strategy="majority")
    # "a" appears first, tied 1-1 with "b". "a" wins.
    result = voter.vote(["a", "b"])
    assert result.selected == "a"
    # Reversed order: "b" first.
    result2 = voter.vote(["b", "a"])
    assert result2.selected == "b"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        MultiSampleVoter(strategy="bogus")


def test_empty_samples_raises():
    voter = MultiSampleVoter()
    with pytest.raises(ValueError):
        voter.vote([])


def test_weights_length_mismatch_raises():
    voter = MultiSampleVoter(strategy="weighted")
    with pytest.raises(ValueError):
        voter.vote(["a", "b"], weights=[1.0])


def test_negative_weights_raise():
    voter = MultiSampleVoter(strategy="weighted")
    with pytest.raises(ValueError):
        voter.vote(["a", "b"], weights=[1.0, -0.5])


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------
def test_confidence_matches_proportion():
    voter = MultiSampleVoter(strategy="majority")
    result = voter.vote(["x", "x", "x", "y"])
    assert result.selected == "x"
    assert result.confidence == pytest.approx(3.0 / 4.0)


# ---------------------------------------------------------------------------
# Custom similarity
# ---------------------------------------------------------------------------
def test_custom_similarity_fn():
    # Exact equality similarity: 1.0 if equal else 0.0.
    def eq_sim(a: str, b: str) -> float:
        return 1.0 if a == b else 0.0

    voter = MultiSampleVoter(strategy="usc", similarity_fn=eq_sim)
    # "foo" appears twice, so foo has mean sim 0.5 (one match, one miss).
    # "bar" and "baz" both have mean sim 0.0.
    samples = ["foo", "foo", "bar", "baz"]
    result = voter.vote(samples)
    assert result.selected == "foo"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_determinism_majority():
    voter = MultiSampleVoter(strategy="majority")
    samples = ["a", "b", "a", "c", "b", "a"]
    r1 = voter.vote(samples)
    r2 = voter.vote(samples)
    assert r1.selected == r2.selected
    assert r1.votes == r2.votes
    assert r1.confidence == r2.confidence


def test_determinism_usc():
    voter = MultiSampleVoter(strategy="usc")
    samples = ["hello world", "hello there", "goodbye world", "hello world!"]
    r1 = voter.vote(samples)
    r2 = voter.vote(samples)
    assert r1.selected == r2.selected
    assert r1.votes == r2.votes
    assert r1.confidence == r2.confidence


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_all_identical_samples():
    voter = MultiSampleVoter(strategy="majority")
    result = voter.vote(["same", "same", "same"])
    assert result.selected == "same"
    assert result.confidence == pytest.approx(1.0)


def test_all_identical_usc():
    voter = MultiSampleVoter(strategy="usc")
    result = voter.vote(["same", "same", "same"])
    assert result.selected == "same"
    assert result.confidence == pytest.approx(1.0)


def test_single_sample_returns_itself():
    voter = MultiSampleVoter(strategy="majority")
    result = voter.vote(["only"])
    assert result.selected == "only"
    assert result.confidence == pytest.approx(1.0)


def test_single_sample_usc():
    voter = MultiSampleVoter(strategy="usc")
    result = voter.vote(["only"])
    assert result.selected == "only"
    assert result.confidence == pytest.approx(1.0)


def test_weighted_without_weights_falls_back_to_unit():
    voter = MultiSampleVoter(strategy="weighted")
    result = voter.vote(["a", "a", "b"])
    assert result.selected == "a"
    assert result.votes["a"] == pytest.approx(2.0)
    assert result.votes["b"] == pytest.approx(1.0)
