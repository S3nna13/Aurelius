"""Tests for judge consistency metrics."""

import pytest

from src.eval.judge_consistency import (
    consensus_rate,
    majority_label,
    majority_vote,
    pairwise_agreement,
)


def test_pairwise_agreement_measures_matches():
    score = pairwise_agreement(["a", "b", "c"], ["a", "x", "c"])
    assert score == pytest.approx(2.0 / 3.0)


def test_pairwise_agreement_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        pairwise_agreement(["a"], ["a", "b"])


def test_majority_label_returns_most_common():
    assert majority_label(["pass", "pass", "fail"]) == "pass"


def test_majority_label_rejects_empty_input():
    with pytest.raises(ValueError):
        majority_label([])


def test_consensus_rate_detects_full_agreement_itemwise():
    outputs = [["a", "b"], ["a", "b"], ["a", "c"]]
    assert consensus_rate(outputs) == pytest.approx(0.5)


def test_majority_vote_aggregates_per_item():
    outputs = [["pass", "fail"], ["pass", "pass"], ["pass", "fail"]]
    assert majority_vote(outputs) == ["pass", "fail"]


def test_majority_vote_rejects_length_mismatch():
    with pytest.raises(ValueError):
        majority_vote([["a"], ["a", "b"]])
