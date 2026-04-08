"""Tests for evidence overlap helpers."""

import pytest

from src.eval.evidence_overlap import evidence_jaccard, max_evidence_overlap, mean_evidence_overlap


def test_evidence_jaccard_one_for_identical_texts():
    assert evidence_jaccard("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_evidence_jaccard_zero_for_disjoint_texts():
    assert evidence_jaccard("cat", "dog") == pytest.approx(0.0)


def test_mean_evidence_overlap_averages_pairs():
    score = mean_evidence_overlap(["cat sat", "cat sat", "dog barked"])
    assert 0.0 < score < 1.0


def test_mean_evidence_overlap_handles_single_passage():
    assert mean_evidence_overlap(["one"]) == pytest.approx(1.0)


def test_max_evidence_overlap_selects_best_match():
    score = max_evidence_overlap("cat sat", ["dog barked", "cat sat here"])
    assert score > 0.5


def test_max_evidence_overlap_handles_empty_passages():
    assert max_evidence_overlap("query", []) == pytest.approx(0.0)


def test_evidence_jaccard_handles_empty_texts():
    assert evidence_jaccard("", "") == pytest.approx(1.0)

