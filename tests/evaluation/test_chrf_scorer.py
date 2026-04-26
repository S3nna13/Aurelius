"""Tests for chrf_scorer — chrF evaluation metric."""

from __future__ import annotations

import pytest

from src.evaluation.chrf_scorer import chrf_score


class TestChrfScore:
    def test_exact_match(self):
        ref = "the cat sat on the mat"
        hyp = "the cat sat on the mat"
        assert chrf_score([ref], hyp) == pytest.approx(100.0, abs=0.5)

    def test_no_match(self):
        ref = "the cat sat on the mat"
        hyp = "xxxxx yyyyy zzzzz aaaaa bbbbb"
        assert chrf_score([ref], hyp) < 30.0

    def test_empty_hypothesis(self):
        assert chrf_score(["hello world"], "") == 0.0

    def test_empty_reference(self):
        assert chrf_score([""], "hello") == 0.0

    def test_partial_match(self):
        ref = "the cat sat"
        hyp = "the dog sat"
        score = chrf_score([ref], hyp)
        assert 0.0 < score < 100.0

    def test_multiple_references(self):
        refs = ["cat dog", "bird fish"]
        hyp = "cat dog"
        score = chrf_score(refs, hyp)
        assert score > 50.0

    def test_none_hypothesis(self):
        assert chrf_score(["ref"], None) == 0.0

    def test_empty_references_list(self):
        assert chrf_score([], "test") == 0.0
