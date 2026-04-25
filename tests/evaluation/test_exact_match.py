"""Tests for exact match scorer."""
from __future__ import annotations

import pytest

from src.evaluation.exact_match import ExactMatchScorer


class TestExactMatchScorer:
    def test_exact_match(self):
        assert ExactMatchScorer().score("hello", "hello") == 1.0

    def test_case_insensitive(self):
        scorer = ExactMatchScorer(case_sensitive=False)
        assert scorer.score("Hello", "hello") == 1.0

    def test_case_sensitive(self):
        scorer = ExactMatchScorer(case_sensitive=True)
        assert scorer.score("Hello", "hello") == 0.0

    def test_no_match(self):
        assert ExactMatchScorer().score("abc", "xyz") == 0.0

    def test_batch(self):
        scorer = ExactMatchScorer()
        scores = scorer.score_batch(["a", "b"], ["a", "c"])
        assert scores == [1.0, 0.0]