"""Tests for pairwise_evaluator."""

from __future__ import annotations

from src.evaluation.pairwise_evaluator import PairwiseEvaluator, PairwiseResult


class TestPairwiseResult:
    def test_winner(self):
        r = PairwiseResult(winner="A", score_a=0.8, score_b=0.3)
        assert r.winner == "A"


class TestPairwiseEvaluator:
    def test_a_wins(self):
        e = PairwiseEvaluator()
        r = e.compare("great answer", "bad answer")
        assert r.score_a > r.score_b

    def test_tie(self):
        e = PairwiseEvaluator()
        r = e.compare("same", "same")
        assert r.winner in ("A", "B", "tie")

    def test_empty(self):
        e = PairwiseEvaluator()
        r = e.compare("", "")
        assert r.score_a == r.score_b
