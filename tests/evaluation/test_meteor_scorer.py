"""Tests for meteor_scorer — METEOR evaluation metric."""

from __future__ import annotations

import pytest

from src.evaluation.meteor_scorer import meteor_score


class TestMeteorScore:
    def test_exact_match(self):
        ref = "the cat sat on the mat"
        hyp = "the cat sat on the mat"
        assert meteor_score([ref], hyp) == pytest.approx(1.0, abs=0.01)

    def test_partial_match(self):
        ref = "the cat sat on the mat"
        hyp = "a cat sat on a mat"
        score = meteor_score([ref], hyp)
        assert 0.0 < score < 1.0

    def test_no_match(self):
        ref = "the cat sat on the mat"
        hyp = "dog dog dog"
        assert meteor_score([ref], hyp) == pytest.approx(0.0, abs=0.01)

    def test_empty_hypothesis(self):
        assert meteor_score(["hello world"], "") == 0.0

    def test_empty_reference(self):
        assert meteor_score([""], "hello") == 0.0

    def test_multiple_references_best_selected(self):
        refs = ["the cat", "a dog"]
        hyp = "a dog"
        score = meteor_score(refs, hyp)
        assert score > 0.5

    def test_penalty_for_longer_hypothesis(self):
        ref = "short"
        hyp = "short extra words here that do not match"
        score = meteor_score([ref], hyp)
        assert score < 1.0

    def test_none_hypothesis_returns_zero(self):
        assert meteor_score(["ref"], None) == 0.0

    def test_empty_references_list(self):
        assert meteor_score([], "test") == 0.0
