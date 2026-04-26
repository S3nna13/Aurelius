"""Tests for Universal Self-Consistency (Chen et al., arXiv:2311.17311).

Covers SelfConsistencyVoter, UniversalSCScorer, TemperatureSampler, and
SelfConsistencyDecoder — approximately 14 test cases.
"""

from __future__ import annotations

import pytest
import torch
from aurelius.inference.universal_self_consistency import (
    SelfConsistencyDecoder,
    SelfConsistencyVoter,
    TemperatureSampler,
    UniversalSCScorer,
)
from torch import Tensor

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_model_fn(fixed_output: list[int] | None = None):
    """Return a deterministic model_fn that always yields the same token list."""
    tokens = fixed_output if fixed_output is not None else [10, 20, 30]

    def model_fn(prompt_ids: Tensor, temperature: float) -> Tensor:
        return torch.tensor(tokens, dtype=torch.long)

    return model_fn


def _make_cycling_model_fn(outputs: list[list[int]]):
    """Return a model_fn that cycles through given token lists."""
    call_count = [0]

    def model_fn(prompt_ids: Tensor, temperature: float) -> Tensor:
        idx = call_count[0] % len(outputs)
        call_count[0] += 1
        return torch.tensor(outputs[idx], dtype=torch.long)

    return model_fn


PROMPT = torch.tensor([1, 2, 3], dtype=torch.long)


# ===========================================================================
# SelfConsistencyVoter
# ===========================================================================


class TestSelfConsistencyVoter:
    # Test 1 — basic majority vote
    def test_vote_returns_majority_answer(self):
        voter = SelfConsistencyVoter()
        winner, counts = voter.vote(["cat", "cat", "dog"])
        assert winner == "cat"
        assert counts["cat"] == 2
        assert counts["dog"] == 1

    # Test 2 — all-same answers → confidence = 1.0
    def test_vote_all_same_confidence_is_one(self):
        voter = SelfConsistencyVoter()
        answer, confidence = voter.vote_with_confidence(["yes", "yes", "yes"])
        assert answer == "yes"
        assert confidence == pytest.approx(1.0)

    # Test 3 — diverse answers → correct counts
    def test_vote_diverse_counts(self):
        voter = SelfConsistencyVoter()
        answers = ["a", "b", "c", "b", "a", "b"]
        winner, counts = voter.vote(answers)
        assert winner == "b"
        assert counts["b"] == 3
        assert counts["a"] == 2
        assert counts["c"] == 1

    # Test 4 — vote_with_confidence returns float in [0, 1]
    def test_vote_with_confidence_in_range(self):
        voter = SelfConsistencyVoter()
        for answers in [
            ["x"],
            ["x", "y"],
            ["x", "x", "y", "z"],
        ]:
            _, conf = voter.vote_with_confidence(answers)
            assert 0.0 < conf <= 1.0

    # Test 5 — empty list raises ValueError
    def test_vote_empty_raises(self):
        voter = SelfConsistencyVoter()
        with pytest.raises(ValueError):
            voter.vote([])
        with pytest.raises(ValueError):
            voter.vote_with_confidence([])

    # Bonus — custom answer_extractor is applied before comparison
    def test_vote_with_custom_extractor(self):
        def extractor(s):
            return s.strip().lower()

        voter = SelfConsistencyVoter(answer_extractor=extractor)
        winner, counts = voter.vote(["  YES ", "Yes", "no"])
        # Both "  YES " and "Yes" normalise to "yes"
        assert extractor(winner) == "yes"
        assert counts["yes"] == 2


# ===========================================================================
# UniversalSCScorer
# ===========================================================================


class TestUniversalSCScorer:
    # Test 6 — score output has correct shape (N,)
    def test_score_output_shape(self):
        scorer = UniversalSCScorer()
        answers = ["a", "b", "c", "d"]
        scores = scorer.score(answers)
        assert scores.shape == (len(answers),)

    # Test 7 — all-same answers → all scores = 1.0
    def test_score_all_same_is_one(self):
        scorer = UniversalSCScorer()
        scores = scorer.score(["yes", "yes", "yes", "yes"])
        assert torch.allclose(scores, torch.ones(4))

    # Test 8 — all-different answers → all scores = 0.0
    def test_score_all_different_is_zero(self):
        scorer = UniversalSCScorer()
        scores = scorer.score(["a", "b", "c", "d"])
        assert torch.allclose(scores, torch.zeros(4))

    # Test 9 — select_best returns answer with highest score
    def test_select_best_returns_highest_scored(self):
        scorer = UniversalSCScorer()
        # "b" appears 3 times, so it should have the highest score
        answers = ["a", "b", "b", "b", "c"]
        scores = scorer.score(answers)
        best = scorer.select_best(answers, scores)
        assert best == "b"

    # Bonus — single answer
    def test_score_single_answer(self):
        scorer = UniversalSCScorer()
        scores = scorer.score(["only"])
        assert scores.shape == (1,)
        assert float(scores[0]) == pytest.approx(0.0)

    # Bonus — empty list
    def test_score_empty(self):
        scorer = UniversalSCScorer()
        scores = scorer.score([])
        assert scores.shape == (0,)


# ===========================================================================
# TemperatureSampler
# ===========================================================================


class TestTemperatureSampler:
    def test_sample_n_length(self):
        """sample_n returns exactly n samples."""
        calls = []

        def model_fn(input_ids, temperature):
            calls.append(temperature)
            return 42

        sampler = TemperatureSampler(model_fn=model_fn, temperatures=[0.7, 1.0])
        result = sampler.sample_n(PROMPT, n=5)
        assert len(result) == 5

    def test_temperature_cycling(self):
        """Temperatures cycle through the provided list."""
        used_temps = []

        def model_fn(input_ids, temperature):
            used_temps.append(temperature)
            return 0

        sampler = TemperatureSampler(model_fn=model_fn, temperatures=[0.5, 1.0, 1.5])
        sampler.sample_n(PROMPT, n=6)
        assert used_temps == [0.5, 1.0, 1.5, 0.5, 1.0, 1.5]

    def test_returns_int_list(self):
        sampler = TemperatureSampler(model_fn=lambda ids, t: 7, temperatures=[1.0])
        result = sampler.sample_n(PROMPT, n=3)
        assert all(isinstance(x, int) for x in result)


# ===========================================================================
# SelfConsistencyDecoder
# ===========================================================================


class TestSelfConsistencyDecoder:
    # Test 10 — generate returns (LongTensor, dict)
    def test_generate_return_types(self):
        decoder = SelfConsistencyDecoder(
            model_fn=_make_model_fn([1, 2, 3]),
            n_samples=4,
            use_universal=True,
        )
        best, stats = decoder.generate(PROMPT)
        assert isinstance(best, Tensor)
        assert best.dtype == torch.long
        assert isinstance(stats, dict)

    # Test 11 — stats has required keys
    def test_stats_has_required_keys(self):
        decoder = SelfConsistencyDecoder(
            model_fn=_make_model_fn([5, 6]),
            n_samples=3,
            use_universal=True,
        )
        _, stats = decoder.generate(PROMPT)
        assert "n_samples" in stats
        assert "confidence" in stats
        assert "all_completions" in stats

    # Test 12 — n_samples completions generated
    def test_n_samples_completions_generated(self):
        n = 7
        decoder = SelfConsistencyDecoder(
            model_fn=_make_model_fn([10]),
            n_samples=n,
            use_universal=True,
        )
        _, stats = decoder.generate(PROMPT)
        assert stats["n_samples"] == n
        assert len(stats["all_completions"]) == n

    # Test 13 — use_universal=False falls back to majority vote
    def test_use_universal_false_falls_back_to_majority_vote(self):
        # Two distinct outputs: [1,2,3] x5 and [9,9,9] x2 → majority is [1,2,3]
        outputs = [[1, 2, 3]] * 5 + [[9, 9, 9]] * 2
        decoder = SelfConsistencyDecoder(
            model_fn=_make_cycling_model_fn(outputs),
            n_samples=7,
            use_universal=False,
        )
        best, stats = decoder.generate(PROMPT)
        assert best.tolist() == [1, 2, 3]
        # confidence should reflect 5/7
        assert stats["confidence"] == pytest.approx(5 / 7)

    # Test 14 — confidence in [0, 1]
    def test_confidence_in_range(self):
        decoder = SelfConsistencyDecoder(
            model_fn=_make_model_fn([3, 3, 3]),
            n_samples=5,
            use_universal=True,
        )
        _, stats = decoder.generate(PROMPT)
        assert 0.0 <= stats["confidence"] <= 1.0

    # Bonus — all_completions are LongTensors
    def test_all_completions_are_long_tensors(self):
        decoder = SelfConsistencyDecoder(
            model_fn=_make_model_fn([1, 2]),
            n_samples=4,
            use_universal=False,
        )
        _, stats = decoder.generate(PROMPT)
        for comp in stats["all_completions"]:
            assert isinstance(comp, Tensor)
            assert comp.dtype == torch.long
