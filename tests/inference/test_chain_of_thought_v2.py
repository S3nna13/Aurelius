"""Tests for src/inference/chain_of_thought_v2.py (aurelius.inference.chain_of_thought_v2)."""

from __future__ import annotations

import math

import pytest
import torch
from aurelius.inference.chain_of_thought_v2 import (
    AnswerExtractor,
    CoTBudgetAllocator,
    CoTConfig,
    CoTScorer,
    SelfConsistencyCoT,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_config() -> CoTConfig:
    return CoTConfig()


@pytest.fixture()
def extractor() -> AnswerExtractor:
    return AnswerExtractor(trigger="The answer is", vocab_size=1000)


@pytest.fixture()
def scorer() -> CoTScorer:
    return CoTScorer(vocab_size=1000)


@pytest.fixture()
def sc(default_config: CoTConfig, extractor: AnswerExtractor) -> SelfConsistencyCoT:
    return SelfConsistencyCoT(config=default_config, extractor=extractor)


# ---------------------------------------------------------------------------
# 1. CoTConfig defaults
# ---------------------------------------------------------------------------


class TestCoTConfig:
    def test_defaults(self, default_config: CoTConfig) -> None:
        assert default_config.n_samples == 8
        assert default_config.temperature == pytest.approx(0.7)
        assert default_config.max_reasoning_tokens == 256
        assert default_config.answer_trigger == "The answer is"


# ---------------------------------------------------------------------------
# 2–4. AnswerExtractor
# ---------------------------------------------------------------------------


class TestAnswerExtractor:
    def _make_trigger(self) -> torch.LongTensor:
        # trigger token ids: [10, 20, 30]
        return torch.tensor([10, 20, 30], dtype=torch.long)

    def _make_sequence_with_trigger(self) -> torch.LongTensor:
        # Sequence: [1, 2, 10, 20, 30, 5, 6, 7]
        # trigger starts at index 2 → answer span = (5, 8)
        return torch.tensor([1, 2, 10, 20, 30, 5, 6, 7], dtype=torch.long)

    def _make_sequence_without_trigger(self) -> torch.LongTensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)

    # ---- 2. extract_span finds trigger correctly -------------------------
    def test_extract_span_found(self, extractor: AnswerExtractor) -> None:
        trig = self._make_trigger()
        seq = self._make_sequence_with_trigger()
        start, end = extractor.extract_span(seq, trig)
        assert start == 5, f"Expected answer start=5, got {start}"
        assert end == 8, f"Expected answer end=8, got {end}"
        # answer tokens are seq[5:8] = [5, 6, 7]
        assert seq[start:end].tolist() == [5, 6, 7]

    # ---- 3. extract_span returns (-1, -1) when trigger not found ----------
    def test_extract_span_not_found(self, extractor: AnswerExtractor) -> None:
        trig = self._make_trigger()
        seq = self._make_sequence_without_trigger()
        assert extractor.extract_span(seq, trig) == (-1, -1)

    # ---- 4. has_answer True / False --------------------------------------
    def test_has_answer_true(self, extractor: AnswerExtractor) -> None:
        trig = self._make_trigger()
        seq = self._make_sequence_with_trigger()
        assert extractor.has_answer(seq, trig) is True

    def test_has_answer_false(self, extractor: AnswerExtractor) -> None:
        trig = self._make_trigger()
        seq = self._make_sequence_without_trigger()
        assert extractor.has_answer(seq, trig) is False

    # ---- 5. extract_batch returns correct-length list --------------------
    def test_extract_batch_length(self, extractor: AnswerExtractor) -> None:
        trig = self._make_trigger()
        # batch of 4 identical sequences
        batch = torch.stack([self._make_sequence_with_trigger()] * 4)
        results = extractor.extract_batch(batch, trig)
        assert isinstance(results, list)
        assert len(results) == 4
        for span in results:
            assert span == (5, 8)

    def test_extract_batch_mixed(self, extractor: AnswerExtractor) -> None:
        trig = self._make_trigger()
        with_trigger = self._make_sequence_with_trigger()
        without_trigger = self._make_sequence_without_trigger()
        batch = torch.stack([with_trigger, without_trigger])
        results = extractor.extract_batch(batch, trig)
        assert results[0] == (5, 8)
        assert results[1] == (-1, -1)


# ---------------------------------------------------------------------------
# 6–8. CoTScorer
# ---------------------------------------------------------------------------


class TestCoTScorer:
    # ---- 6. score_reasoning_quality is scalar and finite -----------------
    def test_score_is_scalar_and_finite(self, scorer: CoTScorer) -> None:
        reasoning = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        log_probs = torch.tensor([-1.0, -0.5, -0.8, -1.2, -0.3])
        result = scorer.score_reasoning_quality(reasoning, log_probs)
        assert result.dim() == 0, "Expected scalar tensor"
        assert math.isfinite(result.item()), "Score should be finite"

    def test_score_equals_mean_log_probs(self, scorer: CoTScorer) -> None:
        reasoning = torch.zeros(3, dtype=torch.long)
        log_probs = torch.tensor([-1.0, -2.0, -3.0])
        result = scorer.score_reasoning_quality(reasoning, log_probs)
        assert result.item() == pytest.approx(-2.0)

    # ---- 7. length_penalty = 1.0 for target length -----------------------
    def test_length_penalty_ideal(self, scorer: CoTScorer) -> None:
        # target_length=128 → ideal range [64, 256]
        assert scorer.length_penalty(128, target_length=128) == pytest.approx(1.0)
        assert scorer.length_penalty(64, target_length=128) == pytest.approx(1.0)
        assert scorer.length_penalty(256, target_length=128) == pytest.approx(1.0)

    # ---- 8. length_penalty < 1.0 for too-short or too-long ---------------
    def test_length_penalty_too_short(self, scorer: CoTScorer) -> None:
        # length=10, target=128 → lo=64, so 10 < 64 → penalty < 1.0
        penalty = scorer.length_penalty(10, target_length=128)
        assert penalty < 1.0, f"Expected < 1.0, got {penalty}"
        assert penalty >= 0.0

    def test_length_penalty_too_long(self, scorer: CoTScorer) -> None:
        # length=1000, target=128 → hi=256, so 1000 > 256 → penalty < 1.0
        penalty = scorer.length_penalty(1000, target_length=128)
        assert penalty < 1.0, f"Expected < 1.0, got {penalty}"
        assert penalty >= 0.0

    # ---- 9. aggregate_scores returns correct keys ------------------------
    def test_aggregate_scores_keys(self, scorer: CoTScorer) -> None:
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scorer.aggregate_scores(scores)
        assert set(result.keys()) == {"mean", "max", "min", "std"}

    def test_aggregate_scores_values(self, scorer: CoTScorer) -> None:
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scorer.aggregate_scores(scores)
        assert result["mean"] == pytest.approx(3.0)
        assert result["max"] == pytest.approx(5.0)
        assert result["min"] == pytest.approx(1.0)
        assert result["std"] > 0.0


# ---------------------------------------------------------------------------
# 10–12. SelfConsistencyCoT
# ---------------------------------------------------------------------------


class TestSelfConsistencyCoT:
    def _make_spans(self) -> list[torch.LongTensor]:
        a = torch.tensor([42, 7], dtype=torch.long)
        b = torch.tensor([42, 7], dtype=torch.long)
        c = torch.tensor([99, 1], dtype=torch.long)
        return [a, b, c]

    # ---- 10. vote finds majority -----------------------------------------
    def test_vote_majority(self, sc: SelfConsistencyCoT) -> None:
        spans = self._make_spans()
        winner, count = sc.vote(spans)
        assert winner.tolist() == [42, 7], f"Wrong winner: {winner.tolist()}"
        assert count == 2

    def test_vote_single(self, sc: SelfConsistencyCoT) -> None:
        spans = [torch.tensor([5, 6, 7], dtype=torch.long)]
        winner, count = sc.vote(spans)
        assert winner.tolist() == [5, 6, 7]
        assert count == 1

    # ---- 11. rank_by_quality returns descending order -------------------
    def test_rank_by_quality_descending(self, sc: SelfConsistencyCoT) -> None:
        scores = torch.tensor([0.3, 0.9, 0.1, 0.7])
        ranked = sc.rank_by_quality(scores)
        # Should be [1, 3, 0, 2]
        assert ranked.tolist() == [1, 3, 0, 2]

    def test_rank_by_quality_shape(self, sc: SelfConsistencyCoT) -> None:
        scores = torch.tensor([0.1, 0.5, 0.9])
        ranked = sc.rank_by_quality(scores)
        assert ranked.shape[0] == 3

    # ---- 12. select_best returns highest-scored candidate ---------------
    def test_select_best(self, sc: SelfConsistencyCoT) -> None:
        candidates = [
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([3, 4], dtype=torch.long),
            torch.tensor([5, 6], dtype=torch.long),
        ]
        scores = torch.tensor([0.2, 0.9, 0.5])
        best = sc.select_best(candidates, scores)
        assert best.tolist() == [3, 4], f"Expected [3,4], got {best.tolist()}"


# ---------------------------------------------------------------------------
# 13. CoTBudgetAllocator
# ---------------------------------------------------------------------------


class TestCoTBudgetAllocator:
    # ---- 13. allocate respects budget ------------------------------------
    def test_allocate_within_budget(self) -> None:
        alloc = CoTBudgetAllocator(total_token_budget=1000)
        result = alloc.allocate(n_samples=4, max_reasoning_tokens=200)
        assert result["total_tokens"] <= 1000
        assert result["n_samples"] == 4
        assert result["tokens_per_sample"] == 200
        assert result["total_tokens"] == 800

    def test_allocate_reduces_samples_when_over_budget(self) -> None:
        alloc = CoTBudgetAllocator(total_token_budget=500)
        result = alloc.allocate(n_samples=10, max_reasoning_tokens=200)
        assert result["total_tokens"] <= 500
        assert result["n_samples"] < 10
        assert result["n_samples"] == 2

    def test_allocate_returns_required_keys(self) -> None:
        alloc = CoTBudgetAllocator(total_token_budget=2048)
        result = alloc.allocate(n_samples=8, max_reasoning_tokens=256)
        assert set(result.keys()) == {"tokens_per_sample", "n_samples", "total_tokens"}

    def test_remaining(self) -> None:
        alloc = CoTBudgetAllocator(total_token_budget=1000)
        assert alloc.remaining(300) == 700
        assert alloc.remaining(1000) == 0
        assert alloc.remaining(1500) == 0  # clamped to 0
