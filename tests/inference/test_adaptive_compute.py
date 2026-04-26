"""Tests for adaptive inference-time compute scaling.

Uses a lightweight MockModel: nn.Embedding(64, 16) + nn.Linear(16, 64)
returning (None, logits, None) to simulate a transformer-style interface.

vocab=64, d_model=16, T_prompt=4, max_new_tokens=5
"""

import pytest
import torch
import torch.nn as nn

from src.inference.adaptive_compute import (
    AdaptiveInferenceEngine,
    BestOfNSelector,
    ComputeBudget,
    ComputeBudgetPredictor,
    ConfidenceEstimator,
    InferenceResult,
    SequentialReviser,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOCAB = 64
D_MODEL = 16
T_PROMPT = 4
MAX_NEW_TOKENS = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockModel(nn.Module):
    """Tiny model: Embedding + Linear, returns (None, logits, None)."""

    def __init__(self, vocab_size: int = VOCAB, d_model: int = D_MODEL):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        # input_ids: (B, T)
        x = self.embedding(input_ids)  # (B, T, d_model)
        logits = self.lm_head(x)  # (B, T, vocab)
        return (None, logits, None)


@pytest.fixture
def mock_model():
    torch.manual_seed(42)
    return MockModel(vocab_size=VOCAB, d_model=D_MODEL)


@pytest.fixture
def prompt_ids():
    torch.manual_seed(0)
    return torch.randint(0, VOCAB, (1, T_PROMPT))


@pytest.fixture
def encode_fn():
    return lambda s: [ord(c) % VOCAB for c in s] if s else [0]


# ---------------------------------------------------------------------------
# ConfidenceEstimator tests (tests 1-4)
# ---------------------------------------------------------------------------


def test_confidence_max_prob_in_range():
    """Test 1: max_prob returns value in [0, 1]."""
    estimator = ConfidenceEstimator(method="max_prob")
    logits = torch.randn(VOCAB)
    confidence = estimator.estimate(logits)
    assert 0.0 <= confidence <= 1.0, f"Expected [0,1], got {confidence}"


def test_confidence_entropy_in_range():
    """Test 2: entropy method returns value in [0, 1]."""
    estimator = ConfidenceEstimator(method="entropy")
    logits = torch.randn(VOCAB)
    confidence = estimator.estimate(logits)
    assert 0.0 <= confidence <= 1.0, f"Expected [0,1], got {confidence}"


def test_confidence_margin_in_range():
    """Test 3: margin method returns value in [0, 1]."""
    estimator = ConfidenceEstimator(method="margin")
    logits = torch.randn(VOCAB)
    confidence = estimator.estimate(logits)
    assert 0.0 <= confidence <= 1.0, f"Expected [0,1], got {confidence}"


def test_confidence_estimate_sequence_in_range():
    """Test 4: estimate_sequence returns mean confidence in [0, 1]."""
    estimator = ConfidenceEstimator(method="max_prob")
    # Simulate T=5 steps of logits
    logits = torch.randn(5, VOCAB)
    confidence = estimator.estimate_sequence(logits)
    assert 0.0 <= confidence <= 1.0, f"Expected [0,1], got {confidence}"


def test_confidence_methods_differ():
    """Extra: different methods produce valid but potentially different results."""
    logits = torch.randn(VOCAB)
    for method in ("max_prob", "entropy", "margin"):
        c = ConfidenceEstimator(method=method).estimate(logits)
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# BestOfNSelector tests (tests 5-8)
# ---------------------------------------------------------------------------


def test_generate_one_returns_tuple(mock_model, prompt_ids):
    """Test 5: generate_one returns a (output_ids, confidence) tuple."""
    selector = BestOfNSelector(mock_model, temperature=1.0)
    result = selector.generate_one(prompt_ids, max_new_tokens=MAX_NEW_TOKENS)
    assert isinstance(result, tuple)
    assert len(result) == 2
    output_ids, confidence = result
    assert isinstance(output_ids, torch.Tensor)
    assert isinstance(confidence, float)


def test_generate_one_output_shape(mock_model, prompt_ids):
    """Test 6: output_ids shape includes prompt + generated tokens."""
    selector = BestOfNSelector(mock_model, temperature=1.0)
    output_ids, _ = selector.generate_one(prompt_ids, max_new_tokens=MAX_NEW_TOKENS)
    total_len = T_PROMPT + MAX_NEW_TOKENS
    # Accept either (1, total) or (total,) shape
    flat_len = output_ids.numel()
    assert flat_len == total_len, f"Expected {total_len} tokens total, got shape {output_ids.shape}"


def test_select_best_returns_inference_result(mock_model, prompt_ids):
    """Test 7: select_best returns an InferenceResult."""
    selector = BestOfNSelector(mock_model, temperature=1.0)
    result = selector.select_best(prompt_ids, n=3, max_new_tokens=MAX_NEW_TOKENS)
    assert isinstance(result, InferenceResult)


def test_select_best_n_tokens_used_positive(mock_model, prompt_ids):
    """Test 8: InferenceResult.n_tokens_used > 0."""
    selector = BestOfNSelector(mock_model, temperature=1.0)
    result = selector.select_best(prompt_ids, n=2, max_new_tokens=MAX_NEW_TOKENS)
    assert result.n_tokens_used > 0, f"n_tokens_used should be > 0, got {result.n_tokens_used}"


# ---------------------------------------------------------------------------
# SequentialReviser tests (tests 9-10)
# ---------------------------------------------------------------------------


def test_sequential_reviser_returns_inference_result(mock_model, prompt_ids, encode_fn):
    """Test 9: SequentialReviser.run returns an InferenceResult."""
    reviser = SequentialReviser(
        model=mock_model,
        encode_fn=encode_fn,
        max_iterations=3,
        confidence_threshold=0.85,
    )
    result = reviser.run(prompt_ids, max_new_tokens=MAX_NEW_TOKENS)
    assert isinstance(result, InferenceResult)


def test_sequential_reviser_iterations_bounded(mock_model, prompt_ids, encode_fn):
    """Test 10: InferenceResult.n_iterations <= max_iterations."""
    max_iter = 3
    reviser = SequentialReviser(
        model=mock_model,
        encode_fn=encode_fn,
        max_iterations=max_iter,
        confidence_threshold=0.85,
    )
    result = reviser.run(prompt_ids, max_new_tokens=MAX_NEW_TOKENS)
    assert result.n_iterations <= max_iter, (
        f"n_iterations {result.n_iterations} exceeded max {max_iter}"
    )


# ---------------------------------------------------------------------------
# ComputeBudgetPredictor tests (tests 11-13)
# ---------------------------------------------------------------------------


def test_budget_predictor_forward_shape():
    """Test 11: forward returns (B, 3) budget logits."""
    predictor = ComputeBudgetPredictor(vocab_size=VOCAB, d_model=D_MODEL)
    B, T = 3, T_PROMPT
    input_ids = torch.randint(0, VOCAB, (B, T))
    logits = predictor(input_ids)
    assert logits.shape == (B, 3), f"Expected (B=3, 3), got {logits.shape}"


def test_budget_predictor_predict_budget_valid_strings():
    """Test 12: predict_budget returns list of valid budget strings."""
    predictor = ComputeBudgetPredictor(vocab_size=VOCAB, d_model=D_MODEL)
    input_ids = torch.randint(0, VOCAB, (4, T_PROMPT))
    budgets = predictor.predict_budget(input_ids)
    assert isinstance(budgets, list)
    assert len(budgets) == 4
    valid = {"low", "medium", "high"}
    for b in budgets:
        assert b in valid, f"Invalid budget string: {b}"


def test_budget_predictor_high_budget_n_candidates():
    """Test 13: to_compute_budget('high').n_candidates > 1."""
    predictor = ComputeBudgetPredictor(vocab_size=VOCAB, d_model=D_MODEL)
    budget = predictor.to_compute_budget("high")
    assert budget.n_candidates > 1, (
        f"High budget should have n_candidates > 1, got {budget.n_candidates}"
    )


def test_budget_predictor_low_budget_n_candidates_one():
    """Extra: to_compute_budget('low').n_candidates == 1."""
    predictor = ComputeBudgetPredictor(vocab_size=VOCAB, d_model=D_MODEL)
    budget = predictor.to_compute_budget("low")
    assert budget.n_candidates == 1


def test_budget_predictor_medium_budget_n_candidates():
    """Extra: to_compute_budget('medium').n_candidates == 2."""
    predictor = ComputeBudgetPredictor(vocab_size=VOCAB, d_model=D_MODEL)
    budget = predictor.to_compute_budget("medium")
    assert budget.n_candidates == 2


# ---------------------------------------------------------------------------
# AdaptiveInferenceEngine tests (tests 14-16)
# ---------------------------------------------------------------------------


def test_engine_infer_returns_inference_result(mock_model, prompt_ids):
    """Test 14: AdaptiveInferenceEngine.infer returns InferenceResult."""
    engine = AdaptiveInferenceEngine(model=mock_model)
    result = engine.infer(prompt_ids)
    assert isinstance(result, InferenceResult)


def test_engine_batch_infer_returns_correct_length(mock_model):
    """Test 15: batch_infer returns list of same length as inputs."""
    engine = AdaptiveInferenceEngine(model=mock_model)
    inputs = [
        torch.randint(0, VOCAB, (1, T_PROMPT)),
        torch.randint(0, VOCAB, (1, T_PROMPT)),
        torch.randint(0, VOCAB, (1, T_PROMPT)),
    ]
    results = engine.batch_infer(inputs)
    assert isinstance(results, list)
    assert len(results) == len(inputs)
    for r in results:
        assert isinstance(r, InferenceResult)


def test_inference_result_confidence_in_range(mock_model, prompt_ids):
    """Test 16: InferenceResult.confidence in [0, 1]."""
    engine = AdaptiveInferenceEngine(model=mock_model)
    result = engine.infer(prompt_ids)
    assert 0.0 <= result.confidence <= 1.0, f"confidence {result.confidence} not in [0, 1]"


# ---------------------------------------------------------------------------
# Extra integration tests
# ---------------------------------------------------------------------------


def test_engine_with_budget_predictor(mock_model, prompt_ids):
    """Engine uses budget predictor when provided."""
    predictor = ComputeBudgetPredictor(vocab_size=VOCAB, d_model=D_MODEL)
    engine = AdaptiveInferenceEngine(model=mock_model, budget_predictor=predictor)
    result = engine.infer(prompt_ids)
    assert isinstance(result, InferenceResult)
    assert result.strategy_used in ("greedy", "best_of_2", "best_of_4", "best_of_1")


def test_engine_with_override_budget(mock_model, prompt_ids):
    """override_budget is respected."""
    low_budget = ComputeBudget(max_tokens=10, n_candidates=1)
    engine = AdaptiveInferenceEngine(model=mock_model)
    result = engine.infer(prompt_ids, override_budget=low_budget)
    assert isinstance(result, InferenceResult)
    assert result.strategy_used == "greedy"


def test_best_of_n_with_verifier(mock_model, prompt_ids):
    """BestOfNSelector uses verifier_fn when provided."""
    call_count = {"n": 0}

    def fake_verifier(output_ids: torch.Tensor) -> float:
        call_count["n"] += 1
        return float(output_ids.float().mean().item())

    selector = BestOfNSelector(mock_model, verifier_fn=fake_verifier, temperature=1.0)
    result = selector.select_best(prompt_ids, n=3, max_new_tokens=MAX_NEW_TOKENS)
    assert isinstance(result, InferenceResult)
    assert call_count["n"] == 3


def test_confidence_estimator_empty_sequence():
    """estimate_sequence handles empty tensor gracefully."""
    estimator = ConfidenceEstimator(method="max_prob")
    logits = torch.zeros(0, VOCAB)
    confidence = estimator.estimate_sequence(logits)
    assert confidence == 0.0
