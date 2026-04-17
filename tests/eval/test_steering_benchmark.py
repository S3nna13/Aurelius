"""Tests for src/eval/steering_benchmark.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from src.eval.steering_benchmark import (
    SteeringTarget,
    SteeringResult,
    SteeringEvaluator,
    SteeringComparison,
    SteeringVectorNormalizer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 32
SEQ_LEN = 8


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_linear_model_fn(d_model: int, vocab_size: int):
    """Return a simple linear model_fn: (1, T, d_model) -> (1, T, vocab_size)."""
    torch.manual_seed(42)
    W = torch.randn(d_model, vocab_size)

    def model_fn(hidden: Tensor) -> Tensor:
        # hidden: (1, T, d_model)
        return hidden @ W  # (1, T, vocab_size)

    return model_fn, W


@pytest.fixture
def model_fn_and_W():
    return make_linear_model_fn(D_MODEL, VOCAB_SIZE)


@pytest.fixture
def model_fn(model_fn_and_W):
    fn, _ = model_fn_and_W
    return fn


@pytest.fixture
def W(model_fn_and_W):
    _, w = model_fn_and_W
    return w


@pytest.fixture
def hidden_states():
    torch.manual_seed(0)
    return torch.randn(1, SEQ_LEN, D_MODEL)


@pytest.fixture
def steering_vector():
    torch.manual_seed(7)
    return torch.randn(D_MODEL)


@pytest.fixture
def target():
    return SteeringTarget(
        name="positive",
        target_token_ids=[0, 1, 2],
        anti_target_token_ids=[10, 11, 12],
    )


@pytest.fixture
def evaluator(model_fn):
    return SteeringEvaluator(model_fn, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# 1. SteeringTarget stores fields correctly
# ---------------------------------------------------------------------------


def test_steering_target_stores_fields():
    t = SteeringTarget(
        name="sentiment",
        target_token_ids=[1, 2, 3],
        anti_target_token_ids=[4, 5, 6],
    )
    assert t.name == "sentiment"
    assert t.target_token_ids == [1, 2, 3]
    assert t.anti_target_token_ids == [4, 5, 6]


# ---------------------------------------------------------------------------
# 2. SteeringResult fields are accessible
# ---------------------------------------------------------------------------


def test_steering_result_fields_accessible():
    result = SteeringResult(
        method_name="pca",
        concept_shift=0.05,
        fluency_kl=0.01,
        steering_efficiency=0.02,
        steering_norm=2.5,
    )
    assert result.method_name == "pca"
    assert result.concept_shift == pytest.approx(0.05)
    assert result.fluency_kl == pytest.approx(0.01)
    assert result.steering_efficiency == pytest.approx(0.02)
    assert result.steering_norm == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# 3. SteeringEvaluator.evaluate_steering returns SteeringResult
# ---------------------------------------------------------------------------


def test_evaluate_steering_returns_steering_result(evaluator, hidden_states, steering_vector, target):
    result = evaluator.evaluate_steering(hidden_states, steering_vector, target)
    assert isinstance(result, SteeringResult)


# ---------------------------------------------------------------------------
# 4. concept_shift is a float
# ---------------------------------------------------------------------------


def test_concept_shift_is_float(evaluator, hidden_states, steering_vector, target):
    result = evaluator.evaluate_steering(hidden_states, steering_vector, target)
    assert isinstance(result.concept_shift, float)


# ---------------------------------------------------------------------------
# 5. fluency_kl is non-negative
# ---------------------------------------------------------------------------


def test_fluency_kl_non_negative(evaluator, hidden_states, steering_vector, target):
    result = evaluator.evaluate_steering(hidden_states, steering_vector, target)
    assert result.fluency_kl >= 0.0, f"Expected fluency_kl >= 0, got {result.fluency_kl}"


# ---------------------------------------------------------------------------
# 6. steering_norm > 0 when vector is non-zero
# ---------------------------------------------------------------------------


def test_steering_norm_positive_for_nonzero_vector(evaluator, hidden_states, target):
    torch.manual_seed(3)
    vec = torch.randn(D_MODEL)  # non-zero with overwhelming probability
    result = evaluator.evaluate_steering(hidden_states, vec, target)
    assert result.steering_norm > 0.0, f"Expected steering_norm > 0, got {result.steering_norm}"


# ---------------------------------------------------------------------------
# 7. steering_efficiency = concept_shift / (steering_norm + 1e-8)
# ---------------------------------------------------------------------------


def test_steering_efficiency_formula(evaluator, hidden_states, steering_vector, target):
    result = evaluator.evaluate_steering(hidden_states, steering_vector, target)
    expected_efficiency = result.concept_shift / (result.steering_norm + 1e-8)
    assert result.steering_efficiency == pytest.approx(expected_efficiency, rel=1e-5)


# ---------------------------------------------------------------------------
# 8. steering in target direction increases target probs
# ---------------------------------------------------------------------------


def test_steering_increases_target_probs(hidden_states, target):
    """Construct a model_fn such that the steering vector directly increases target logits."""
    torch.manual_seed(99)
    # W maps d_model -> vocab_size; steering in direction W[:, target_ids].mean(1) boosts those logits
    W = torch.zeros(D_MODEL, VOCAB_SIZE)
    # Set up W so token 0 gets higher logit when we steer in direction of W[:, 0]
    direction = torch.zeros(D_MODEL)
    direction[0] = 1.0  # unit vector along dim 0

    # Make W[:, 0] align with direction so steering boosts token 0
    W[0, 0] = 10.0  # large positive weight for token 0 at dim 0

    def model_fn(hidden: Tensor) -> Tensor:
        return hidden @ W

    tgt = SteeringTarget(name="test", target_token_ids=[0], anti_target_token_ids=[1])
    ev = SteeringEvaluator(model_fn, VOCAB_SIZE)

    # Steer strongly in the direction that boosts token 0
    sv = direction * 5.0
    result = ev.evaluate_steering(hidden_states, sv, tgt, alpha=1.0)
    assert result.concept_shift > 0.0, (
        f"Expected positive concept_shift when steering toward target, got {result.concept_shift}"
    )


# ---------------------------------------------------------------------------
# 9. SteeringComparison.compare groups by method_name
# ---------------------------------------------------------------------------


def test_compare_groups_by_method_name():
    r1 = SteeringResult("pca", 0.1, 0.01, 0.05, 2.0)
    r2 = SteeringResult("pca", 0.2, 0.02, 0.10, 2.0)
    r3 = SteeringResult("mean_diff", 0.3, 0.03, 0.15, 2.0)
    comp = SteeringComparison()
    summary = comp.compare([r1, r2, r3])
    assert set(summary.keys()) == {"pca", "mean_diff"}


# ---------------------------------------------------------------------------
# 10. compare averages metrics correctly for single method with 2 results
# ---------------------------------------------------------------------------


def test_compare_averages_metrics_correctly():
    r1 = SteeringResult("pca", concept_shift=0.1, fluency_kl=0.02, steering_efficiency=0.05, steering_norm=2.0)
    r2 = SteeringResult("pca", concept_shift=0.3, fluency_kl=0.04, steering_efficiency=0.15, steering_norm=2.0)
    comp = SteeringComparison()
    summary = comp.compare([r1, r2])

    assert "pca" in summary
    pca = summary["pca"]
    assert pca["mean_concept_shift"] == pytest.approx(0.2, rel=1e-5)
    assert pca["mean_fluency_kl"] == pytest.approx(0.03, rel=1e-5)
    assert pca["mean_efficiency"] == pytest.approx(0.10, rel=1e-5)
    assert pca["n_results"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 11. rank_by returns correct order (highest first)
# ---------------------------------------------------------------------------


def test_rank_by_returns_descending_order():
    r1 = SteeringResult("a", concept_shift=0.1, fluency_kl=0.01, steering_efficiency=0.05, steering_norm=2.0)
    r2 = SteeringResult("b", concept_shift=0.5, fluency_kl=0.05, steering_efficiency=0.25, steering_norm=2.0)
    r3 = SteeringResult("c", concept_shift=0.3, fluency_kl=0.03, steering_efficiency=0.15, steering_norm=2.0)
    comp = SteeringComparison()
    ranked = comp.rank_by([r1, r2, r3], metric="steering_efficiency")
    efficiencies = [r.steering_efficiency for r in ranked]
    assert efficiencies == sorted(efficiencies, reverse=True), (
        f"Expected descending order, got {efficiencies}"
    )
    assert ranked[0].method_name == "b"


# ---------------------------------------------------------------------------
# 12. rank_by returns descending order by the given metric
# ---------------------------------------------------------------------------


def test_rank_by_descending_for_any_metric():
    r1 = SteeringResult("a", concept_shift=0.1, fluency_kl=0.30, steering_efficiency=0.05, steering_norm=2.0)
    r2 = SteeringResult("b", concept_shift=0.5, fluency_kl=0.05, steering_efficiency=0.25, steering_norm=2.0)
    r3 = SteeringResult("c", concept_shift=0.3, fluency_kl=0.15, steering_efficiency=0.15, steering_norm=2.0)
    comp = SteeringComparison()
    # rank by fluency_kl descending (even if semantically lower is better, API is always descending)
    ranked = comp.rank_by([r1, r2, r3], metric="fluency_kl")
    kls = [r.fluency_kl for r in ranked]
    assert kls == sorted(kls, reverse=True), f"Expected descending, got {kls}"
    assert ranked[0].method_name == "a"


# ---------------------------------------------------------------------------
# 13. SteeringVectorNormalizer.normalize produces unit vector
# ---------------------------------------------------------------------------


def test_normalizer_normalize_unit_vector():
    torch.manual_seed(5)
    normalizer = SteeringVectorNormalizer()
    v = torch.randn(D_MODEL)
    v_norm = normalizer.normalize(v)
    norm = v_norm.norm().item()
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


# ---------------------------------------------------------------------------
# 14. scale_to_norm produces correct norm
# ---------------------------------------------------------------------------


def test_normalizer_scale_to_norm():
    torch.manual_seed(6)
    normalizer = SteeringVectorNormalizer()
    v = torch.randn(D_MODEL)
    target_norm = 3.7
    v_scaled = normalizer.scale_to_norm(v, target_norm)
    actual_norm = v_scaled.norm().item()
    assert abs(actual_norm - target_norm) < 1e-5, (
        f"Expected norm {target_norm}, got {actual_norm}"
    )


# ---------------------------------------------------------------------------
# 15. project_out removes basis component (dot product with basis ≈ 0 after projection)
# ---------------------------------------------------------------------------


def test_normalizer_project_out_removes_component():
    torch.manual_seed(8)
    normalizer = SteeringVectorNormalizer()
    v = torch.randn(D_MODEL)
    basis = torch.randn(D_MODEL)
    projected = normalizer.project_out(v, basis)
    # The dot product of projected vector with (normalized) basis should be ~0
    basis_hat = basis / (basis.norm() + 1e-8)
    dot = torch.dot(projected, basis_hat).item()
    assert abs(dot) < 1e-5, f"Expected dot product ≈ 0 after project_out, got {dot}"
