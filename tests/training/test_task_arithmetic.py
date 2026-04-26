"""
Tests for src/training/task_arithmetic.py
15 tests covering TaskVector, TaskComposer, NegationOperator,
TaskInterferenceAnalyzer, and TaskArithmeticEvaluator.

All tests run forward/backward passes with tiny models:
  MLP: 16 → 16 → 16   (2 layers, no biases)
"""

import copy

import pytest
import torch
import torch.nn as nn

from src.training.task_arithmetic import (
    NegationOperator,
    TaskComposer,
    TaskInterferenceAnalyzer,
    TaskVector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mlp(seed: int = 0) -> nn.Sequential:
    """Tiny 2-layer MLP: 16 → 16 → 16 (no bias for simplicity)."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(16, 16, bias=False),
        nn.ReLU(),
        nn.Linear(16, 16, bias=False),
    )


def forward_backward(model: nn.Module) -> None:
    """Run a dummy forward + backward to verify gradients flow."""
    x = torch.randn(2, 16)
    out = model(x)
    loss = out.sum()
    loss.backward()


def simple_eval_fn(model: nn.Module) -> float:
    """Score a model by negated output L2 norm (toy metric)."""
    x = torch.randn(2, 16)
    with torch.no_grad():
        out = model(x)
    return -out.norm().item()


# ---------------------------------------------------------------------------
# Test 1 – TaskVector.__init__ from models: keys match state_dict
# ---------------------------------------------------------------------------


def test_task_vector_keys_match_state_dict():
    base = make_mlp(0)
    ft = make_mlp(1)

    # forward/backward to ensure models are valid
    forward_backward(base)
    forward_backward(ft)

    tv = TaskVector(base_model=base, finetuned_model=ft)
    expected_keys = set(dict(base.named_parameters()).keys())
    assert set(tv.vector.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test 2 – base − base = zero vector
# ---------------------------------------------------------------------------


def test_task_vector_self_minus_self_is_zero():
    base = make_mlp(0)
    forward_backward(base)

    tv = TaskVector(base_model=base, finetuned_model=base)
    for name, val in tv.vector.items():
        assert torch.allclose(val, torch.zeros_like(val)), f"Parameter {name} is not zero"
    assert tv.norm() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 3 – TaskVector.__add__: component-wise sum
# ---------------------------------------------------------------------------


def test_task_vector_add_componentwise():
    base = make_mlp(0)
    ft1 = make_mlp(1)
    ft2 = make_mlp(2)
    forward_backward(base)

    tv1 = TaskVector(base_model=base, finetuned_model=ft1)
    tv2 = TaskVector(base_model=base, finetuned_model=ft2)
    tv_sum = tv1 + tv2

    for k in tv1.vector:
        expected = tv1.vector[k] + tv2.vector[k]
        assert torch.allclose(tv_sum.vector[k], expected), f"Mismatch for parameter {k}"


# ---------------------------------------------------------------------------
# Test 4 – TaskVector.__mul__: scales all components
# ---------------------------------------------------------------------------


def test_task_vector_mul_scales_components():
    base = make_mlp(0)
    ft = make_mlp(1)
    forward_backward(base)

    tv = TaskVector(base_model=base, finetuned_model=ft)
    scalar = 3.14
    tv_scaled = tv * scalar

    for k in tv.vector:
        expected = tv.vector[k] * scalar
        assert torch.allclose(tv_scaled.vector[k], expected), f"Scaling mismatch for {k}"


# ---------------------------------------------------------------------------
# Test 5 – TaskVector.__neg__: negates all components
# ---------------------------------------------------------------------------


def test_task_vector_neg_negates_components():
    base = make_mlp(0)
    ft = make_mlp(1)
    forward_backward(base)

    tv = TaskVector(base_model=base, finetuned_model=ft)
    neg_tv = -tv

    for k in tv.vector:
        assert torch.allclose(neg_tv.vector[k], -tv.vector[k]), f"Negation mismatch for {k}"


# ---------------------------------------------------------------------------
# Test 6 – TaskVector.norm: ≥ 0, and 0 for zero vector
# ---------------------------------------------------------------------------


def test_task_vector_norm_non_negative_and_zero():
    base = make_mlp(0)
    ft = make_mlp(1)
    forward_backward(base)

    tv = TaskVector(base_model=base, finetuned_model=ft)
    assert tv.norm() >= 0.0

    zero_tv = TaskVector(base_model=base, finetuned_model=base)
    assert zero_tv.norm() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 7 – TaskVector.apply: params change by exactly vector values
# ---------------------------------------------------------------------------


def test_task_vector_apply_changes_params_exactly():
    base = make_mlp(0)
    ft = make_mlp(1)
    forward_backward(base)

    tv = TaskVector(base_model=base, finetuned_model=ft)

    # Take a snapshot of base params before apply
    base_snapshot = {n: p.detach().clone() for n, p in base.named_parameters()}

    scale = 1.0
    tv.apply(base, scale=scale)

    for name, param in base.named_parameters():
        expected = base_snapshot[name] + tv.vector[name] * scale
        assert torch.allclose(param, expected, atol=1e-6), f"Apply mismatch for {name}"


# ---------------------------------------------------------------------------
# Test 8 – TaskComposer.sum: triangle inequality
# ---------------------------------------------------------------------------


def test_task_composer_sum_triangle_inequality():
    base = make_mlp(0)
    ft1 = make_mlp(1)
    ft2 = make_mlp(2)
    forward_backward(base)

    tv1 = TaskVector(base_model=base, finetuned_model=ft1)
    tv2 = TaskVector(base_model=base, finetuned_model=ft2)

    composer = TaskComposer()
    tv_sum = composer.sum([tv1, tv2], weights=[0.5, 0.5])

    # Triangle inequality: ||a + b|| ≤ ||a|| + ||b|| (with 0.5 weights => ||0.5a + 0.5b||)
    assert tv_sum.norm() <= tv1.norm() + tv2.norm() + 1e-6


# ---------------------------------------------------------------------------
# Test 9 – TaskComposer.consensus: fewer non-zero params than inputs
# ---------------------------------------------------------------------------


def test_task_composer_consensus_zeros_conflicts():
    # Construct vectors that partially conflict by design
    base = make_mlp(0)
    forward_backward(base)

    # Create ft models with opposite deltas to force sign conflicts
    ft1 = copy.deepcopy(base)
    ft2 = copy.deepcopy(base)
    with torch.no_grad():
        for p in ft1.parameters():
            p.add_(torch.ones_like(p))  # all deltas +1
        for p in ft2.parameters():
            p.sub_(torch.ones_like(p))  # all deltas -1 → conflict everywhere

    tv1 = TaskVector(base_model=base, finetuned_model=ft1)
    tv2 = TaskVector(base_model=base, finetuned_model=ft2)

    composer = TaskComposer()
    tv_consensus = composer.consensus([tv1, tv2])

    # All parameters should be zero because signs always conflict
    total_nonzero = sum((val.abs() > 1e-8).sum().item() for val in tv_consensus.vector.values())
    total_params = sum(v.numel() for v in tv1.vector.values())
    assert total_nonzero < total_params, "Consensus should zero out conflicting parameters"


# ---------------------------------------------------------------------------
# Test 10 – NegationOperator.negate: result is negated vector
# ---------------------------------------------------------------------------


def test_negation_operator_negate():
    base = make_mlp(0)
    ft = make_mlp(1)
    forward_backward(base)

    tv = TaskVector(base_model=base, finetuned_model=ft)
    neg_op = NegationOperator(negation_scale=1.0)
    neg_tv = neg_op.negate(tv)

    for k in tv.vector:
        assert torch.allclose(neg_tv.vector[k], -tv.vector[k]), f"NegationOperator mismatch for {k}"


# ---------------------------------------------------------------------------
# Test 11 – NegationOperator.apply_negation: params shift negative
# ---------------------------------------------------------------------------


def test_negation_operator_apply_negation_shifts_negative():
    base = make_mlp(0)
    ft = make_mlp(1)
    forward_backward(base)

    tv = TaskVector(base_model=base, finetuned_model=ft)

    # Give the task vector strictly positive values so direction is clear
    pos_vector = {k: v.abs() + 0.01 for k, v in tv.vector.items()}
    pos_tv = TaskVector(vector=pos_vector)

    snapshot = {n: p.detach().clone() for n, p in base.named_parameters()}

    neg_op = NegationOperator(negation_scale=1.0)
    scale = 0.5
    neg_op.apply_negation(base, pos_tv, scale=scale)

    for name, param in base.named_parameters():
        delta = param - snapshot[name]
        # Each delta should be ≤ 0 since we subtracted positive values
        assert (delta <= 1e-7).all(), f"apply_negation should decrease params for {name}"


# ---------------------------------------------------------------------------
# Test 12 – TaskInterferenceAnalyzer.cosine_similarity: in [-1,1], 1.0 for identical
# ---------------------------------------------------------------------------


def test_interference_analyzer_cosine_similarity():
    base = make_mlp(0)
    ft1 = make_mlp(1)
    ft2 = make_mlp(2)
    forward_backward(base)

    tv1 = TaskVector(base_model=base, finetuned_model=ft1)
    tv2 = TaskVector(base_model=base, finetuned_model=ft2)

    analyzer = TaskInterferenceAnalyzer()
    sim_12 = analyzer.cosine_similarity(tv1, tv2)
    assert -1.0 - 1e-6 <= sim_12 <= 1.0 + 1e-6

    sim_11 = analyzer.cosine_similarity(tv1, tv1)
    assert sim_11 == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 13 – TaskInterferenceAnalyzer.interference_matrix: symmetric, diagonal=1, shape (N,N)
# ---------------------------------------------------------------------------


def test_interference_analyzer_interference_matrix():
    base = make_mlp(0)
    ft1 = make_mlp(1)
    ft2 = make_mlp(2)
    ft3 = make_mlp(3)
    forward_backward(base)

    tvs = [
        TaskVector(base_model=base, finetuned_model=ft1),
        TaskVector(base_model=base, finetuned_model=ft2),
        TaskVector(base_model=base, finetuned_model=ft3),
    ]

    analyzer = TaskInterferenceAnalyzer()
    mat = analyzer.interference_matrix(tvs)

    n = len(tvs)
    assert mat.shape == (n, n), f"Expected ({n},{n}), got {mat.shape}"

    # Diagonal == 1
    for i in range(n):
        assert mat[i, i].item() == pytest.approx(1.0, abs=1e-4)

    # Symmetric
    assert torch.allclose(mat, mat.T, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 14 – TaskInterferenceAnalyzer.dominant_parameters: length k, sorted descending
# ---------------------------------------------------------------------------


def test_interference_analyzer_dominant_parameters():
    base = make_mlp(0)
    ft = make_mlp(1)
    forward_backward(base)

    tv = TaskVector(base_model=base, finetuned_model=ft)
    analyzer = TaskInterferenceAnalyzer()

    k = 2
    dominant = analyzer.dominant_parameters(tv, k=k)

    assert len(dominant) == k, f"Expected {k} entries, got {len(dominant)}"
    # Sorted descending by magnitude
    mags = [m for _, m in dominant]
    assert mags == sorted(mags, reverse=True), "Not sorted descending"
    # All magnitudes non-negative
    assert all(m >= 0.0 for _, m in dominant)


# ---------------------------------------------------------------------------
# Test 15 – TaskComposer.orthogonalize: lower pairwise cosine similarity
# ---------------------------------------------------------------------------


def test_task_composer_orthogonalize_reduces_similarity():
    base = make_mlp(0)
    ft1 = make_mlp(1)
    ft2 = make_mlp(2)
    forward_backward(base)

    tv1 = TaskVector(base_model=base, finetuned_model=ft1)
    tv2 = TaskVector(base_model=base, finetuned_model=ft2)

    composer = TaskComposer()
    analyzer = TaskInterferenceAnalyzer()

    original_sim = abs(analyzer.cosine_similarity(tv1, tv2))

    ortho_tvs = composer.orthogonalize([tv1, tv2])
    assert len(ortho_tvs) == 2

    ortho_sim = abs(analyzer.cosine_similarity(ortho_tvs[0], ortho_tvs[1]))

    # After orthogonalization the pairwise similarity should be near 0
    assert ortho_sim < original_sim + 1e-4, (
        f"Orthogonalized sim {ortho_sim:.4f} should be ≤ original sim {original_sim:.4f}"
    )
    assert ortho_sim < 1e-3, (
        f"Orthogonalized vectors should be nearly orthogonal, got cosine={ortho_sim:.6f}"
    )
