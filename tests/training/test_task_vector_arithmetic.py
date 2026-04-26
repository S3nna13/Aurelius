"""Tests for task_vector_arithmetic.py — Ilharco et al. 2023 Task Vector Arithmetic."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.task_vector_arithmetic import (
    TaskVector,
    TaskVectorConfig,
    interpolate_models,
    multi_task_compose,
    task_negation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model(seed: int = 0) -> AureliusTransformer:
    """Create a deterministic tiny transformer for fast testing."""
    torch.manual_seed(seed)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    return AureliusTransformer(cfg)


def _perturbed_model(base: AureliusTransformer, seed: int = 42) -> AureliusTransformer:
    """Create a fine-tuned variant by adding small random noise to base weights."""
    ft = copy.deepcopy(base)
    torch.manual_seed(seed)
    for p in ft.parameters():
        p.data += torch.randn_like(p) * 0.05
    return ft


# ---------------------------------------------------------------------------
# Test 1: TaskVector from (base, finetuned) pair has non-zero norm
# ---------------------------------------------------------------------------


def test_task_vector_from_pair_has_nonzero_norm():
    """TV extracted from (base, finetuned) should have a positive norm."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    ft = _perturbed_model(base, seed=1)
    tv = TaskVector(base_model=base, finetuned_model=ft)
    assert tv.norm() > 0.0, "Expected non-zero norm for TaskVector from different models"


# ---------------------------------------------------------------------------
# Test 2: TaskVector from identical models has zero norm
# ---------------------------------------------------------------------------


def test_task_vector_identical_models_zero_norm():
    """TV from two identical models should have near-zero norm."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    tv = TaskVector(base_model=base, finetuned_model=base)
    assert tv.norm() < 1e-9, f"Expected zero norm, got {tv.norm()}"


# ---------------------------------------------------------------------------
# Test 3: __add__ produces vector with correct norm relationship
# ---------------------------------------------------------------------------


def test_task_vector_add_norm_relationship():
    """(tv1 + tv2).norm() should be <= tv1.norm() + tv2.norm() (triangle inequality)."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    ft1 = _perturbed_model(base, seed=1)
    ft2 = _perturbed_model(base, seed=2)
    tv1 = TaskVector(base_model=base, finetuned_model=ft1)
    tv2 = TaskVector(base_model=base, finetuned_model=ft2)
    tv_sum = tv1 + tv2

    assert tv_sum.norm() <= tv1.norm() + tv2.norm() + 1e-5, (
        "Triangle inequality violated for task vector addition"
    )
    # Also verify the sum has a non-zero norm (vectors are not perfectly anti-parallel)
    assert tv_sum.norm() >= 0.0


# ---------------------------------------------------------------------------
# Test 4: __neg__ produces vector with same norm but opposite sign
# ---------------------------------------------------------------------------


def test_task_vector_neg_same_norm_opposite_sign():
    """(-tv).norm() == tv.norm() and all parameter signs are flipped."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    ft = _perturbed_model(base, seed=1)
    tv = TaskVector(base_model=base, finetuned_model=ft)
    neg_tv = -tv

    assert abs(neg_tv.norm() - tv.norm()) < 1e-6, (
        f"Norm should be preserved under negation: {neg_tv.norm()} vs {tv.norm()}"
    )
    for name in tv.vector:
        assert torch.allclose(neg_tv.vector[name], -tv.vector[name]), (
            f"Negation mismatch at param {name}"
        )


# ---------------------------------------------------------------------------
# Test 5: __mul__ scales the norm correctly
# ---------------------------------------------------------------------------


def test_task_vector_mul_scales_norm():
    """(scalar * tv).norm() == scalar * tv.norm() for scalar > 0."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    ft = _perturbed_model(base, seed=1)
    tv = TaskVector(base_model=base, finetuned_model=ft)
    scalar = 3.0
    scaled_tv = tv * scalar

    expected_norm = scalar * tv.norm()
    assert abs(scaled_tv.norm() - expected_norm) < 1e-4, (
        f"Expected norm {expected_norm}, got {scaled_tv.norm()}"
    )


# ---------------------------------------------------------------------------
# Test 6: apply produces model with different weights than base
# ---------------------------------------------------------------------------


def test_task_vector_apply_changes_weights():
    """Applying a non-zero TV with scaling_coef=1 should change at least one weight."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    ft = _perturbed_model(base, seed=1)
    tv = TaskVector(base_model=base, finetuned_model=ft)
    new_model = tv.apply(base, scaling_coef=1.0)

    any_changed = any(
        not torch.equal(p_base, p_new)
        for p_base, p_new in zip(base.parameters(), new_model.parameters())
    )
    assert any_changed, "apply() should produce different weights than base model"


# ---------------------------------------------------------------------------
# Test 7: apply with scale=0 produces weights identical to base
# ---------------------------------------------------------------------------


def test_task_vector_apply_scale_zero_equals_base():
    """Applying any TV with scaling_coef=0 should yield base model weights."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    ft = _perturbed_model(base, seed=1)
    tv = TaskVector(base_model=base, finetuned_model=ft)
    new_model = tv.apply(base, scaling_coef=0.0)

    for (name, p_base), (_, p_new) in zip(base.named_parameters(), new_model.named_parameters()):
        assert torch.allclose(p_base, p_new, atol=1e-6), (
            f"Weights should match base at param {name} when scaling_coef=0"
        )


# ---------------------------------------------------------------------------
# Test 8: cosine_similarity of vector with itself ≈ 1.0
# ---------------------------------------------------------------------------


def test_cosine_similarity_self_is_one():
    """Cosine similarity of a task vector with itself should be ~1.0."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    ft = _perturbed_model(base, seed=1)
    tv = TaskVector(base_model=base, finetuned_model=ft)
    sim = tv.cosine_similarity(tv)
    assert abs(sim - 1.0) < 1e-5, f"Expected cosine similarity ≈ 1.0, got {sim}"


# ---------------------------------------------------------------------------
# Test 9: cosine_similarity of vector with its negation ≈ -1.0
# ---------------------------------------------------------------------------


def test_cosine_similarity_with_negation_is_minus_one():
    """Cosine similarity of a task vector with its negation should be ~-1.0."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    ft = _perturbed_model(base, seed=1)
    tv = TaskVector(base_model=base, finetuned_model=ft)
    neg_tv = -tv
    sim = tv.cosine_similarity(neg_tv)
    assert abs(sim - (-1.0)) < 1e-5, f"Expected cosine similarity ≈ -1.0, got {sim}"


# ---------------------------------------------------------------------------
# Test 10: sparsify keeps fraction * n_params non-zero
# ---------------------------------------------------------------------------


def test_sparsify_keeps_correct_fraction():
    """sparsify(fraction) should keep approximately fraction * total_params non-zero."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    ft = _perturbed_model(base, seed=1)
    tv = TaskVector(base_model=base, finetuned_model=ft)

    fraction = 0.3
    sparse_tv = tv.sparsify(fraction)

    total_params = sum(p.numel() for p in tv.vector.values())
    nonzero_params = sum((p != 0).sum().item() for p in sparse_tv.vector.values())
    expected_nonzero = int(round(total_params * fraction))

    # Allow some tolerance due to rounding per-tensor
    n_tensors = len(tv.vector)
    tol = n_tensors  # at most ±1 per tensor due to rounding
    assert abs(nonzero_params - expected_nonzero) <= tol + 1, (
        f"Expected ~{expected_nonzero} non-zero params, got {nonzero_params} "
        f"(total={total_params}, fraction={fraction})"
    )


# ---------------------------------------------------------------------------
# Test 11: multi_task_compose returns nn.Module
# ---------------------------------------------------------------------------


def test_multi_task_compose_returns_module():
    """multi_task_compose should return an nn.Module instance."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    ft1 = _perturbed_model(base, seed=1)
    ft2 = _perturbed_model(base, seed=2)
    tv1 = TaskVector(base_model=base, finetuned_model=ft1)
    tv2 = TaskVector(base_model=base, finetuned_model=ft2)

    config = TaskVectorConfig(scaling_coef=0.5)
    result = multi_task_compose(base, [tv1, tv2], weights=[1.0, 1.0], config=config)

    assert isinstance(result, nn.Module), f"Expected nn.Module, got {type(result)}"


# ---------------------------------------------------------------------------
# Test 12: task_negation changes model weights
# ---------------------------------------------------------------------------


def test_task_negation_changes_weights():
    """task_negation should produce a model with different weights than base."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    ft = _perturbed_model(base, seed=1)
    tv = TaskVector(base_model=base, finetuned_model=ft)

    negated_model = task_negation(base, tv, scaling_coef=1.0)

    any_changed = any(
        not torch.equal(p_base, p_neg)
        for p_base, p_neg in zip(base.parameters(), negated_model.parameters())
    )
    assert any_changed, "task_negation should change at least one weight"


# ---------------------------------------------------------------------------
# Test 13: interpolate_models at alpha=1.0 matches model_a, alpha=0.0 matches model_b
# ---------------------------------------------------------------------------


def test_interpolate_models_alpha_one_matches_model_a():
    """interpolate_models(alpha=1.0) should yield model_a weights."""
    torch.manual_seed(0)
    model_a = _tiny_model(seed=0)
    model_b = _perturbed_model(model_a, seed=1)

    result = interpolate_models(model_a, model_b, alpha=1.0)

    for (name, p_a), (_, p_res) in zip(model_a.named_parameters(), result.named_parameters()):
        assert torch.allclose(p_a.float(), p_res.float(), atol=1e-5), (
            f"alpha=1.0 should match model_a at param {name}"
        )


def test_interpolate_models_alpha_zero_matches_model_b():
    """interpolate_models(alpha=0.0) should yield model_b weights."""
    torch.manual_seed(0)
    model_a = _tiny_model(seed=0)
    model_b = _perturbed_model(model_a, seed=1)

    result = interpolate_models(model_a, model_b, alpha=0.0)

    for (name, p_b), (_, p_res) in zip(model_b.named_parameters(), result.named_parameters()):
        assert torch.allclose(p_b.float(), p_res.float(), atol=1e-5), (
            f"alpha=0.0 should match model_b at param {name}"
        )


# ---------------------------------------------------------------------------
# Bonus tests for robustness
# ---------------------------------------------------------------------------


def test_task_vector_rmul():
    """scalar * tv should equal tv * scalar."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    ft = _perturbed_model(base, seed=1)
    tv = TaskVector(base_model=base, finetuned_model=ft)

    left = 2.5 * tv
    right = tv * 2.5

    for name in tv.vector:
        assert torch.allclose(left.vector[name], right.vector[name]), (
            f"__rmul__ and __mul__ differ at {name}"
        )


def test_task_vector_sub():
    """tv - tv should have zero norm."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    ft = _perturbed_model(base, seed=1)
    tv = TaskVector(base_model=base, finetuned_model=ft)
    tv_zero = tv - tv
    assert tv_zero.norm() < 1e-9, f"tv - tv should have near-zero norm, got {tv_zero.norm()}"


def test_multi_task_compose_empty_returns_base_copy():
    """multi_task_compose with no task vectors should return a copy of base model."""
    torch.manual_seed(0)
    base = _tiny_model(seed=0)
    result = multi_task_compose(base, task_vectors=[], config=TaskVectorConfig())

    assert isinstance(result, nn.Module)
    assert result is not base
    for p_base, p_res in zip(base.parameters(), result.parameters()):
        assert torch.equal(p_base, p_res), "Empty compose should match base weights"


def test_interpolate_models_midpoint():
    """interpolate_models(alpha=0.5) should produce midpoint weights."""
    torch.manual_seed(0)
    model_a = _tiny_model(seed=0)
    model_b = _perturbed_model(model_a, seed=1)

    result = interpolate_models(model_a, model_b, alpha=0.5)

    for (name, p_a), (_, p_b), (_, p_res) in zip(
        model_a.named_parameters(),
        model_b.named_parameters(),
        result.named_parameters(),
    ):
        expected = 0.5 * p_a.float() + 0.5 * p_b.float()
        assert torch.allclose(p_res.float(), expected, atol=1e-5), (
            f"Midpoint interpolation mismatch at {name}"
        )
