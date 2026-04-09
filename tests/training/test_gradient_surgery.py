"""Tests for gradient_surgery.py: conflict detection, projection, and multi-task aggregation."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.gradient_surgery import (
    flatten_gradients,
    unflatten_gradients,
    compute_gradient_conflict,
    project_gradient,
    gradient_surgery_step,
    gradient_vaccine,
    GradientSurgeon,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model_with_grads(seed: int = 0) -> nn.Module:
    """Create a small nn.Linear with synthetic gradients assigned."""
    torch.manual_seed(seed)
    model = nn.Linear(64, 64)
    # Assign synthetic gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    return model


def make_model_no_grads() -> nn.Module:
    """Create a small nn.Linear with no gradients."""
    torch.manual_seed(1)
    return nn.Linear(64, 64)


def total_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.grad is not None)


# ---------------------------------------------------------------------------
# flatten_gradients tests
# ---------------------------------------------------------------------------

def test_flatten_gradients_shape():
    """flatten_gradients output shape matches sum of parameter sizes."""
    model = make_model_with_grads()
    flat = flatten_gradients(model)
    expected = total_param_count(model)
    assert flat.shape == (expected,), (
        f"Expected shape ({expected},), got {flat.shape}"
    )


def test_flatten_gradients_skips_none():
    """flatten_gradients skips parameters without gradients."""
    model = nn.Linear(64, 64)
    # Only set grad on weight, not bias
    model.weight.grad = torch.randn_like(model.weight)
    # model.bias.grad remains None
    flat = flatten_gradients(model)
    assert flat.shape == (model.weight.numel(),)


# ---------------------------------------------------------------------------
# unflatten_gradients round-trip tests
# ---------------------------------------------------------------------------

def test_flatten_unflatten_roundtrip():
    """flatten_gradients -> unflatten_gradients round-trips grad values."""
    model = make_model_with_grads(seed=42)
    original_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    flat = flatten_gradients(model)
    # Perturb grads to confirm unflatten overwrites them
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    unflatten_gradients(flat, model)

    for name, p in model.named_parameters():
        if name in original_grads:
            assert torch.allclose(p.grad, original_grads[name]), (
                f"Parameter '{name}' grad did not round-trip correctly"
            )


def test_unflatten_only_updates_non_none():
    """unflatten_gradients only updates parameters that had non-None gradients."""
    model = nn.Linear(64, 64)
    # Only weight gets a grad
    model.weight.grad = torch.ones_like(model.weight)
    # bias.grad is None

    flat = flatten_gradients(model)
    new_flat = flat * 2.0
    unflatten_gradients(new_flat, model)

    assert torch.allclose(model.weight.grad, torch.ones_like(model.weight) * 2.0)
    assert model.bias.grad is None, "bias grad should remain None"


# ---------------------------------------------------------------------------
# compute_gradient_conflict tests
# ---------------------------------------------------------------------------

def test_compute_gradient_conflict_parallel():
    """Parallel (same direction) vectors -> cosine similarity ~1.0."""
    g = torch.tensor([1.0, 2.0, 3.0])
    result = compute_gradient_conflict(g, g * 5.0)
    assert abs(result - 1.0) < 1e-5, f"Expected ~1.0, got {result}"


def test_compute_gradient_conflict_anti_parallel():
    """Anti-parallel vectors -> cosine similarity ~-1.0."""
    g = torch.tensor([1.0, 2.0, 3.0])
    result = compute_gradient_conflict(g, -g)
    assert abs(result - (-1.0)) < 1e-5, f"Expected ~-1.0, got {result}"


def test_compute_gradient_conflict_orthogonal():
    """Orthogonal vectors -> cosine similarity ~0.0."""
    g1 = torch.tensor([1.0, 0.0, 0.0])
    g2 = torch.tensor([0.0, 1.0, 0.0])
    result = compute_gradient_conflict(g1, g2)
    assert abs(result) < 1e-5, f"Expected ~0.0, got {result}"


# ---------------------------------------------------------------------------
# project_gradient tests
# ---------------------------------------------------------------------------

def test_project_gradient_output_shape():
    """project_gradient returns same shape as grad."""
    grad = torch.randn(10, 5)
    onto = torch.randn(10, 5)
    result = project_gradient(grad, onto)
    assert result.shape == grad.shape, (
        f"Expected shape {grad.shape}, got {result.shape}"
    )


def test_project_gradient_self_projection():
    """Projecting a vector onto itself returns itself (self-projection = self)."""
    g = torch.tensor([3.0, 4.0])
    result = project_gradient(g, g)
    assert torch.allclose(result, g, atol=1e-5), (
        f"Self-projection should equal self, got {result}"
    )


# ---------------------------------------------------------------------------
# gradient_surgery_step tests
# ---------------------------------------------------------------------------

def test_gradient_surgery_step_output_shape():
    """gradient_surgery_step returns a tensor with shape (total_params,)."""
    dim = 64 * 64 + 64  # matches nn.Linear(64, 64)
    grads = [torch.randn(dim) for _ in range(3)]
    result = gradient_surgery_step(grads)
    assert result.shape == (dim,), f"Expected shape ({dim},), got {result.shape}"


def test_gradient_surgery_step_conflicting_gradients_reduced():
    """PCGrad should reduce or eliminate conflicts between anti-parallel gradients."""
    dim = 100
    g1 = torch.ones(dim)
    g2 = -torch.ones(dim)  # perfectly anti-parallel

    conflict_before = compute_gradient_conflict(g1, g2)
    assert conflict_before < 0, "Pre-surgery gradients should conflict"

    result = gradient_surgery_step([g1, g2])
    # The result should have zero or near-zero magnitude because both gradients
    # cancel after surgery (each removes the other's projection)
    assert result.norm().item() < 1e-5 or True, "Surgery ran without error"
    # More specifically: the mean of projected g1 and projected g2 should be small
    # g1 projected away from g2 -> [0,...,0], g2 projected away from g1 -> [0,...,0]
    assert result.norm().item() < 1e-4, (
        f"Anti-parallel gradients should cancel after surgery, norm={result.norm().item()}"
    )


# ---------------------------------------------------------------------------
# gradient_vaccine tests
# ---------------------------------------------------------------------------

def test_gradient_vaccine_output_shape():
    """gradient_vaccine returns a tensor with shape (total_params,)."""
    dim = 64 * 64 + 64
    grads = [torch.randn(dim) for _ in range(3)]
    result = gradient_vaccine(grads)
    assert result.shape == (dim,), f"Expected shape ({dim},), got {result.shape}"


# ---------------------------------------------------------------------------
# GradientSurgeon tests
# ---------------------------------------------------------------------------

def test_gradient_surgeon_aggregate_mean():
    """GradientSurgeon.aggregate with method='mean' returns element-wise mean."""
    dim = 50
    torch.manual_seed(0)
    grads = [torch.randn(dim) for _ in range(4)]
    surgeon = GradientSurgeon(method="mean")
    result = surgeon.aggregate(grads)
    expected = torch.stack(grads).mean(dim=0)
    assert result.shape == (dim,)
    assert torch.allclose(result.float(), expected.float(), atol=1e-5), (
        "Mean aggregation should match torch.stack(...).mean()"
    )


def test_gradient_surgeon_conflict_matrix_shape():
    """GradientSurgeon.conflict_matrix returns (n_tasks, n_tasks) tensor."""
    n_tasks = 5
    dim = 80
    grads = [torch.randn(dim) for _ in range(n_tasks)]
    surgeon = GradientSurgeon(method="pcgrad")
    matrix = surgeon.conflict_matrix(grads)
    assert matrix.shape == (n_tasks, n_tasks), (
        f"Expected ({n_tasks}, {n_tasks}), got {matrix.shape}"
    )


def test_gradient_surgeon_conflict_matrix_diagonal():
    """Diagonal of conflict_matrix should be ~1.0 (self-similarity)."""
    n_tasks = 4
    dim = 60
    torch.manual_seed(7)
    grads = [torch.randn(dim) for _ in range(n_tasks)]
    surgeon = GradientSurgeon(method="mean")
    matrix = surgeon.conflict_matrix(grads)
    for i in range(n_tasks):
        assert abs(matrix[i, i].item() - 1.0) < 1e-4, (
            f"Diagonal[{i}] should be ~1.0, got {matrix[i, i].item()}"
        )


def test_gradient_surgeon_aggregate_pcgrad_shape():
    """GradientSurgeon.aggregate with method='pcgrad' returns correct shape."""
    dim = 128
    grads = [torch.randn(dim) for _ in range(3)]
    surgeon = GradientSurgeon(method="pcgrad")
    result = surgeon.aggregate(grads)
    assert result.shape == (dim,), f"Expected ({dim},), got {result.shape}"


def test_gradient_surgeon_aggregate_vaccine_shape():
    """GradientSurgeon.aggregate with method='vaccine' returns correct shape."""
    dim = 128
    grads = [torch.randn(dim) for _ in range(3)]
    surgeon = GradientSurgeon(method="vaccine")
    result = surgeon.aggregate(grads)
    assert result.shape == (dim,), f"Expected ({dim},), got {result.shape}"


def test_gradient_surgeon_invalid_method():
    """GradientSurgeon should raise ValueError for unknown method."""
    with pytest.raises(ValueError, match="Unknown method"):
        GradientSurgeon(method="invalid_method")
