"""Tests for gradient_surgery.py: conflict detection, projection, multi-task aggregation,
PCGradProjector, PCGradOptimizer, and TaskLossAggregator."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from aurelius.training.gradient_surgery import (
    PCGradOptimizer,
    PCGradProjector,
    TaskLossAggregator,
)

from src.training.gradient_surgery import (
    GradientSurgeon,
    compute_gradient_conflict,
    flatten_gradients,
    gradient_surgery_step,
    gradient_vaccine,
    project_gradient,
    unflatten_gradients,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_model_with_grads(seed: int = 0) -> nn.Module:
    """Create a small nn.Linear with synthetic gradients assigned."""
    torch.manual_seed(seed)
    model = nn.Linear(64, 64)
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
    assert flat.shape == (expected,), f"Expected shape ({expected},), got {flat.shape}"


def test_flatten_gradients_skips_none():
    """flatten_gradients skips parameters without gradients."""
    model = nn.Linear(64, 64)
    model.weight.grad = torch.randn_like(model.weight)
    flat = flatten_gradients(model)
    assert flat.shape == (model.weight.numel(),)


# ---------------------------------------------------------------------------
# unflatten_gradients round-trip tests
# ---------------------------------------------------------------------------


def test_flatten_unflatten_roundtrip():
    """flatten_gradients -> unflatten_gradients round-trips grad values."""
    model = make_model_with_grads(seed=42)
    original_grads = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }

    flat = flatten_gradients(model)
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
    model.weight.grad = torch.ones_like(model.weight)

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
    assert result.shape == grad.shape, f"Expected shape {grad.shape}, got {result.shape}"


def test_project_gradient_self_projection():
    """Projecting a vector onto itself returns itself (self-projection = self)."""
    g = torch.tensor([3.0, 4.0])
    result = project_gradient(g, g)
    assert torch.allclose(result, g, atol=1e-5), f"Self-projection should equal self, got {result}"


# ---------------------------------------------------------------------------
# gradient_surgery_step tests
# ---------------------------------------------------------------------------


def test_gradient_surgery_step_output_shape():
    """gradient_surgery_step returns a tensor with shape (total_params,)."""
    dim = 64 * 64 + 64
    grads = [torch.randn(dim) for _ in range(3)]
    result = gradient_surgery_step(grads)
    assert result.shape == (dim,), f"Expected shape ({dim},), got {result.shape}"


def test_gradient_surgery_step_conflicting_gradients_reduced():
    """PCGrad should reduce or eliminate conflicts between anti-parallel gradients."""
    dim = 100
    g1 = torch.ones(dim)
    g2 = -torch.ones(dim)

    conflict_before = compute_gradient_conflict(g1, g2)
    assert conflict_before < 0, "Pre-surgery gradients should conflict"

    result = gradient_surgery_step([g1, g2])
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


# ===========================================================================
# PCGradProjector tests (15 required tests)
# ===========================================================================


# 1. project output list has same length
def test_pcgrad_project_output_length():
    """PCGradProjector.project returns a list of the same length as input."""
    projector = PCGradProjector()
    grads = [torch.randn(10) for _ in range(4)]
    result = projector.project(grads)
    assert len(result) == len(grads), f"Expected {len(grads)} outputs, got {len(result)}"


# 2. Non-conflicting grads (positive dot product) -> unchanged
def test_pcgrad_project_non_conflicting_unchanged():
    """Non-conflicting gradient pairs (positive dot product) remain unchanged."""
    projector = PCGradProjector()
    g1 = torch.tensor([1.0, 2.0, 3.0])
    g2 = torch.tensor([2.0, 1.0, 1.0])  # dot > 0
    assert torch.dot(g1, g2) > 0
    result = projector.project([g1, g2])
    assert torch.allclose(result[0], g1, atol=1e-6), "g1 should be unchanged (non-conflicting)"
    assert torch.allclose(result[1], g2, atol=1e-6), "g2 should be unchanged (non-conflicting)"


# 3. Conflicting grads -> dot product becomes >= 0 after projection
def test_pcgrad_project_conflicting_resolved():
    """After projection, conflicting gradient pairs have non-negative dot product."""
    projector = PCGradProjector()
    g1 = torch.tensor([1.0, 0.0, 0.0])
    g2 = torch.tensor([-1.0, 0.0, 0.0])  # directly opposing
    assert torch.dot(g1, g2) < 0
    result = projector.project([g1, g2])
    dot_after = torch.dot(result[0], result[1]).item()
    assert dot_after >= -1e-6, f"Dot product of projected grads should be >= 0, got {dot_after}"


# 4. Projection is idempotent on non-conflicting grads
def test_pcgrad_project_idempotent_non_conflicting():
    """Projecting non-conflicting grads twice yields the same result."""
    projector = PCGradProjector()
    g1 = torch.tensor([3.0, 4.0, 0.0])
    g2 = torch.tensor([1.0, 1.0, 0.0])  # positive dot product
    assert torch.dot(g1, g2) > 0
    result1 = projector.project([g1, g2])
    result2 = projector.project(result1)
    assert torch.allclose(result1[0], result2[0], atol=1e-5), "Idempotency failed for g1"
    assert torch.allclose(result1[1], result2[1], atol=1e-5), "Idempotency failed for g2"


# 5. 3 tasks: PCGrad processes all tasks and returns a list of length 3;
#    verify that purely anti-parallel pairs among the three are resolved
#    (tasks that only conflict with one other task) and that outputs are
#    non-trivially different from the inputs when conflicts exist.
def test_pcgrad_project_three_tasks_all_resolved():
    """With 3 tasks where tasks 0 and 1 are anti-parallel and task 2 is orthogonal,
    the projected g0 and g1 should have non-negative dot product with each other."""
    projector = PCGradProjector()
    # g0 and g1 are anti-parallel; g2 is orthogonal to both
    g0 = torch.tensor([1.0, 0.0, 0.0])
    g1 = torch.tensor([-1.0, 0.0, 0.0])
    g2 = torch.tensor([0.0, 1.0, 0.0])
    result = projector.project([g0, g1, g2])
    assert len(result) == 3, "Should return 3 projected gradients"
    # g0 vs g1 conflict should be resolved: both should project to near-zero
    # in the conflicting dimension and their dot should be >= 0
    dot_01 = torch.dot(result[0].flatten(), result[1].flatten()).item()
    assert dot_01 >= -1e-5, f"Conflict between tasks 0 and 1 should be resolved: dot={dot_01}"
    # g2 is orthogonal to both; it should be unchanged (no conflicts to resolve)
    assert torch.allclose(result[2], g2, atol=1e-6), "g2 (orthogonal) should remain unchanged"


# 6. project_params writes to param.grad
def test_pcgrad_project_params_writes_grad():
    """project_params writes results to param.grad (not None)."""
    projector = PCGradProjector()
    torch.manual_seed(0)
    params = [nn.Parameter(torch.randn(4, 4)), nn.Parameter(torch.randn(4))]
    per_task_grads = [
        [torch.randn_like(p) for p in params],
        [torch.randn_like(p) for p in params],
    ]
    projector.project_params(per_task_grads, params)
    for p in params:
        assert p.grad is not None, "param.grad should not be None after project_params"


# 7. project_params: param.grad shape matches param shape
def test_pcgrad_project_params_grad_shape():
    """project_params produces param.grad with the same shape as each param."""
    projector = PCGradProjector()
    torch.manual_seed(1)
    params = [nn.Parameter(torch.randn(8, 8)), nn.Parameter(torch.randn(8))]
    per_task_grads = [
        [torch.randn_like(p) for p in params],
        [torch.randn_like(p) for p in params],
        [torch.randn_like(p) for p in params],
    ]
    projector.project_params(per_task_grads, params)
    for p in params:
        assert p.grad.shape == p.shape, f"param.grad shape {p.grad.shape} != param shape {p.shape}"


# 8. PCGradOptimizer.pc_step runs without error (simple linear model, 2 tasks)
def test_pcgrad_optimizer_pc_step_runs():
    """PCGradOptimizer.pc_step completes without raising an exception."""
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    projector = PCGradProjector()
    pc_opt = PCGradOptimizer(base_opt, projector)

    x = torch.randn(3, 4)
    out = model(x)
    loss1 = out[:, 0].mean()
    loss2 = out[:, 1].mean()

    pc_opt.pc_step([loss1, loss2], list(model.parameters()))


# 9. Model params updated after pc_step
def test_pcgrad_optimizer_params_updated():
    """Model parameters change after pc_step is called."""
    torch.manual_seed(7)
    model = nn.Linear(4, 2)
    params_before = [p.data.clone() for p in model.parameters()]

    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    projector = PCGradProjector()
    pc_opt = PCGradOptimizer(base_opt, projector)

    x = torch.randn(5, 4)
    out = model(x)
    loss1 = (out[:, 0] ** 2).mean()
    loss2 = (out[:, 1] ** 2).mean()

    pc_opt.pc_step([loss1, loss2], list(model.parameters()))

    for before, after in zip(params_before, model.parameters()):
        assert not torch.allclose(before, after.data), "Params should have been updated"


# 10. pc_step with identical task losses: no conflict, normal gradient
def test_pcgrad_optimizer_identical_losses_no_conflict():
    """With identical losses, there are no gradient conflicts and pc_step runs."""
    torch.manual_seed(3)
    model = nn.Linear(4, 1)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    projector = PCGradProjector()
    pc_opt = PCGradOptimizer(base_opt, projector)

    x = torch.randn(5, 4)
    out = model(x)
    loss = out.mean()
    # Two identical losses -> identical gradients -> no conflict
    pc_opt.pc_step([loss, loss], list(model.parameters()))


# ---------------------------------------------------------------------------
# TaskLossAggregator tests
# ---------------------------------------------------------------------------


# 11. aggregate with uniform weights = simple mean
def test_task_loss_aggregator_uniform_mean():
    """TaskLossAggregator.aggregate with no weights returns the simple mean."""
    agg = TaskLossAggregator()
    losses = [torch.tensor(2.0), torch.tensor(4.0), torch.tensor(6.0)]
    result = agg.aggregate(losses)
    expected = torch.tensor(4.0)
    assert torch.allclose(result, expected, atol=1e-6), (
        f"Uniform-weight mean should be 4.0, got {result.item()}"
    )


# 12. Custom weights: weighted sum correct
def test_task_loss_aggregator_custom_weights():
    """TaskLossAggregator.aggregate respects custom weights."""
    weights = [0.5, 0.3, 0.2]
    agg = TaskLossAggregator(weights=weights)
    losses = [torch.tensor(10.0), torch.tensor(20.0), torch.tensor(30.0)]
    result = agg.aggregate(losses)
    expected = 0.5 * 10.0 + 0.3 * 20.0 + 0.2 * 30.0  # = 17.0
    assert abs(result.item() - expected) < 1e-5, (
        f"Weighted sum should be {expected}, got {result.item()}"
    )


# 13. gradient_conflict_ratio for all-same grads = 0.0
def test_task_loss_aggregator_conflict_ratio_no_conflict():
    """gradient_conflict_ratio returns 0.0 when all gradients point the same direction."""
    agg = TaskLossAggregator()
    g = torch.tensor([1.0, 2.0, 3.0])
    grads = [g.clone() for _ in range(4)]
    ratio = agg.gradient_conflict_ratio(grads)
    assert abs(ratio) < 1e-9, f"Expected 0.0, got {ratio}"


# 14. gradient_conflict_ratio for all-opposite grads = 1.0
def test_task_loss_aggregator_conflict_ratio_all_conflict():
    """gradient_conflict_ratio returns 1.0 when every pair of gradients conflicts."""
    agg = TaskLossAggregator()
    g = torch.tensor([1.0, 0.0])
    # Two perfectly opposing gradients: every pair conflicts
    grads = [g, -g]
    ratio = agg.gradient_conflict_ratio(grads)
    assert abs(ratio - 1.0) < 1e-9, f"Expected 1.0, got {ratio}"


# 15. gradient_conflict_ratio output in [0, 1]
def test_task_loss_aggregator_conflict_ratio_bounds():
    """gradient_conflict_ratio always returns a value in [0, 1]."""
    torch.manual_seed(99)
    agg = TaskLossAggregator()
    grads = [torch.randn(20) for _ in range(5)]
    ratio = agg.gradient_conflict_ratio(grads)
    assert 0.0 <= ratio <= 1.0, f"Conflict ratio {ratio} out of [0, 1]"
