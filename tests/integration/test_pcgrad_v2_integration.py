"""
Integration test for PCGrad v2.

Verifies end-to-end flow:
  - 2-task setup with guaranteed conflicting gradients on a linear model
  - PCGradV2.apply_to_optimizer runs a full step
  - Resolved gradients have better (less negative) dot product than originals
  - TRAINING_REGISTRY["pcgrad_v2"] is wired to PCGradV2
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training import TRAINING_REGISTRY
from src.training.pcgrad_v2 import PCGradV2, PCGradV2Config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_conflicting_grads(n_params: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two gradient tensors that are guaranteed to conflict (cos < 0)."""
    torch.manual_seed(seed)
    g = torch.randn(n_params)
    return g, -g + 0.01 * torch.randn(n_params)  # near-opposing


# ---------------------------------------------------------------------------
# Integration: 2-task conflicting gradient resolution on a linear model
# ---------------------------------------------------------------------------


def test_pcgrad_v2_resolves_conflicting_gradients():
    """PCGradV2 should improve dot product for conflicting gradient pairs."""
    torch.manual_seed(1)
    model = nn.Linear(8, 4)
    cfg = PCGradV2Config(n_tasks=2, adaptive_weight=True, alpha=1.0)
    pcg = PCGradV2(cfg)

    x = torch.randn(5, 8)
    # Task A: minimise sum of outputs; Task B: maximise sum → direct conflict
    loss_a = model(x).sum()
    loss_b = -model(x).sum()

    # Capture raw per-task gradients before the optimizer step
    params = list(model.parameters())

    raw_grads: list[list[torch.Tensor]] = []
    for loss in [loss_a, loss_b]:
        model.zero_grad()
        loss.backward(retain_graph=True)
        raw_grads.append(
            [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        )

    # Confirm the raw gradients are conflicting for at least one parameter
    has_conflict = any(
        (raw_grads[0][p].reshape(-1) @ raw_grads[1][p].reshape(-1)).item() < 0
        for p in range(len(params))
    )
    assert has_conflict, "Test setup: expected at least one conflicting parameter"

    # Resolve
    resolved = pcg.resolve(raw_grads)

    # For every parameter: dot product of resolved pair should be >= raw pair
    for p in range(len(params)):
        g0_flat = raw_grads[0][p].reshape(-1)
        g1_flat = raw_grads[1][p].reshape(-1)
        r0_flat = resolved[0][p].reshape(-1)
        r1_flat = resolved[1][p].reshape(-1)
        raw_dot = (g0_flat @ g1_flat).item()
        resolved_dot = (r0_flat @ r1_flat).item()
        assert resolved_dot >= raw_dot - 1e-5, (
            f"param {p}: resolved_dot={resolved_dot:.4f} < raw_dot={raw_dot:.4f}"
        )


def test_pcgrad_v2_optimizer_step_reduces_n_conflicts():
    """apply_to_optimizer should detect conflicts and return correct stats."""
    torch.manual_seed(2)
    model = nn.Linear(6, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    cfg = PCGradV2Config(n_tasks=2, adaptive_weight=True, alpha=1.0)
    pcg = PCGradV2(cfg)

    x = torch.randn(4, 6)
    loss_a = model(x).sum()
    loss_b = -model(x).sum()

    result = pcg.apply_to_optimizer(optimizer, [loss_a, loss_b])

    assert "n_conflicts" in result
    assert "mean_conflict_score" in result
    # For directly opposing tasks we expect at least one conflict detected
    assert result["n_conflicts"] > 0
    # mean score should be < 0 (conflicting tasks)
    assert result["mean_conflict_score"] < 0.0


def test_pcgrad_v2_gradient_bank_multi_batch():
    """GradientBank correctly accumulates and averages gradients."""
    from src.training.pcgrad_v2 import GradientBank

    bank = GradientBank(n_tasks=2, bank_size=3)
    grads_0 = [torch.ones(4)]
    grads_1 = [torch.ones(4) * 2.0]

    full = False
    for _ in range(3):
        full = bank.add(0, grads_0) or full
        full = bank.add(1, grads_1) or full

    assert full, "Bank should be full after 3 batches per task"

    accumulated = bank.get_accumulated()
    assert len(accumulated) == 2
    # Task 0 average should be 1.0
    assert torch.allclose(accumulated[0][0], torch.ones(4), atol=1e-6)
    # Task 1 average should be 2.0
    assert torch.allclose(accumulated[1][0], torch.ones(4) * 2.0, atol=1e-6)

    bank.clear()
    assert len(bank._storage[0]) == 0


def test_registry_wired():
    """TRAINING_REGISTRY['pcgrad_v2'] must point to PCGradV2."""
    assert "pcgrad_v2" in TRAINING_REGISTRY
    assert TRAINING_REGISTRY["pcgrad_v2"] is PCGradV2

    # Instantiate from registry — should work
    cls = TRAINING_REGISTRY["pcgrad_v2"]
    instance = cls()
    assert isinstance(instance, PCGradV2)
