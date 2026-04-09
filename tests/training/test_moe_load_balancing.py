"""Tests for src/training/moe_load_balancing.py.

All tests use tiny dimensions (N=32 tokens, n_experts=4, top_k=2) so they
run fast on CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.moe_load_balancing import (
    LoadBalancingConfig,
    LoadBalancedMoELayer,
    MoELoadBalancer,
    combine_expert_outputs,
    compute_aux_load_balance_loss,
    compute_expert_utilization,
    compute_router_z_loss,
    top_k_routing,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

N = 32       # number of tokens (B*T)
E = 4        # n_experts
K = 2        # top_k
D = 16       # d_model
B = 4        # batch size
T = 8        # sequence length  (B * T == N)


def _uniform_logits() -> torch.Tensor:
    """(N, E) logits that produce uniform router probabilities."""
    return torch.zeros(N, E)


def _uniform_mask() -> torch.Tensor:
    """(N, E) mask where each token is assigned to exactly two experts in a
    round-robin fashion so that every expert gets the same total count."""
    mask = torch.zeros(N, E)
    for i in range(N):
        mask[i, i % E] = 1.0
        mask[i, (i + 1) % E] = 1.0
    return mask


def _single_expert_mask() -> torch.Tensor:
    """(N, E) mask where all tokens are routed only to expert 0."""
    mask = torch.zeros(N, E)
    mask[:, 0] = 1.0
    return mask


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = LoadBalancingConfig()
    assert cfg.n_experts == 8
    assert cfg.aux_loss_coeff == pytest.approx(0.01)
    assert cfg.z_loss_coeff == pytest.approx(0.001)
    assert cfg.top_k == 2
    assert cfg.capacity_factor == pytest.approx(1.25)
    assert cfg.jitter_eps == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. test_compute_router_z_loss_shape
# ---------------------------------------------------------------------------

def test_compute_router_z_loss_shape():
    logits = torch.randn(N, E)
    z = compute_router_z_loss(logits)
    assert z.ndim == 0, "z_loss should be a scalar"


# ---------------------------------------------------------------------------
# 3. test_compute_router_z_loss_nonneg
# ---------------------------------------------------------------------------

def test_compute_router_z_loss_nonneg():
    logits = torch.randn(N, E)
    z = compute_router_z_loss(logits)
    assert z.item() >= 0.0, "z_loss must be non-negative"


# ---------------------------------------------------------------------------
# 4. test_compute_aux_loss_balanced
# ---------------------------------------------------------------------------

def test_compute_aux_loss_balanced():
    """Perfectly balanced routing should give lower aux loss than all-to-one."""
    probs_uniform = torch.full((N, E), 1.0 / E)
    mask_balanced = _uniform_mask()
    mask_skewed = _single_expert_mask()

    loss_balanced = compute_aux_load_balance_loss(probs_uniform, mask_balanced)
    loss_skewed = compute_aux_load_balance_loss(probs_uniform, mask_skewed)

    # With uniform probs and uniform mask: loss = E * E * (1/E)^2 = 1.0
    # With skewed mask: all load on expert 0 → f_0 = 1 → loss = E * (1 * 1/E) = 1.0
    # Both equal 1.0 with uniform probs; use peaked probs to differentiate.
    peaked_probs = torch.zeros(N, E)
    peaked_probs[:, 0] = 1.0  # 100% probability on expert 0

    loss_balanced_peaked = compute_aux_load_balance_loss(peaked_probs, mask_balanced)
    loss_skewed_peaked = compute_aux_load_balance_loss(peaked_probs, mask_skewed)

    assert loss_balanced_peaked.item() < loss_skewed_peaked.item(), (
        "Balanced routing should have lower aux loss than skewed routing "
        "when router probabilities are peaked on one expert"
    )


# ---------------------------------------------------------------------------
# 5. test_compute_aux_loss_shape
# ---------------------------------------------------------------------------

def test_compute_aux_loss_shape():
    probs = torch.full((N, E), 1.0 / E)
    mask = _uniform_mask()
    loss = compute_aux_load_balance_loss(probs, mask)
    assert loss.ndim == 0, "aux_loss should be a scalar"


# ---------------------------------------------------------------------------
# 6. test_compute_expert_utilization_keys
# ---------------------------------------------------------------------------

def test_compute_expert_utilization_keys():
    mask = _uniform_mask()
    result = compute_expert_utilization(mask)
    assert set(result.keys()) == {
        "utilization_std", "min_utilization", "max_utilization", "cv"
    }


# ---------------------------------------------------------------------------
# 7. test_compute_expert_utilization_balanced
# ---------------------------------------------------------------------------

def test_compute_expert_utilization_balanced():
    """Uniform routing should give near-zero coefficient of variation."""
    mask = _uniform_mask()
    result = compute_expert_utilization(mask)
    assert result["cv"] == pytest.approx(0.0, abs=1e-5), (
        "Perfectly balanced routing should have cv ≈ 0"
    )


# ---------------------------------------------------------------------------
# 8. test_top_k_routing_shapes
# ---------------------------------------------------------------------------

def test_top_k_routing_shapes():
    logits = torch.randn(N, E)
    router_probs, expert_mask, top_k_indices = top_k_routing(logits, top_k=K)

    assert router_probs.shape == (N, E), f"Expected ({N}, {E}), got {router_probs.shape}"
    assert expert_mask.shape == (N, E), f"Expected ({N}, {E}), got {expert_mask.shape}"
    assert top_k_indices.shape == (N, K), f"Expected ({N}, {K}), got {top_k_indices.shape}"


# ---------------------------------------------------------------------------
# 9. test_top_k_routing_probs_normalized
# ---------------------------------------------------------------------------

def test_top_k_routing_probs_normalized():
    """The full softmax router_probs should sum to ~1 per token."""
    logits = torch.randn(N, E)
    router_probs, _, _ = top_k_routing(logits, top_k=K)

    row_sums = router_probs.sum(dim=-1)  # (N,)
    assert torch.allclose(row_sums, torch.ones(N), atol=1e-5), (
        "router_probs should sum to 1 per token"
    )


# ---------------------------------------------------------------------------
# 10. test_top_k_routing_capacity_respected
# ---------------------------------------------------------------------------

def test_top_k_routing_capacity_respected():
    """No expert should receive more than capacity tokens."""
    logits = torch.randn(N, E)
    cfg = LoadBalancingConfig(n_experts=E, top_k=K, capacity_factor=1.25)
    capacity = max(1, int(N / E * cfg.capacity_factor * K))

    _, expert_mask, _ = top_k_routing(
        logits, top_k=K, capacity_factor=cfg.capacity_factor
    )

    per_expert_counts = expert_mask.sum(dim=0)  # (E,)
    for e in range(E):
        assert per_expert_counts[e].item() <= capacity + 1e-6, (
            f"Expert {e} received {per_expert_counts[e].item()} tokens "
            f"but capacity is {capacity}"
        )


# ---------------------------------------------------------------------------
# 11. test_moe_layer_output_shape
# ---------------------------------------------------------------------------

def test_moe_layer_output_shape():
    cfg = LoadBalancingConfig(n_experts=E, top_k=K)
    layer = LoadBalancedMoELayer(
        d_model=D, d_expert=D * 4, n_experts=E, top_k=K, cfg=cfg
    )
    x = torch.randn(B, T, D)
    output, _ = layer(x)
    assert output.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {output.shape}"


# ---------------------------------------------------------------------------
# 12. test_moe_layer_aux_loss_keys
# ---------------------------------------------------------------------------

def test_moe_layer_aux_loss_keys():
    cfg = LoadBalancingConfig(n_experts=E, top_k=K)
    layer = LoadBalancedMoELayer(
        d_model=D, d_expert=D * 4, n_experts=E, top_k=K, cfg=cfg
    )
    x = torch.randn(B, T, D)
    _, aux_losses = layer(x)
    assert set(aux_losses.keys()) == {"aux_loss", "z_loss", "total_aux"}


# ---------------------------------------------------------------------------
# 13. test_moe_layer_aux_loss_positive
# ---------------------------------------------------------------------------

def test_moe_layer_aux_loss_positive():
    cfg = LoadBalancingConfig(n_experts=E, top_k=K)
    layer = LoadBalancedMoELayer(
        d_model=D, d_expert=D * 4, n_experts=E, top_k=K, cfg=cfg
    )
    x = torch.randn(B, T, D)
    _, aux_losses = layer(x)
    assert aux_losses["total_aux"].item() > 0.0, "total_aux should be positive"


# ---------------------------------------------------------------------------
# 14. test_load_balancer_combined_loss_keys
# ---------------------------------------------------------------------------

def test_load_balancer_combined_loss_keys():
    cfg = LoadBalancingConfig(n_experts=E, top_k=K)
    layer = LoadBalancedMoELayer(
        d_model=D, d_expert=D * 4, n_experts=E, top_k=K, cfg=cfg
    )
    x = torch.randn(B, T, D)
    _, aux_losses = layer(x)

    balancer = MoELoadBalancer(model=nn.Linear(D, D), cfg=cfg)
    task_loss = torch.tensor(2.5)
    total_loss, info = balancer.compute_combined_loss(task_loss, [aux_losses])

    assert set(info.keys()) == {"task_loss", "aux_loss", "z_loss", "total_loss"}


# ---------------------------------------------------------------------------
# 15. test_load_balancer_total_exceeds_task
# ---------------------------------------------------------------------------

def test_load_balancer_total_exceeds_task():
    """total_loss should be greater than task_loss when aux losses are positive."""
    cfg = LoadBalancingConfig(n_experts=E, top_k=K, aux_loss_coeff=0.01, z_loss_coeff=0.001)
    layer = LoadBalancedMoELayer(
        d_model=D, d_expert=D * 4, n_experts=E, top_k=K, cfg=cfg
    )
    x = torch.randn(B, T, D)
    _, aux_losses = layer(x)

    balancer = MoELoadBalancer(model=nn.Linear(D, D), cfg=cfg)
    task_loss = torch.tensor(2.5)
    total_loss, info = balancer.compute_combined_loss(task_loss, [aux_losses])

    assert info["total_loss"] > info["task_loss"], (
        f"total_loss ({info['total_loss']:.6f}) should exceed "
        f"task_loss ({info['task_loss']:.6f})"
    )
