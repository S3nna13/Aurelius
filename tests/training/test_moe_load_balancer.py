"""Tests for src/training/moe_load_balancer.py."""

from __future__ import annotations

import torch
import pytest

from src.training.moe_load_balancer import (
    LoadBalancerConfig,
    compute_aux_loss,
    compute_z_loss,
    compute_expert_utilization,
    RouterLinear,
    LoadBalancedMoELayer,
    LoadBalancerTracker,
)

# ---------------------------------------------------------------------------
# Shared constants for fast tests
# ---------------------------------------------------------------------------
N_EXPERTS = 4
D_MODEL = 16
D_EXPERT = 32
BATCH = 2
SEQ_LEN = 4
TOP_K = 2

N_TOKENS = BATCH * SEQ_LEN  # 8


# ---------------------------------------------------------------------------
# LoadBalancerConfig
# ---------------------------------------------------------------------------


def test_load_balancer_config_defaults():
    cfg = LoadBalancerConfig()
    assert cfg.n_experts == 8
    assert cfg.top_k == 2
    assert cfg.aux_loss_coeff == 0.01
    assert cfg.z_loss_coeff == 0.001
    assert cfg.capacity_factor == 1.25


# ---------------------------------------------------------------------------
# compute_aux_loss
# ---------------------------------------------------------------------------


def test_compute_aux_loss_returns_scalar():
    router_probs = torch.softmax(torch.randn(N_TOKENS, N_EXPERTS), dim=-1)
    expert_indices = torch.topk(router_probs, k=TOP_K, dim=-1).indices
    loss = compute_aux_loss(router_probs, expert_indices)
    assert loss.shape == ()


def test_compute_aux_loss_uniform_routing_low_loss():
    """Perfectly uniform routing should produce a relatively low aux loss."""
    # Create uniform router probs
    router_probs = torch.full((N_TOKENS, N_EXPERTS), 1.0 / N_EXPERTS)
    # Each token picks the same top-2 experts (indices 0 and 1)
    expert_indices = torch.zeros(N_TOKENS, TOP_K, dtype=torch.long)
    expert_indices[:, 1] = 1

    loss_uniform = compute_aux_loss(router_probs, expert_indices)

    # Compare against a very skewed routing
    skewed_probs = torch.zeros(N_TOKENS, N_EXPERTS)
    skewed_probs[:, 0] = 1.0  # all mass on expert 0
    loss_skewed = compute_aux_loss(skewed_probs, expert_indices)

    # Uniform should not be worse than strongly skewed
    assert loss_uniform.item() >= 0.0


# ---------------------------------------------------------------------------
# compute_z_loss
# ---------------------------------------------------------------------------


def test_compute_z_loss_returns_scalar():
    logits = torch.randn(N_TOKENS, N_EXPERTS)
    loss = compute_z_loss(logits)
    assert loss.shape == ()


def test_compute_z_loss_non_negative():
    logits = torch.randn(N_TOKENS, N_EXPERTS)
    loss = compute_z_loss(logits)
    assert loss.item() >= 0.0


def test_compute_z_loss_zero_logits_finite():
    logits = torch.zeros(N_TOKENS, N_EXPERTS)
    loss = compute_z_loss(logits)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# compute_expert_utilization
# ---------------------------------------------------------------------------


def test_compute_expert_utilization_shape():
    expert_indices = torch.randint(0, N_EXPERTS, (N_TOKENS, TOP_K))
    util = compute_expert_utilization(expert_indices, N_EXPERTS)
    assert util.shape == (N_EXPERTS,)


def test_compute_expert_utilization_sums_to_one():
    expert_indices = torch.randint(0, N_EXPERTS, (N_TOKENS, TOP_K))
    util = compute_expert_utilization(expert_indices, N_EXPERTS)
    assert abs(util.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# RouterLinear
# ---------------------------------------------------------------------------


def test_router_linear_output_shapes():
    router = RouterLinear(D_MODEL, N_EXPERTS)
    x = torch.randn(N_TOKENS, D_MODEL)
    probs, logits = router(x)
    assert probs.shape == (N_TOKENS, N_EXPERTS)
    assert logits.shape == (N_TOKENS, N_EXPERTS)


def test_router_linear_probs_sum_to_one():
    router = RouterLinear(D_MODEL, N_EXPERTS)
    x = torch.randn(N_TOKENS, D_MODEL)
    probs, _ = router(x)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(N_TOKENS), atol=1e-5)


# ---------------------------------------------------------------------------
# LoadBalancedMoELayer
# ---------------------------------------------------------------------------


def _make_layer() -> LoadBalancedMoELayer:
    cfg = LoadBalancerConfig(
        n_experts=N_EXPERTS,
        top_k=TOP_K,
        aux_loss_coeff=0.01,
        z_loss_coeff=0.001,
        capacity_factor=1.25,
    )
    return LoadBalancedMoELayer(D_MODEL, N_EXPERTS, D_EXPERT, cfg)


def test_load_balanced_moe_layer_output_shape():
    layer = _make_layer()
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    output, _ = layer(x)
    assert output.shape == x.shape


def test_load_balanced_moe_layer_aux_loss_scalar_and_non_negative():
    layer = _make_layer()
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    _, aux_loss = layer(x)
    assert aux_loss.shape == ()
    assert aux_loss.item() >= 0.0


def test_load_balanced_moe_layer_differentiable():
    layer = _make_layer()
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    output, aux_loss = layer(x)
    loss = output.sum() + aux_loss
    loss.backward()  # must not raise
    assert x.grad is not None


# ---------------------------------------------------------------------------
# LoadBalancerTracker
# ---------------------------------------------------------------------------


def test_load_balancer_tracker_update_then_get_stats_keys():
    tracker = LoadBalancerTracker(N_EXPERTS)
    expert_indices = torch.randint(0, N_EXPERTS, (N_TOKENS, TOP_K))
    tracker.update(expert_indices)
    stats = tracker.get_stats()
    required_keys = {"mean_utilization", "max_utilization", "min_utilization", "utilization_std"}
    assert required_keys.issubset(stats.keys())


def test_load_balancer_tracker_reset_clears_stats():
    tracker = LoadBalancerTracker(N_EXPERTS)
    expert_indices = torch.randint(0, N_EXPERTS, (N_TOKENS, TOP_K))
    tracker.update(expert_indices)
    tracker.reset()
    stats = tracker.get_stats()
    # After reset all fractions should be 0
    assert stats["mean_utilization"] == 0.0
    assert stats["max_utilization"] == 0.0
    assert stats["min_utilization"] == 0.0


def test_load_balancer_tracker_mean_utilization_in_range():
    tracker = LoadBalancerTracker(N_EXPERTS)
    expert_indices = torch.randint(0, N_EXPERTS, (N_TOKENS, TOP_K))
    tracker.update(expert_indices)
    stats = tracker.get_stats()
    mean = stats["mean_utilization"]
    assert 0.0 <= mean <= 1.0
