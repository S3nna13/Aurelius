"""Tests for new-style MoE with load-balance auxiliary loss.

Covers MoEBalancedConfig, ExpertLayer, RouterWithLoadBalancing,
MoEBalancedLayer, MoEBalancedTransformer, and compute_expert_utilization.
"""
from __future__ import annotations

import math
import torch
import pytest

from src.model.moe_balanced import (
    MoEBalancedConfig,
    ExpertLayer,
    RouterWithLoadBalancing,
    MoEBalancedLayer,
    MoEBalancedTransformer,
    compute_expert_utilization,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cfg() -> MoEBalancedConfig:
    return MoEBalancedConfig(
        n_experts=8,
        top_k=2,
        d_model=64,
        d_ff=256,
        capacity_factor=1.25,
        load_balance_coef=1e-2,
    )


@pytest.fixture()
def x(cfg) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 16, cfg.d_model)


@pytest.fixture()
def x_flat(cfg) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(32, cfg.d_model)  # (N=32, d_model)


# ---------------------------------------------------------------------------
# 1. MoEBalancedConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = MoEBalancedConfig()
    assert cfg.n_experts == 8
    assert cfg.top_k == 2
    assert cfg.d_model == 64
    assert cfg.d_ff == 256
    assert cfg.capacity_factor == 1.25
    assert cfg.load_balance_coef == 1e-2


def test_config_custom():
    cfg = MoEBalancedConfig(n_experts=4, top_k=1, d_model=128, d_ff=512)
    assert cfg.n_experts == 4
    assert cfg.top_k == 1
    assert cfg.d_model == 128
    assert cfg.d_ff == 512


# ---------------------------------------------------------------------------
# 2. ExpertLayer
# ---------------------------------------------------------------------------

def test_expert_layer_output_shape(cfg):
    expert = ExpertLayer(cfg.d_model, cfg.d_ff)
    x = torch.randn(10, cfg.d_model)
    out = expert(x)
    assert out.shape == (10, cfg.d_model), f"Expected (10, {cfg.d_model}), got {out.shape}"


def test_expert_layer_batch_shape(cfg):
    expert = ExpertLayer(cfg.d_model, cfg.d_ff)
    x = torch.randn(1, cfg.d_model)
    out = expert(x)
    assert out.shape == (1, cfg.d_model)


# ---------------------------------------------------------------------------
# 3. RouterWithLoadBalancing
# ---------------------------------------------------------------------------

def test_router_output_shapes(cfg, x_flat):
    router = RouterWithLoadBalancing(
        d_model=cfg.d_model,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        load_balance_coef=cfg.load_balance_coef,
    )
    weights, indices, loss = router(x_flat)
    N = x_flat.shape[0]
    assert weights.shape == (N, cfg.top_k), f"weights shape {weights.shape}"
    assert indices.shape == (N, cfg.top_k), f"indices shape {indices.shape}"
    assert loss.shape == torch.Size([]), f"loss should be scalar, got {loss.shape}"


def test_router_weights_sum_to_one(cfg, x_flat):
    router = RouterWithLoadBalancing(
        d_model=cfg.d_model,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        load_balance_coef=cfg.load_balance_coef,
    )
    weights, _, _ = router(x_flat)
    sums = weights.sum(dim=-1)  # (N,)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"Routing weights should sum to 1 per token. Got: {sums[:5]}"
    )


def test_router_indices_in_range(cfg, x_flat):
    router = RouterWithLoadBalancing(
        d_model=cfg.d_model,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        load_balance_coef=cfg.load_balance_coef,
    )
    _, indices, _ = router(x_flat)
    assert indices.min() >= 0
    assert indices.max() < cfg.n_experts


def test_router_load_balance_loss_nonneg(cfg, x_flat):
    router = RouterWithLoadBalancing(
        d_model=cfg.d_model,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        load_balance_coef=cfg.load_balance_coef,
    )
    _, _, loss = router(x_flat)
    assert loss.item() >= 0.0, f"load_balance_loss should be non-negative, got {loss.item()}"


def test_router_loss_is_scalar_tensor(cfg, x_flat):
    router = RouterWithLoadBalancing(
        d_model=cfg.d_model,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        load_balance_coef=cfg.load_balance_coef,
    )
    _, _, loss = router(x_flat)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, "load_balance_loss must be a 0-d scalar tensor"


# ---------------------------------------------------------------------------
# 4. MoEBalancedLayer
# ---------------------------------------------------------------------------

def test_moe_layer_output_shape(cfg, x):
    layer = MoEBalancedLayer(cfg)
    output, aux_loss = layer(x)
    assert output.shape == x.shape, (
        f"Output shape {output.shape} != input shape {x.shape}"
    )


def test_moe_layer_aux_loss_nonneg(cfg, x):
    layer = MoEBalancedLayer(cfg)
    _, aux_loss = layer(x)
    assert aux_loss.item() >= 0.0, f"aux_loss should be non-negative, got {aux_loss.item()}"


def test_moe_layer_aux_loss_is_scalar(cfg, x):
    layer = MoEBalancedLayer(cfg)
    _, aux_loss = layer(x)
    assert aux_loss.shape == torch.Size([]), (
        f"aux_loss should be scalar, got shape {aux_loss.shape}"
    )


def test_moe_layer_gradient_flows(cfg, x):
    layer = MoEBalancedLayer(cfg)
    x_req = x.clone().requires_grad_(True)
    output, aux_loss = layer(x_req)
    total = output.sum() + aux_loss
    total.backward()
    assert x_req.grad is not None, "Gradient should flow back through MoEBalancedLayer"
    assert not torch.all(x_req.grad == 0), "Gradient should be non-zero"


# ---------------------------------------------------------------------------
# 5. MoEBalancedTransformer
# ---------------------------------------------------------------------------

def test_transformer_output_shape(cfg, x):
    model = MoEBalancedTransformer(cfg, n_layers=2)
    output, total_aux = model(x)
    assert output.shape == x.shape
    assert total_aux.shape == torch.Size([])


def test_transformer_total_aux_loss_nonneg(cfg, x):
    model = MoEBalancedTransformer(cfg, n_layers=3)
    _, total_aux = model(x)
    assert total_aux.item() >= 0.0


# ---------------------------------------------------------------------------
# 6. compute_expert_utilization
# ---------------------------------------------------------------------------

def test_utilization_per_expert_load_sums_to_one(cfg):
    torch.manual_seed(42)
    indices = torch.randint(0, cfg.n_experts, (100, cfg.top_k))
    result = compute_expert_utilization(indices, cfg.n_experts)
    total = sum(result["per_expert_load"])
    assert abs(total - 1.0) < 1e-6, f"per_expert_load should sum to 1.0, got {total}"


def test_utilization_load_imbalance_gte_one(cfg):
    torch.manual_seed(7)
    indices = torch.randint(0, cfg.n_experts, (200, cfg.top_k))
    result = compute_expert_utilization(indices, cfg.n_experts)
    assert result["load_imbalance"] >= 1.0, (
        f"load_imbalance should be >= 1.0, got {result['load_imbalance']}"
    )


def test_utilization_perfect_balance(cfg):
    """When all experts get exactly equal load, imbalance == 1.0."""
    n_experts = cfg.n_experts
    top_k = cfg.top_k
    # Construct perfectly balanced routing: cycle through experts
    N = n_experts * 10
    flat = torch.arange(N * top_k) % n_experts
    indices = flat.view(N, top_k)
    result = compute_expert_utilization(indices, n_experts)
    assert abs(result["load_imbalance"] - 1.0) < 1e-5, (
        f"Expected load_imbalance=1.0 for perfectly balanced routing, "
        f"got {result['load_imbalance']}"
    )


def test_utilization_entropy_is_finite(cfg):
    torch.manual_seed(3)
    indices = torch.randint(0, cfg.n_experts, (50, cfg.top_k))
    result = compute_expert_utilization(indices, cfg.n_experts)
    assert math.isfinite(result["entropy"]), "entropy should be a finite number"


def test_utilization_returns_correct_keys(cfg):
    indices = torch.randint(0, cfg.n_experts, (32, cfg.top_k))
    result = compute_expert_utilization(indices, cfg.n_experts)
    assert "per_expert_load" in result
    assert "load_imbalance" in result
    assert "entropy" in result


def test_utilization_per_expert_load_length(cfg):
    indices = torch.randint(0, cfg.n_experts, (32, cfg.top_k))
    result = compute_expert_utilization(indices, cfg.n_experts)
    assert len(result["per_expert_load"]) == cfg.n_experts
