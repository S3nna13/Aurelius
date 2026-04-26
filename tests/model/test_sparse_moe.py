"""Tests for sparse_moe.py."""

from __future__ import annotations

import pytest
import torch

from src.model.sparse_moe import (
    ExpertFFN,
    SparseMoEConfig,
    SparseMoELayer,
    TokenRouter,
    combine_expert_outputs,
    compute_capacity,
    dispatch_tokens,
)

B, T, D = 2, 8, 32
N_EXPERTS = 4
N_ACTIVE = 2
D_FF = 64


@pytest.fixture()
def cfg() -> SparseMoEConfig:
    return SparseMoEConfig(
        d_model=D,
        n_experts=N_EXPERTS,
        n_active=N_ACTIVE,
        expert_d_ff=D_FF,
        capacity_factor=1.25,
        aux_loss_coeff=0.01,
        expert_dropout=0.0,
        jitter_noise=0.0,
    )


@pytest.fixture()
def layer(cfg: SparseMoEConfig) -> SparseMoELayer:
    return SparseMoELayer(cfg)


@pytest.fixture()
def x() -> torch.Tensor:
    return torch.randn(B, T, D)


# 1. SparseMoEConfig defaults
def test_config_defaults() -> None:
    cfg = SparseMoEConfig()
    assert cfg.d_model == 512
    assert cfg.n_experts == 8
    assert cfg.n_active == 2
    assert cfg.capacity_factor == 1.25
    assert cfg.expert_d_ff == 2048
    assert cfg.aux_loss_coeff == 0.01
    assert cfg.expert_dropout == 0.0
    assert cfg.jitter_noise == 0.0


# 2. ExpertFFN output shape
def test_expert_ffn_output_shape() -> None:
    expert = ExpertFFN(d_model=D, d_ff=D_FF)
    inp = torch.randn(B * T, D)
    out = expert(inp)
    assert out.shape == (B * T, D)


# 3. ExpertFFN with dropout=0 is deterministic in eval mode
def test_expert_ffn_deterministic_eval() -> None:
    expert = ExpertFFN(d_model=D, d_ff=D_FF, dropout=0.0)
    expert.eval()
    inp = torch.randn(B * T, D)
    out1 = expert(inp)
    out2 = expert(inp)
    assert torch.allclose(out1, out2)


# 4. TokenRouter output shapes correct
def test_token_router_output_shapes() -> None:
    router = TokenRouter(d_model=D, n_experts=N_EXPERTS, n_active=N_ACTIVE)
    inp = torch.randn(B, T, D)
    router_probs, top_k_indices, top_k_weights = router(inp)
    assert router_probs.shape == (B * T, N_EXPERTS)
    assert top_k_indices.shape == (B * T, N_ACTIVE)
    assert top_k_weights.shape == (B * T, N_ACTIVE)


# 5. TokenRouter probabilities sum to 1
def test_token_router_probs_sum_to_one() -> None:
    router = TokenRouter(d_model=D, n_experts=N_EXPERTS, n_active=N_ACTIVE)
    inp = torch.randn(B, T, D)
    router_probs, _, _ = router(inp)
    row_sums = router_probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# 6. compute_capacity returns positive int
def test_compute_capacity_positive() -> None:
    cap = compute_capacity(T=T, n_experts=N_EXPERTS, n_active=N_ACTIVE, capacity_factor=1.25)
    assert isinstance(cap, int)
    assert cap >= 1


# 7. compute_capacity increases with higher capacity_factor
def test_compute_capacity_increases_with_factor() -> None:
    cap_low = compute_capacity(T=T, n_experts=N_EXPERTS, n_active=N_ACTIVE, capacity_factor=1.0)
    cap_high = compute_capacity(T=T, n_experts=N_EXPERTS, n_active=N_ACTIVE, capacity_factor=2.0)
    assert cap_high >= cap_low


# 8. dispatch_tokens output shapes
def test_dispatch_tokens_output_shapes() -> None:
    N = B * T
    capacity = compute_capacity(T=T, n_experts=N_EXPERTS, n_active=N_ACTIVE, capacity_factor=1.25)
    x_flat = torch.randn(N, D)
    x_3d = torch.randn(B, T, D)
    router = TokenRouter(d_model=D, n_experts=N_EXPERTS, n_active=N_ACTIVE)
    _, indices, weights = router(x_3d)
    expert_inputs, dispatch_mask = dispatch_tokens(x_flat, indices, weights, N_EXPERTS, capacity)
    assert expert_inputs.shape == (N_EXPERTS, capacity, D)
    assert dispatch_mask.shape == (N, N_ACTIVE)
    assert dispatch_mask.dtype == torch.bool


# 9. combine_expert_outputs output shape
def test_combine_expert_outputs_shape() -> None:
    N = B * T
    capacity = compute_capacity(T=T, n_experts=N_EXPERTS, n_active=N_ACTIVE, capacity_factor=1.25)
    x_flat = torch.randn(N, D)
    x_3d = torch.randn(B, T, D)
    router = TokenRouter(d_model=D, n_experts=N_EXPERTS, n_active=N_ACTIVE)
    _, indices, weights = router(x_3d)
    expert_inputs, dispatch_mask = dispatch_tokens(x_flat, indices, weights, N_EXPERTS, capacity)
    expert_outputs = torch.randn_like(expert_inputs)
    output = combine_expert_outputs(expert_outputs, indices, weights, dispatch_mask, N)
    assert output.shape == (N, D)


# 10. SparseMoELayer forward output shape
def test_sparse_moe_layer_output_shape(layer: SparseMoELayer, x: torch.Tensor) -> None:
    output, _ = layer(x)
    assert output.shape == (B, T, D)


# 11. SparseMoELayer returns scalar aux_loss
def test_sparse_moe_layer_aux_loss_scalar(layer: SparseMoELayer, x: torch.Tensor) -> None:
    _, aux_loss = layer(x)
    assert aux_loss.ndim == 0


# 12. SparseMoELayer aux_loss >= 0
def test_sparse_moe_layer_aux_loss_nonneg(layer: SparseMoELayer, x: torch.Tensor) -> None:
    _, aux_loss = layer(x)
    assert float(aux_loss.detach()) >= 0.0


# 13. SparseMoELayer get_routing_stats returns correct keys
def test_sparse_moe_layer_routing_stats_keys(layer: SparseMoELayer, x: torch.Tensor) -> None:
    layer(x)
    stats = layer.get_routing_stats()
    expected_keys = {"mean_expert_load", "max_expert_load", "token_drop_rate"}
    assert set(stats.keys()) == expected_keys
    for val in stats.values():
        assert isinstance(val, float)


# 14. SparseMoELayer gradients flow through
def test_sparse_moe_layer_gradients(layer: SparseMoELayer) -> None:
    x = torch.randn(B, T, D, requires_grad=True)
    output, aux_loss = layer(x)
    loss = output.sum() + aux_loss
    loss.backward()
    assert x.grad is not None
    assert layer.router.router.weight.grad is not None
