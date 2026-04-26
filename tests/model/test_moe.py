"""Tests for Sparse Mixture-of-Experts layer (src/model/moe.py).

All imports use the public aurelius namespace as required.
"""

from __future__ import annotations

import torch
from aurelius.model.moe import (
    ExpertFFN,
    MoEBlock,
    RouterConfig,
    SparseMoELayer,
    TopKRouter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D_MODEL = 32
D_FF = 64
N_EXPERTS = 4
TOP_K = 2
N_HEADS = 4
B, T = 2, 8


def make_x(b=B, t=T, d=D_MODEL):
    return torch.randn(b, t, d)


# ---------------------------------------------------------------------------
# RouterConfig
# ---------------------------------------------------------------------------


def test_router_config_defaults():
    """RouterConfig fields default correctly."""
    cfg = RouterConfig(n_experts=8)
    assert cfg.n_experts == 8
    assert cfg.top_k == 2
    assert cfg.capacity_factor == 1.25
    assert cfg.jitter_noise == 0.0


def test_router_config_custom():
    """RouterConfig accepts custom values."""
    cfg = RouterConfig(n_experts=4, top_k=1, capacity_factor=2.0, jitter_noise=0.1)
    assert cfg.top_k == 1
    assert cfg.capacity_factor == 2.0
    assert cfg.jitter_noise == 0.1


# ---------------------------------------------------------------------------
# TopKRouter
# ---------------------------------------------------------------------------


def test_topk_router_output_shapes():
    """TopKRouter returns tensors of the correct shapes."""
    router = TopKRouter(D_MODEL, N_EXPERTS, TOP_K)
    x = make_x()
    weights, indices, loss = router(x)
    assert weights.shape == (B, T, TOP_K), f"weights shape {weights.shape}"
    assert indices.shape == (B, T, TOP_K), f"indices shape {indices.shape}"
    assert loss.ndim == 0, "router_loss must be a scalar"


def test_topk_router_indices_in_range():
    """dispatch_indices must be in [0, n_experts)."""
    router = TopKRouter(D_MODEL, N_EXPERTS, TOP_K)
    x = make_x()
    _, indices, _ = router(x)
    assert (indices >= 0).all(), "found negative expert index"
    assert (indices < N_EXPERTS).all(), "found expert index >= n_experts"


def test_topk_router_weights_leq_one():
    """dispatch_weights (softmax slice) must each be in (0, 1] and their sum
    per token must be <= 1 (since they are a sub-slice of a full softmax)."""
    router = TopKRouter(D_MODEL, N_EXPERTS, TOP_K)
    x = make_x()
    weights, _, _ = router(x)
    assert (weights > 0).all(), "weights must be positive"
    assert (weights <= 1.0 + 1e-6).all(), "softmax weights must be <= 1"
    weight_sum = weights.sum(dim=-1)  # (B, T)
    assert (weight_sum <= 1.0 + 1e-6).all(), "top-k weight sum per token must be <= 1"


def test_topk_router_loss_scalar_nonneg():
    """router_loss must be a non-negative scalar."""
    router = TopKRouter(D_MODEL, N_EXPERTS, TOP_K)
    x = make_x()
    _, _, loss = router(x)
    assert loss.ndim == 0, "router_loss must be a 0-dim scalar"
    assert loss.item() >= 0.0, f"router_loss must be >= 0, got {loss.item()}"
    assert torch.isfinite(loss), "router_loss must be finite"


def test_topk_router_jitter_noise_training():
    """Jitter noise is applied during training mode (outputs should differ)."""
    torch.manual_seed(0)
    router = TopKRouter(D_MODEL, N_EXPERTS, TOP_K, jitter_noise=1.0)
    router.train()
    x = make_x()
    w1, idx1, _ = router(x)
    w2, idx2, _ = router(x)
    # With large noise, weights should differ across two forward passes
    assert not torch.allclose(w1, w2), "jitter noise should produce different weights"


# ---------------------------------------------------------------------------
# ExpertFFN
# ---------------------------------------------------------------------------


def test_expert_ffn_output_shape():
    """ExpertFFN output must match input shape."""
    expert = ExpertFFN(D_MODEL, D_FF)
    x = make_x()  # (B, T, D_MODEL)
    out = expert(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"


def test_expert_ffn_grad_flows():
    """Gradients must flow back through ExpertFFN to the input."""
    expert = ExpertFFN(D_MODEL, D_FF)
    x = make_x().requires_grad_(True)
    out = expert(x)
    out.sum().backward()
    assert x.grad is not None, "no gradient at input"
    assert x.grad.abs().sum().item() > 0, "gradient is zero at input"


# ---------------------------------------------------------------------------
# SparseMoELayer
# ---------------------------------------------------------------------------


def test_sparse_moe_output_shape():
    """SparseMoELayer output must have same shape as input."""
    layer = SparseMoELayer(D_MODEL, D_FF, N_EXPERTS, TOP_K)
    x = make_x()
    out, _ = layer(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"


def test_sparse_moe_router_loss_nonneg():
    """SparseMoELayer router_loss must be a non-negative scalar."""
    layer = SparseMoELayer(D_MODEL, D_FF, N_EXPERTS, TOP_K)
    x = make_x()
    _, loss = layer(x)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_sparse_moe_grad_flows_to_input():
    """Gradients must flow through SparseMoELayer back to the input tensor."""
    layer = SparseMoELayer(D_MODEL, D_FF, N_EXPERTS, TOP_K)
    x = make_x().requires_grad_(True)
    out, loss = layer(x)
    (out.sum() + loss).backward()
    assert x.grad is not None, "no gradient at input"
    assert x.grad.abs().sum().item() > 0, "gradient is zero at input"


# ---------------------------------------------------------------------------
# MoEBlock
# ---------------------------------------------------------------------------


def test_moe_block_output_shape():
    """MoEBlock output must have same shape as input."""
    block = MoEBlock(D_MODEL, D_FF, N_EXPERTS, TOP_K, N_HEADS)
    x = make_x()
    out, _ = block(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"


def test_moe_block_finite_output():
    """MoEBlock must produce finite (no NaN / Inf) output."""
    block = MoEBlock(D_MODEL, D_FF, N_EXPERTS, TOP_K, N_HEADS)
    x = make_x()
    out, loss = block(x)
    assert torch.isfinite(out).all(), "output contains NaN or Inf"
    assert torch.isfinite(loss), "router_loss is not finite"


def test_moe_block_grad_flows():
    """Gradients must flow through MoEBlock back to the input."""
    block = MoEBlock(D_MODEL, D_FF, N_EXPERTS, TOP_K, N_HEADS)
    x = make_x().requires_grad_(True)
    out, loss = block(x)
    (out.sum() + loss).backward()
    assert x.grad is not None, "no gradient at input"
    assert x.grad.abs().sum().item() > 0, "gradient is zero at input"


# ---------------------------------------------------------------------------
# Load balancing
# ---------------------------------------------------------------------------


def test_load_balance_no_single_expert_dominates():
    """With random inputs no single expert should receive > 90% of tokens."""
    torch.manual_seed(42)
    layer = SparseMoELayer(D_MODEL, D_FF, N_EXPERTS, TOP_K)
    x = make_x(b=16, t=32)  # larger batch for reliable statistics
    with torch.no_grad():
        _, indices, _ = layer.router(x)  # (16, 32, TOP_K)

    idx_flat = indices.reshape(-1)  # (16 * 32 * TOP_K,)
    idx_flat.numel()
    for e in range(N_EXPERTS):
        frac = (idx_flat == e).float().mean().item()
        assert frac < 0.9, (
            f"Expert {e} received {frac:.2%} of tokens — "
            "load is dangerously imbalanced for a freshly initialised router"
        )


# ---------------------------------------------------------------------------
# top_k = 1
# ---------------------------------------------------------------------------


def test_top_k_1_shapes():
    """top_k=1 (hard routing) must produce correct shapes and finite output."""
    layer = SparseMoELayer(D_MODEL, D_FF, N_EXPERTS, top_k=1)
    x = make_x()
    out, loss = layer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_top_k_1_router_shapes():
    """TopKRouter with top_k=1 returns shapes (..., 1) for weights and indices."""
    router = TopKRouter(D_MODEL, N_EXPERTS, top_k=1)
    x = make_x()
    weights, indices, loss = router(x)
    assert weights.shape == (B, T, 1)
    assert indices.shape == (B, T, 1)
    assert loss.ndim == 0
