"""Tests for MoE with top-p routing (moe_topk.py)."""

from __future__ import annotations

import torch

from src.model.moe_topk import (
    ExpertFFN,
    MoETopPConfig,
    MoETopPLayer,
    MoETopPTransformer,
    top_p_aux_loss,
    top_p_routing,
)

# ---------------------------------------------------------------------------
# 1. MoETopPConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = MoETopPConfig()
    assert cfg.n_experts == 8
    assert cfg.d_model == 256
    assert cfg.d_ff == 512
    assert cfg.top_p == 0.9
    assert cfg.min_experts == 1
    assert cfg.max_experts == 4
    assert cfg.aux_loss_coeff == 0.01


# ---------------------------------------------------------------------------
# 2. ExpertFFN output shape matches input
# ---------------------------------------------------------------------------


def test_expert_ffn_output_shape():
    d_model, d_ff = 64, 128
    expert = ExpertFFN(d_model, d_ff)
    x = torch.randn(4, 10, d_model)
    out = expert(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 3. ExpertFFN is differentiable
# ---------------------------------------------------------------------------


def test_expert_ffn_differentiable():
    expert = ExpertFFN(32, 64)
    x = torch.randn(2, 5, 32, requires_grad=True)
    out = expert(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# 4. top_p_routing output shapes
# ---------------------------------------------------------------------------


def test_top_p_routing_output_shapes():
    B, T, n_experts, max_experts = 3, 7, 8, 4
    N = B * T
    logits = torch.randn(N, n_experts)
    weights, indices = top_p_routing(logits, top_p=0.9, min_experts=1, max_experts=max_experts)
    assert weights.shape == (N, max_experts), f"weights shape {weights.shape}"
    assert indices.shape == (N, max_experts), f"indices shape {indices.shape}"


# ---------------------------------------------------------------------------
# 5. top_p_routing selected_weights sum <= 1.0 per token
# ---------------------------------------------------------------------------


def test_top_p_routing_weights_sum():
    N, n_experts = 20, 8
    logits = torch.randn(N, n_experts)
    weights, _ = top_p_routing(logits, top_p=0.9, min_experts=1, max_experts=4)
    sums = weights.sum(dim=-1)
    # Each token's weights should sum to ~1.0 (or 0 if no experts, but min_experts=1 prevents that)
    assert (sums <= 1.0 + 1e-5).all(), f"Some token weights exceed 1.0: {sums.max()}"
    assert (sums >= 0.0).all(), "Negative weight sums"


# ---------------------------------------------------------------------------
# 6. top_p_routing selected_indices in [-1, n_experts)
# ---------------------------------------------------------------------------


def test_top_p_routing_indices_valid():
    N, n_experts = 15, 8
    logits = torch.randn(N, n_experts)
    _, indices = top_p_routing(logits, top_p=0.9, min_experts=1, max_experts=4)
    # All indices must be either -1 (unused) or in [0, n_experts)
    assert (indices >= -1).all(), "Index below -1 found"
    assert (indices < n_experts).all(), f"Index >= n_experts found: {indices.max()}"


# ---------------------------------------------------------------------------
# 7. top_p_routing with top_p=1.0 uses up to max_experts
# ---------------------------------------------------------------------------


def test_top_p_routing_full_coverage():
    """With top_p=1.0, every token should use max_experts experts."""
    N, n_experts, max_experts = 10, 8, 4
    logits = torch.randn(N, n_experts)
    weights, indices = top_p_routing(logits, top_p=1.0, min_experts=1, max_experts=max_experts)
    active_counts = (indices >= 0).sum(dim=-1)
    assert (active_counts == max_experts).all(), (
        f"With top_p=1.0, all tokens should use max_experts={max_experts}, got: {active_counts}"
    )


# ---------------------------------------------------------------------------
# 8. top_p_routing with top_p=0.0 uses min_experts
# ---------------------------------------------------------------------------


def test_top_p_routing_min_experts():
    """With top_p=0.0, every token should use exactly min_experts experts."""
    N, n_experts, min_experts = 10, 8, 1
    logits = torch.randn(N, n_experts)
    weights, indices = top_p_routing(logits, top_p=0.0, min_experts=min_experts, max_experts=4)
    active_counts = (indices >= 0).sum(dim=-1)
    assert (active_counts == min_experts).all(), (
        f"With top_p=0.0, all tokens should use min_experts={min_experts}, got: {active_counts}"
    )


# ---------------------------------------------------------------------------
# 9. top_p_aux_loss returns scalar
# ---------------------------------------------------------------------------


def test_top_p_aux_loss_scalar():
    N, n_experts = 12, 8
    logits = torch.randn(N, n_experts)
    _, indices = top_p_routing(logits, top_p=0.9, min_experts=1, max_experts=4)
    loss = top_p_aux_loss(logits, indices, n_experts)
    assert loss.ndim == 0, f"Expected scalar (0-d), got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 10. top_p_aux_loss >= 0
# ---------------------------------------------------------------------------


def test_top_p_aux_loss_nonneg():
    N, n_experts = 12, 8
    logits = torch.randn(N, n_experts)
    _, indices = top_p_routing(logits, top_p=0.9, min_experts=1, max_experts=4)
    loss = top_p_aux_loss(logits, indices, n_experts)
    assert loss.item() >= 0.0, f"aux_loss should be >= 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 11. MoETopPLayer output shape is (B, T, d_model)
# ---------------------------------------------------------------------------


def test_moe_layer_output_shape():
    cfg = MoETopPConfig(n_experts=4, d_model=32, d_ff=64, max_experts=2)
    layer = MoETopPLayer(cfg)
    B, T = 2, 6
    x = torch.randn(B, T, cfg.d_model)
    out, _ = layer(x)
    assert out.shape == (B, T, cfg.d_model), f"Expected {(B, T, cfg.d_model)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 12. MoETopPLayer aux_loss is scalar >= 0
# ---------------------------------------------------------------------------


def test_moe_layer_aux_loss():
    cfg = MoETopPConfig(n_experts=4, d_model=32, d_ff=64, max_experts=2)
    layer = MoETopPLayer(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    _, aux_loss = layer(x)
    assert aux_loss.ndim == 0, f"aux_loss should be scalar, got shape {aux_loss.shape}"
    assert aux_loss.item() >= 0.0, f"aux_loss should be >= 0, got {aux_loss.item()}"


# ---------------------------------------------------------------------------
# 13. MoETopPLayer.routing_stats returns dict with expected keys
# ---------------------------------------------------------------------------


def test_moe_layer_routing_stats():
    cfg = MoETopPConfig(n_experts=4, d_model=32, d_ff=64, max_experts=2)
    layer = MoETopPLayer(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    layer(x)  # populate cache
    stats = layer.routing_stats()
    assert isinstance(stats, dict), "routing_stats() should return a dict"
    assert "mean_experts_used" in stats, "Missing key: mean_experts_used"
    assert "expert_utilization" in stats, "Missing key: expert_utilization"
    assert isinstance(stats["expert_utilization"], list), "expert_utilization should be a list"
    assert len(stats["expert_utilization"]) == cfg.n_experts


# ---------------------------------------------------------------------------
# 14. MoETopPTransformer output logits shape is (B, T, vocab_size)
# ---------------------------------------------------------------------------


def test_transformer_logits_shape():
    cfg = MoETopPConfig(n_experts=4, d_model=32, d_ff=64, max_experts=2)
    vocab_size = 128
    model = MoETopPTransformer(cfg, n_layers=2, vocab_size=vocab_size)
    B, T = 2, 8
    input_ids = torch.randint(0, vocab_size, (B, T))
    logits, _ = model(input_ids)
    assert logits.shape == (B, T, vocab_size), f"Expected {(B, T, vocab_size)}, got {logits.shape}"


# ---------------------------------------------------------------------------
# 15. MoETopPTransformer aux_loss is scalar
# ---------------------------------------------------------------------------


def test_transformer_aux_loss_scalar():
    cfg = MoETopPConfig(n_experts=4, d_model=32, d_ff=64, max_experts=2)
    model = MoETopPTransformer(cfg, n_layers=2, vocab_size=128)
    input_ids = torch.randint(0, 128, (2, 6))
    _, aux_loss = model(input_ids)
    assert aux_loss.ndim == 0, f"aux_loss should be scalar, got shape {aux_loss.shape}"


# ---------------------------------------------------------------------------
# 16. MoETopPTransformer is differentiable
# ---------------------------------------------------------------------------


def test_transformer_differentiable():
    cfg = MoETopPConfig(n_experts=4, d_model=32, d_ff=64, max_experts=2)
    model = MoETopPTransformer(cfg, n_layers=2, vocab_size=128)
    input_ids = torch.randint(0, 128, (2, 6))
    logits, aux_loss = model(input_ids)
    loss = logits.sum() + aux_loss
    loss.backward()
    # Check at least one parameter has gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No parameter gradients after backward()"
