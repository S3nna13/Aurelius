"""Tests for Grouped Query Attention (Ainslie et al. 2023).

Import path: aurelius.model.grouped_query_attention
"""

import pytest
import torch

from aurelius.model.grouped_query_attention import (
    GQAConfig,
    GQALayer,
    GroupedQueryAttention,
    MultiHeadAttentionBaseline,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

B = 2
T = 8
D_MODEL = 32
N_Q_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 8
D_FF = 64


@pytest.fixture()
def cfg() -> GQAConfig:
    return GQAConfig(
        d_model=D_MODEL,
        n_q_heads=N_Q_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
    )


@pytest.fixture()
def x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, D_MODEL)


@pytest.fixture()
def gqa(cfg: GQAConfig) -> GroupedQueryAttention:
    torch.manual_seed(42)
    return GroupedQueryAttention(cfg)


@pytest.fixture()
def layer(cfg: GQAConfig) -> GQALayer:
    torch.manual_seed(42)
    return GQALayer(cfg, D_FF)


# ---------------------------------------------------------------------------
# 1. GQAConfig validation
# ---------------------------------------------------------------------------


def test_config_valid():
    """Valid config should not raise."""
    cfg = GQAConfig(d_model=32, n_q_heads=4, n_kv_heads=2, head_dim=8)
    assert cfg.n_q_per_kv == 2


def test_config_invalid_kv_heads():
    """n_q_heads not divisible by n_kv_heads must raise ValueError."""
    with pytest.raises(ValueError, match="divisible"):
        GQAConfig(d_model=32, n_q_heads=4, n_kv_heads=3, head_dim=8)


# ---------------------------------------------------------------------------
# 2. GQA output shape
# ---------------------------------------------------------------------------


def test_gqa_output_shape(gqa: GroupedQueryAttention, x: torch.Tensor):
    out = gqa(x)
    assert out.shape == (B, T, D_MODEL), f"Expected {(B, T, D_MODEL)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 3. GQA output is finite (no NaN / Inf)
# ---------------------------------------------------------------------------


def test_gqa_output_finite(gqa: GroupedQueryAttention, x: torch.Tensor):
    out = gqa(x)
    assert torch.isfinite(out).all(), "GQA output contains NaN or Inf"


# ---------------------------------------------------------------------------
# 4. Gradient flows to input
# ---------------------------------------------------------------------------


def test_gqa_gradient_flows(gqa: GroupedQueryAttention, x: torch.Tensor):
    x_req = x.detach().requires_grad_(True)
    out = gqa(x_req)
    out.sum().backward()
    assert x_req.grad is not None, "No gradient at input"
    assert torch.isfinite(x_req.grad).all(), "Input gradient contains NaN/Inf"


# ---------------------------------------------------------------------------
# 5. GQA with n_kv_heads == n_q_heads behaves like standard MHA (same shapes)
# ---------------------------------------------------------------------------


def test_gqa_equal_heads_same_shape(x: torch.Tensor):
    """When n_kv == n_q the output shape must equal that of standard MHA."""
    cfg_full = GQAConfig(
        d_model=D_MODEL, n_q_heads=N_Q_HEADS, n_kv_heads=N_Q_HEADS, head_dim=HEAD_DIM
    )
    gqa_full = GroupedQueryAttention(cfg_full)
    mha = MultiHeadAttentionBaseline(D_MODEL, N_Q_HEADS, HEAD_DIM)

    out_gqa = gqa_full(x)
    out_mha = mha(x)

    assert out_gqa.shape == out_mha.shape, (
        f"Shape mismatch: GQA {out_gqa.shape} vs MHA {out_mha.shape}"
    )


# ---------------------------------------------------------------------------
# 6. n_kv_heads=1 (MQA extreme) — output shape correct
# ---------------------------------------------------------------------------


def test_gqa_single_kv_head(x: torch.Tensor):
    cfg_mqa = GQAConfig(
        d_model=D_MODEL, n_q_heads=N_Q_HEADS, n_kv_heads=1, head_dim=HEAD_DIM
    )
    model = GroupedQueryAttention(cfg_mqa)
    out = model(x)
    assert out.shape == (B, T, D_MODEL), f"MQA shape wrong: {out.shape}"


# ---------------------------------------------------------------------------
# 7. Causal mask — position t is unaffected by future tokens
# ---------------------------------------------------------------------------


def test_causal_mask_no_future_leakage(gqa: GroupedQueryAttention, x: torch.Tensor):
    """Changing tokens after position t must not alter output at position t."""
    pivot = T // 2  # position we check (inclusive)

    with torch.no_grad():
        out_orig = gqa(x)

        # Corrupt all tokens strictly after `pivot`
        x_corrupt = x.clone()
        x_corrupt[:, pivot + 1:, :] = torch.randn_like(x_corrupt[:, pivot + 1:, :])
        out_corrupt = gqa(x_corrupt)

    # Output up to and including pivot must be identical
    assert torch.allclose(
        out_orig[:, : pivot + 1, :],
        out_corrupt[:, : pivot + 1, :],
        atol=1e-5,
    ), "Causal mask broken: future tokens affected past positions"


# ---------------------------------------------------------------------------
# 8. GQALayer output shape
# ---------------------------------------------------------------------------


def test_layer_output_shape(layer: GQALayer, x: torch.Tensor):
    out = layer(x)
    assert out.shape == (B, T, D_MODEL), f"Layer shape wrong: {out.shape}"


# ---------------------------------------------------------------------------
# 9. GQALayer output finite
# ---------------------------------------------------------------------------


def test_layer_output_finite(layer: GQALayer, x: torch.Tensor):
    out = layer(x)
    assert torch.isfinite(out).all(), "GQALayer output contains NaN or Inf"


# ---------------------------------------------------------------------------
# 10. GQALayer gradient flows
# ---------------------------------------------------------------------------


def test_layer_gradient_flows(layer: GQALayer, x: torch.Tensor):
    x_req = x.detach().requires_grad_(True)
    out = layer(x_req)
    out.sum().backward()
    assert x_req.grad is not None, "No gradient at input for GQALayer"
    assert torch.isfinite(x_req.grad).all(), "GQALayer input gradient has NaN/Inf"


# ---------------------------------------------------------------------------
# 11. KV parameter count < Q parameter count when n_kv_heads < n_q_heads
# ---------------------------------------------------------------------------


def test_kv_param_count_smaller(gqa: GroupedQueryAttention):
    q_params = sum(p.numel() for p in gqa.W_q.parameters())
    k_params = sum(p.numel() for p in gqa.W_k.parameters())
    v_params = sum(p.numel() for p in gqa.W_v.parameters())
    assert k_params < q_params, "W_k should have fewer params than W_q for GQA"
    assert v_params < q_params, "W_v should have fewer params than W_q for GQA"


# ---------------------------------------------------------------------------
# 12. Works with sequence length T=1
# ---------------------------------------------------------------------------


def test_gqa_single_token(gqa: GroupedQueryAttention):
    x1 = torch.randn(B, 1, D_MODEL)
    out = gqa(x1)
    assert out.shape == (B, 1, D_MODEL), f"T=1 shape wrong: {out.shape}"
    assert torch.isfinite(out).all(), "T=1 output not finite"


# ---------------------------------------------------------------------------
# 13. mask parameter is accepted without error
# ---------------------------------------------------------------------------


def test_gqa_accepts_mask(gqa: GroupedQueryAttention, x: torch.Tensor):
    """Passing an explicit additive mask must not raise."""
    # Build a simple all-zeros mask (no-op) of shape (B, T, T)
    mask = torch.zeros(B, T, T)
    out = gqa(x, mask=mask)
    assert out.shape == (B, T, D_MODEL)
    assert torch.isfinite(out).all()
