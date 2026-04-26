"""Tests for flash_attention_sim.py — tiled Flash Attention simulation.

Covers: config defaults, output shapes, numerical equivalence (causal and
non-causal), gradient flow, single-token edge case, causal masking
correctness, multi-head module shapes, and the compare_attention_outputs
helper.

Tiny dimensions are used throughout so tests run fast on CPU.
"""

from __future__ import annotations

import torch

from src.model.flash_attention_sim import (
    FlashAttentionSim,
    FlashAttnConfig,
    MultiHeadFlashAttn,
    compare_attention_outputs,
    standard_attention,
    tiled_attention,
)

# ---------------------------------------------------------------------------
# Shared tiny dimensions
# ---------------------------------------------------------------------------
D_MODEL = 16
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS  # 8
BLOCK = 4
BATCH = 2
SEQ = 8


def make_qkv(
    batch: int = BATCH,
    heads: int = N_HEADS,
    seq: int = SEQ,
    d_head: int = D_HEAD,
    seed: int = 42,
):
    """Return random (Q, K, V) tensors of shape (batch, heads, seq, d_head)."""
    torch.manual_seed(seed)
    Q = torch.randn(batch, heads, seq, d_head)
    K = torch.randn(batch, heads, seq, d_head)
    V = torch.randn(batch, heads, seq, d_head)
    return Q, K, V


# ===========================================================================
# 1. FlashAttnConfig defaults
# ===========================================================================


def test_config_defaults():
    """FlashAttnConfig must have the specified default values."""
    cfg = FlashAttnConfig()
    assert cfg.block_size == 64
    assert cfg.causal is True
    assert cfg.dropout_p == 0.0
    assert cfg.scale is None


# ===========================================================================
# 2. standard_attention output shape
# ===========================================================================


def test_standard_attention_output_shape():
    Q, K, V = make_qkv()
    out = standard_attention(Q, K, V)
    assert out.shape == (BATCH, N_HEADS, SEQ, D_HEAD), (
        f"Expected {(BATCH, N_HEADS, SEQ, D_HEAD)}, got {out.shape}"
    )


# ===========================================================================
# 3. tiled_attention output shape matches standard
# ===========================================================================


def test_tiled_attention_output_shape():
    Q, K, V = make_qkv()
    out = tiled_attention(Q, K, V, causal=True, block_size=BLOCK)
    assert out.shape == (BATCH, N_HEADS, SEQ, D_HEAD), (
        f"Expected {(BATCH, N_HEADS, SEQ, D_HEAD)}, got {out.shape}"
    )


# ===========================================================================
# 4. Numerical equivalence — causal=True
# ===========================================================================


def test_numerical_equivalence_causal():
    """tiled_attention must match standard_attention with causal=True."""
    Q, K, V = make_qkv()

    # Build causal mask for standard_attention (True = masked out)
    idx = torch.arange(SEQ)
    mask = idx.unsqueeze(0) > idx.unsqueeze(1)  # (SEQ, SEQ)

    std = standard_attention(Q, K, V, mask=mask)
    tiled = tiled_attention(Q, K, V, causal=True, block_size=BLOCK)

    max_diff = (std.float() - tiled.float()).abs().max().item()
    assert max_diff < 1e-4, f"Causal: max absolute difference {max_diff:.2e} exceeds 1e-4"


# ===========================================================================
# 5. Numerical equivalence — causal=False
# ===========================================================================


def test_numerical_equivalence_non_causal():
    """tiled_attention must match standard_attention with causal=False."""
    Q, K, V = make_qkv(seed=7)

    std = standard_attention(Q, K, V)
    tiled = tiled_attention(Q, K, V, causal=False, block_size=BLOCK)

    max_diff = (std.float() - tiled.float()).abs().max().item()
    assert max_diff < 1e-4, f"Non-causal: max absolute difference {max_diff:.2e} exceeds 1e-4"


# ===========================================================================
# 6. FlashAttentionSim output shape
# ===========================================================================


def test_flash_attention_sim_output_shape():
    cfg = FlashAttnConfig(block_size=BLOCK, causal=True)
    model = FlashAttentionSim(D_MODEL, N_HEADS, cfg)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = model(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), f"Expected {(BATCH, SEQ, D_MODEL)}, got {out.shape}"


# ===========================================================================
# 7. Gradient flows through tiled_attention
# ===========================================================================


def test_gradient_flow_tiled_attention():
    """Gradients must reach Q, K, V inputs through tiled_attention."""
    Q, K, V = make_qkv()
    Q = Q.detach().requires_grad_(True)
    K = K.detach().requires_grad_(True)
    V = V.detach().requires_grad_(True)

    out = tiled_attention(Q, K, V, causal=True, block_size=BLOCK)
    loss = out.sum()
    loss.backward()

    assert Q.grad is not None, "Q.grad is None — gradient did not flow"
    assert K.grad is not None, "K.grad is None — gradient did not flow"
    assert V.grad is not None, "V.grad is None — gradient did not flow"

    assert Q.grad.abs().max() > 0, "Q.grad is all-zero"
    assert V.grad.abs().max() > 0, "V.grad is all-zero"


# ===========================================================================
# 8. MultiHeadFlashAttn output shape
# ===========================================================================


def test_multi_head_flash_attn_output_shape():
    cfg = FlashAttnConfig(block_size=BLOCK, causal=True)
    model = MultiHeadFlashAttn(D_MODEL, N_HEADS, cfg)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = model(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), f"Expected {(BATCH, SEQ, D_MODEL)}, got {out.shape}"


# ===========================================================================
# 9. Single-token sequence (T=1)
# ===========================================================================


def test_single_token_sequence():
    """tiled_attention must handle seq_len=1 without error."""
    Q, K, V = make_qkv(seq=1, seed=99)
    out = tiled_attention(Q, K, V, causal=True, block_size=BLOCK)
    assert out.shape == (BATCH, N_HEADS, 1, D_HEAD)

    std = standard_attention(Q, K, V)
    max_diff = (std.float() - out.float()).abs().max().item()
    assert max_diff < 1e-4, f"Single-token diff {max_diff:.2e} too large"


# ===========================================================================
# 10. Causal masking — future tokens have zero weight
# ===========================================================================


def test_causal_masking_future_tokens_zero_weight():
    """With causal masking, zeroing future K/V should not affect past outputs."""
    torch.manual_seed(0)
    SEQ_LEN = 8
    PIVOT = 4

    Q = torch.randn(1, 1, SEQ_LEN, D_HEAD)
    K = torch.randn(1, 1, SEQ_LEN, D_HEAD)
    V = torch.randn(1, 1, SEQ_LEN, D_HEAD)

    K_mod = K.clone()
    V_mod = V.clone()
    K_mod[:, :, PIVOT:, :] = 0.0
    V_mod[:, :, PIVOT:, :] = 0.0

    out_orig = tiled_attention(Q, K, V, causal=True, block_size=BLOCK)
    out_mod = tiled_attention(Q, K_mod, V_mod, causal=True, block_size=BLOCK)

    max_diff = (
        (out_orig[:, :, :PIVOT, :].float() - out_mod[:, :, :PIVOT, :].float()).abs().max().item()
    )
    assert max_diff < 1e-5, (
        f"Causal masking failed: positions < {PIVOT} differ by {max_diff:.2e} "
        "when future K/V are zeroed"
    )


# ===========================================================================
# 11. compare_attention_outputs — returns small number
# ===========================================================================


def test_compare_attention_outputs_small():
    """compare_attention_outputs must return a value < 1e-4."""
    Q, K, V = make_qkv()
    diff = compare_attention_outputs(Q, K, V, block_size=BLOCK, causal=True)
    assert isinstance(diff, float), "compare_attention_outputs should return float"
    assert diff < 1e-4, f"Max diff {diff:.2e} exceeds 1e-4"


# ===========================================================================
# 12. FlashAttentionSim has no bias in projections
# ===========================================================================


def test_flash_attention_sim_no_bias():
    cfg = FlashAttnConfig(block_size=BLOCK)
    model = FlashAttentionSim(D_MODEL, N_HEADS, cfg)
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        layer = getattr(model, name)
        assert layer.bias is None, f"{name} should have no bias"


# ===========================================================================
# 13. Gradient flows through FlashAttentionSim parameters
# ===========================================================================


def test_gradient_flow_flash_attention_sim():
    """All parameters of FlashAttentionSim must receive non-zero gradients."""
    cfg = FlashAttnConfig(block_size=BLOCK, causal=True)
    model = FlashAttentionSim(D_MODEL, N_HEADS, cfg)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    loss = model(x).sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' has no gradient"
        assert param.grad.abs().max() > 0, f"Parameter '{name}' gradient is all-zero"


# ===========================================================================
# 14. MultiHeadFlashAttn residual connection is present
# ===========================================================================


def test_multi_head_residual_present():
    """MultiHeadFlashAttn output must include the residual skip connection."""
    cfg = FlashAttnConfig(block_size=BLOCK, causal=True)
    model = MultiHeadFlashAttn(D_MODEL, N_HEADS, cfg)
    model.eval()
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, D_MODEL)

    with torch.no_grad():
        out_mh = model(x)
        out_attn_only = model.attn(model.norm(x))

    assert not torch.allclose(out_mh, out_attn_only), (
        "MultiHeadFlashAttn output identical to attn-only — residual may be missing"
    )


# ===========================================================================
# 15. compare_attention_outputs non-causal also stays small
# ===========================================================================


def test_compare_attention_outputs_non_causal():
    Q, K, V = make_qkv(seed=13)
    diff = compare_attention_outputs(Q, K, V, block_size=BLOCK, causal=False)
    assert diff < 1e-4, f"Non-causal max diff {diff:.2e} exceeds 1e-4"
