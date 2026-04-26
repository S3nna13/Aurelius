"""Tests for src/model/sliding_window_attn.py.

Tiny configs are used throughout to keep tests fast.
"""

import torch

from src.model.sliding_window_attn import (
    SlidingWindowAttention,
    SWABlock,
    SWAConfig,
    build_sliding_window_mask,
    compare_swa_vs_full_attention,
    sliding_window_attention,
)

# ---------------------------------------------------------------------------
# Shared tiny constants
# ---------------------------------------------------------------------------
D_MODEL = 16
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS  # 8
WINDOW = 4
BATCH = 2
SEQ = 8

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_config(**kwargs) -> SWAConfig:
    defaults = dict(
        d_model=D_MODEL, n_heads=N_HEADS, window_size=WINDOW, causal=True, dropout_p=0.0
    )
    defaults.update(kwargs)
    return SWAConfig(**defaults)


# ===========================================================================
# 1. SWAConfig defaults
# ===========================================================================


def test_swa_config_defaults():
    cfg = SWAConfig()
    assert cfg.d_model == 64
    assert cfg.n_heads == 4
    assert cfg.window_size == 128
    assert cfg.causal is True
    assert cfg.dropout_p == 0.0


# ===========================================================================
# 2. build_sliding_window_mask — shape
# ===========================================================================


def test_mask_shape():
    mask = build_sliding_window_mask(SEQ, WINDOW)
    assert mask.shape == (SEQ, SEQ), f"Expected ({SEQ},{SEQ}), got {mask.shape}"


# ===========================================================================
# 3. Causal mask: positions strictly above diagonal beyond window are -inf
# ===========================================================================


def test_causal_mask_future_tokens_blocked():
    mask = build_sliding_window_mask(SEQ, WINDOW, causal=True)
    for i in range(SEQ):
        for j in range(i + 1, SEQ):
            assert mask[i, j] < -1e8, (
                f"Future token (i={i}, j={j}) should be blocked, got {mask[i, j]}"
            )


# ===========================================================================
# 4. Within-window positions are 0.0
# ===========================================================================


def test_causal_mask_within_window_is_zero():
    mask = build_sliding_window_mask(SEQ, WINDOW, causal=True)
    for i in range(SEQ):
        for j in range(max(0, i - WINDOW), i + 1):
            assert mask[i, j] == 0.0, (
                f"Within-window position (i={i}, j={j}) should be 0.0, got {mask[i, j]}"
            )


# ===========================================================================
# 5. Beyond-window past tokens are also blocked
# ===========================================================================


def test_causal_mask_beyond_window_past_blocked():
    mask = build_sliding_window_mask(SEQ, WINDOW, causal=True)
    for i in range(SEQ):
        for j in range(0, max(0, i - WINDOW)):
            assert mask[i, j] < -1e8, (
                f"Beyond-window past (i={i}, j={j}) should be blocked, got {mask[i, j]}"
            )


# ===========================================================================
# 6. sliding_window_attention — output shape
# ===========================================================================


def test_sliding_window_attention_output_shape():
    Q = torch.randn(BATCH, N_HEADS, SEQ, D_HEAD)
    K = torch.randn(BATCH, N_HEADS, SEQ, D_HEAD)
    V = torch.randn(BATCH, N_HEADS, SEQ, D_HEAD)
    out = sliding_window_attention(Q, K, V, window_size=WINDOW, causal=True)
    assert out.shape == (BATCH, N_HEADS, SEQ, D_HEAD)


# ===========================================================================
# 7. SWA blocks future tokens — uniform K/V, positions beyond window ≈ 0 weight
#    Verified by checking that blocked positions contribute nothing to output.
#    Strategy: set V = eye-like so output of position i encodes which j attended.
# ===========================================================================


def test_swa_blocks_future_tokens():
    """Positions beyond the window should receive ~zero weight in softmax."""
    T = SEQ
    # Build Q as identity (token i focuses on its own query)
    Q = torch.zeros(1, 1, T, D_HEAD)
    K = torch.zeros(1, 1, T, D_HEAD)
    # Each K[j] points along dimension j (up to D_HEAD)
    for j in range(min(T, D_HEAD)):
        K[0, 0, j, j] = 1.0
    # V[j] = e_j (one-hot)
    V = torch.zeros(1, 1, T, D_HEAD)
    for j in range(min(T, D_HEAD)):
        V[0, 0, j, j] = 1.0

    out = sliding_window_attention(Q, K, V, window_size=WINDOW, causal=True)
    # For token i=0 with WINDOW=4, only j=0 is visible.
    # Future tokens (j>i) must have zero contribution.
    # Check that token 0 has no signal from tokens > 0 in V dimensions > 0.
    # Since Q[0] = 0, attention is uniform over visible tokens.
    # Key test: output at position i=1 should NOT contain info from j=7
    # We simply verify no output position has value > 0.5 from a blocked future dim.
    for i in range(T):
        for j in range(i + 1, T):
            # If j is future/out-of-window, its V contribution (dim j if j < D_HEAD) should be ~0
            if j < D_HEAD:
                assert out[0, 0, i, j].abs().item() < 1e-6, (
                    f"Token i={i} should not attend to future j={j}, "
                    f"but got {out[0, 0, i, j].item():.6f}"
                )


# ===========================================================================
# 8. SlidingWindowAttention — output shape
# ===========================================================================


def test_sliding_window_attention_module_output_shape():
    cfg = make_config()
    swa = SlidingWindowAttention(cfg)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = swa(x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


# ===========================================================================
# 9. Gradient flows through SlidingWindowAttention
# ===========================================================================


def test_gradient_flows():
    cfg = make_config()
    swa = SlidingWindowAttention(cfg)
    x = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=True)
    out = swa(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient did not flow back to input"
    assert x.grad.shape == x.shape
    # At least one parameter should have a non-zero gradient
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in swa.parameters())
    assert has_grad, "No parameter received a non-zero gradient"


# ===========================================================================
# 10. SWABlock — output shape
# ===========================================================================


def test_swa_block_output_shape():
    cfg = make_config()
    block = SWABlock(cfg)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = block(x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


# ===========================================================================
# 11. SWABlock — residual means output != input
# ===========================================================================


def test_swa_block_residual():
    cfg = make_config()
    block = SWABlock(cfg)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = block(x)
    # The output should differ from the input (attention adds something)
    assert not torch.allclose(out, x), (
        "SWABlock output should differ from input due to residual + attn"
    )


# ===========================================================================
# 12. window_size >= seq_len → same as full causal attention
# ===========================================================================


def test_compare_swa_vs_full_attention_large_window():
    """When window covers the entire sequence, SWA == full causal attention."""
    cfg = make_config(window_size=SEQ * 2)  # window larger than seq
    swa = SlidingWindowAttention(cfg)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    diff = compare_swa_vs_full_attention(x, swa, window_size=SEQ * 2)
    assert diff < 1e-4, f"Expected SWA ≈ full causal attn, but max diff = {diff:.2e}"


# ===========================================================================
# 13. Non-causal mask symmetry
# ===========================================================================


def test_non_causal_mask_symmetry():
    mask = build_sliding_window_mask(SEQ, WINDOW, causal=False)
    # Should be symmetric: mask[i,j] == mask[j,i]
    assert torch.allclose(mask, mask.T), "Non-causal mask should be symmetric"


# ===========================================================================
# 14. Single-token sequence
# ===========================================================================


def test_single_token_sequence():
    """A single-token sequence should work without errors."""
    cfg = make_config()
    swa = SlidingWindowAttention(cfg)
    x = torch.randn(1, 1, D_MODEL)
    out = swa(x)
    assert out.shape == (1, 1, D_MODEL)

    block = SWABlock(cfg)
    out_block = block(x)
    assert out_block.shape == (1, 1, D_MODEL)
