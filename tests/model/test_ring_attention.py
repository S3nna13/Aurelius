"""
Tests for src/model/ring_attention.py

Tiny config used throughout:
    d_model=64, n_heads=4, head_dim=16, chunk_size=8
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.model.ring_attention import RingAttention

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------
D_MODEL = 64
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 16
CHUNK_SIZE = 8


def make_model(**kwargs):
    defaults = dict(d_model=D_MODEL, n_heads=N_HEADS, chunk_size=CHUNK_SIZE, causal=True)
    defaults.update(kwargs)
    return RingAttention(**defaults)


def reference_causal_attn(x, W_q, W_k, W_v, W_o, n_heads):
    """Standard causal multi-head attention using the same weight matrices."""
    B, T, d_model = x.shape
    d_k = d_model // n_heads
    scale = math.sqrt(d_k)

    Q = W_q(x).view(B, T, n_heads, d_k).permute(0, 2, 1, 3)
    K = W_k(x).view(B, T, n_heads, d_k).permute(0, 2, 1, 3)
    V = W_v(x).view(B, T, n_heads, d_k).permute(0, 2, 1, 3)

    S = torch.matmul(Q, K.transpose(-2, -1)) / scale
    mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
    S = S.masked_fill(mask, float("-inf"))
    A = F.softmax(S, dim=-1)
    A = torch.nan_to_num(A, nan=0.0)
    out = torch.matmul(A, V).permute(0, 2, 1, 3).contiguous().view(B, T, d_model)
    return W_o(out)


# ---------------------------------------------------------------------------
# 1. Output shape matches input (B, T, d_model)
# ---------------------------------------------------------------------------
def test_output_shape():
    model = make_model()
    x = torch.randn(2, 32, D_MODEL)
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 2. Gradient flow: finite grads on all params
# ---------------------------------------------------------------------------
def test_gradient_flow():
    model = make_model()
    x = torch.randn(2, 32, D_MODEL, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


# ---------------------------------------------------------------------------
# 3. Determinism under torch.manual_seed
# ---------------------------------------------------------------------------
def test_determinism():
    torch.manual_seed(42)
    model = make_model()
    x = torch.randn(1, 16, D_MODEL)
    torch.manual_seed(99)
    out1 = model(x)
    torch.manual_seed(99)
    out2 = model(x)
    assert torch.allclose(out1, out2), "Output is not deterministic"


# ---------------------------------------------------------------------------
# 4. batch=1, seq_len=chunk_size (single chunk round-trip)
# ---------------------------------------------------------------------------
def test_single_chunk():
    model = make_model()
    x = torch.randn(1, CHUNK_SIZE, D_MODEL)
    out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 5. seq_len = 4 * chunk_size (multiple chunks)
# ---------------------------------------------------------------------------
def test_multiple_chunks():
    model = make_model()
    T = 4 * CHUNK_SIZE
    x = torch.randn(2, T, D_MODEL)
    out = model(x)
    assert out.shape == (2, T, D_MODEL)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 6. seq_len not divisible by chunk_size -> raises clear ValueError
# ---------------------------------------------------------------------------
def test_non_divisible_seqlen_raises():
    model = make_model()
    x = torch.randn(1, CHUNK_SIZE + 3, D_MODEL)
    with pytest.raises(ValueError, match="divisible"):
        model(x)


# ---------------------------------------------------------------------------
# 7. Numerical stability: no NaN/Inf on zeros input
# ---------------------------------------------------------------------------
def test_numerical_stability_zeros():
    model = make_model()
    x = torch.zeros(1, 32, D_MODEL)
    out = model(x)
    assert torch.isfinite(out).all(), "Got NaN/Inf on zero input"


# ---------------------------------------------------------------------------
# 8. Numerical stability: no NaN/Inf on large inputs
# ---------------------------------------------------------------------------
def test_numerical_stability_large():
    model = make_model()
    x = torch.randn(1, 32, D_MODEL) * 1e3
    out = model(x)
    assert torch.isfinite(out).all(), "Got NaN/Inf on large input"


# ---------------------------------------------------------------------------
# 9. Causal correctness: equivalence to standard causal attention (atol=1e-4)
# ---------------------------------------------------------------------------
def test_causal_correctness():
    torch.manual_seed(0)
    model = make_model()
    model.train(False)  # inference mode, no dropout etc.

    T = 2 * CHUNK_SIZE  # 16 tokens; two chunks
    x = torch.randn(1, T, D_MODEL)

    with torch.no_grad():
        ring_out = model(x)
        ref_out = reference_causal_attn(x, model.W_q, model.W_k, model.W_v, model.W_o, N_HEADS)

    assert torch.allclose(ring_out, ref_out, atol=1e-4), (
        f"Max diff: {(ring_out - ref_out).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 10. Online lse accumulator correctness: equivalent to batch softmax on full seq
#     (non-causal variant so we can compare to standard full-attention softmax)
# ---------------------------------------------------------------------------
def test_online_lse_equivalence():
    """For non-causal ring attention, output should equal standard full attention."""
    torch.manual_seed(7)
    model = make_model(causal=False)
    model.train(False)

    T = 2 * CHUNK_SIZE
    x = torch.randn(1, T, D_MODEL)

    with torch.no_grad():
        ring_out = model(x)

        B, T_, d_model = x.shape
        d_k = d_model // N_HEADS
        scale = math.sqrt(d_k)

        Q = model.W_q(x).view(B, T_, N_HEADS, d_k).permute(0, 2, 1, 3)
        K = model.W_k(x).view(B, T_, N_HEADS, d_k).permute(0, 2, 1, 3)
        V = model.W_v(x).view(B, T_, N_HEADS, d_k).permute(0, 2, 1, 3)
        S = torch.matmul(Q, K.transpose(-2, -1)) / scale
        A = F.softmax(S, dim=-1)
        ref = torch.matmul(A, V).permute(0, 2, 1, 3).contiguous().view(B, T_, d_model)
        ref_out = model.W_o(ref)

    assert torch.allclose(ring_out, ref_out, atol=1e-4), (
        f"Max diff: {(ring_out - ref_out).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# 11. Attention mask (padding) does not crash
# ---------------------------------------------------------------------------
def test_attention_mask_no_crash():
    model = make_model()
    B, T = 2, 32
    x = torch.randn(B, T, D_MODEL)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 24:] = False
    out = model(x, attention_mask=mask)
    assert out.shape == (B, T, D_MODEL)
    assert torch.isfinite(out[:, :24, :]).all()


# ---------------------------------------------------------------------------
# 12. n_heads=1 (all dim in one head) still works
# ---------------------------------------------------------------------------
def test_single_head():
    model = RingAttention(d_model=D_MODEL, n_heads=1, chunk_size=CHUNK_SIZE, causal=True)
    x = torch.randn(1, 16, D_MODEL)
    out = model(x)
    assert out.shape == (1, 16, D_MODEL)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 13. n_heads = d_model (head_dim=1) still works
# ---------------------------------------------------------------------------
def test_max_heads():
    d = 16
    model = RingAttention(d_model=d, n_heads=d, chunk_size=CHUNK_SIZE, causal=True)
    x = torch.randn(1, 16, d)
    out = model(x)
    assert out.shape == (1, 16, d)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 14. Invalid d_model / n_heads raises ValueError at construction time
# ---------------------------------------------------------------------------
def test_invalid_heads_raises():
    with pytest.raises(ValueError, match="divisible"):
        RingAttention(d_model=65, n_heads=4, chunk_size=CHUNK_SIZE)


# ---------------------------------------------------------------------------
# 15. Larger batch size works correctly
# ---------------------------------------------------------------------------
def test_large_batch():
    model = make_model()
    x = torch.randn(8, 32, D_MODEL)
    out = model(x)
    assert out.shape == (8, 32, D_MODEL)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 16. Causal masking independence: output at pos t must NOT depend on t+1 tokens
# ---------------------------------------------------------------------------
def test_causal_masking_independence():
    torch.manual_seed(123)
    model = make_model(causal=True)
    model.train(False)

    T = 2 * CHUNK_SIZE
    x = torch.randn(1, T, D_MODEL)
    x2 = x.clone()
    x2[:, T // 2 :, :] = torch.randn(1, T // 2, D_MODEL)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x2)

    assert torch.allclose(out1[:, : T // 2, :], out2[:, : T // 2, :], atol=1e-5), (
        "Causal violation: future token perturbation affected past outputs"
    )
