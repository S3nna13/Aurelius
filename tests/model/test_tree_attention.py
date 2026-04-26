"""
Tests for Tree Attention (src/model/tree_attention.py).

Paper: Shyam et al., 2024. arXiv:2408.04093
Config: d_model=64, n_heads=4, head_dim=16, chunk_size=8
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.tree_attention import TreeAttention, merge_two_chunks

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

D_MODEL = 64
N_HEADS = 4
HEAD_DIM = 16  # D_MODEL // N_HEADS
CHUNK = 8


def make_model(causal: bool = False, chunk_size: int = CHUNK) -> TreeAttention:
    return TreeAttention(d_model=D_MODEL, n_heads=N_HEADS, chunk_size=chunk_size, causal=causal)


def ref_attention(x: torch.Tensor, model: TreeAttention) -> torch.Tensor:
    """Reference output using F.scaled_dot_product_attention (non-causal)."""
    B, T, D = x.shape
    Q = model.W_Q(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)  # (B,H,T,h)
    K = model.W_K(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
    V = model.W_V(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
    attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
    return model.W_O(attn_out)


# ---------------------------------------------------------------------------
# Test 1: Output shape
# ---------------------------------------------------------------------------


def test_output_shape():
    B, T = 2, 4 * CHUNK
    x = torch.randn(B, T, D_MODEL)
    model = make_model()
    out = model(x)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B},{T},{D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: Equivalence to standard attention (non-causal)
# ---------------------------------------------------------------------------


def test_equivalence_to_standard_attention():
    torch.manual_seed(42)
    B, T = 1, 4 * CHUNK
    x = torch.randn(B, T, D_MODEL)
    model = make_model(causal=False)
    model.eval()

    with torch.no_grad():
        tree_out = model(x)
        std_out = ref_attention(x, model)

    assert torch.allclose(tree_out, std_out, atol=1e-5), (
        f"Max diff: {(tree_out - std_out).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 3: Gradient flow on all parameters
# ---------------------------------------------------------------------------


def test_gradient_flow():
    B, T = 2, 2 * CHUNK
    x = torch.randn(B, T, D_MODEL)
    model = make_model()
    out = model(x)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


# ---------------------------------------------------------------------------
# Test 4: Determinism under torch.manual_seed
# ---------------------------------------------------------------------------


def test_determinism():
    B, T = 1, 4 * CHUNK

    torch.manual_seed(7)
    x1 = torch.randn(B, T, D_MODEL)
    model1 = make_model()
    out1 = model1(x1)

    torch.manual_seed(7)
    x2 = torch.randn(B, T, D_MODEL)
    model2 = make_model()
    model2.load_state_dict(model1.state_dict())
    out2 = model2(x2)

    assert torch.equal(out1, out2), "Outputs differ under same seed"


# ---------------------------------------------------------------------------
# Test 5: batch=1, T = chunk_size (single leaf)
# ---------------------------------------------------------------------------


def test_single_leaf():
    B, T = 1, CHUNK
    x = torch.randn(B, T, D_MODEL)
    model = make_model(causal=False)
    model.eval()

    with torch.no_grad():
        tree_out = model(x)
        std_out = ref_attention(x, model)

    assert tree_out.shape == (B, T, D_MODEL)
    assert torch.allclose(tree_out, std_out, atol=1e-5), (
        f"Max diff: {(tree_out - std_out).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 6: T = 4 * chunk_size (4 leaves, binary tree)
# ---------------------------------------------------------------------------


def test_four_leaves():
    torch.manual_seed(11)
    B, T = 2, 4 * CHUNK
    x = torch.randn(B, T, D_MODEL)
    model = make_model(causal=False)
    model.eval()

    with torch.no_grad():
        tree_out = model(x)
        std_out = ref_attention(x, model)

    assert torch.allclose(tree_out, std_out, atol=1e-5), (
        f"Max diff: {(tree_out - std_out).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 7: T = 8 * chunk_size (8 leaves, 3 levels)
# ---------------------------------------------------------------------------


def test_eight_leaves():
    torch.manual_seed(99)
    B, T = 1, 8 * CHUNK
    x = torch.randn(B, T, D_MODEL)
    model = make_model(causal=False)
    model.eval()

    with torch.no_grad():
        tree_out = model(x)
        std_out = ref_attention(x, model)

    assert torch.allclose(tree_out, std_out, atol=1e-5), (
        f"Max diff: {(tree_out - std_out).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 8: T not divisible by chunk_size -> ValueError
# ---------------------------------------------------------------------------


def test_non_divisible_sequence_raises():
    B, T = 1, CHUNK + 3
    x = torch.randn(B, T, D_MODEL)
    model = make_model()
    with pytest.raises(ValueError, match="divisible"):
        model(x)


# ---------------------------------------------------------------------------
# Test 9: No NaN/Inf on zeros input
# ---------------------------------------------------------------------------


def test_no_nan_on_zeros():
    B, T = 1, 4 * CHUNK
    x = torch.zeros(B, T, D_MODEL)
    model = make_model()
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all(), "NaN/Inf on zeros input"


# ---------------------------------------------------------------------------
# Test 10: No NaN/Inf on large inputs
# ---------------------------------------------------------------------------


def test_no_nan_on_large_inputs():
    B, T = 1, 4 * CHUNK
    x = torch.randn(B, T, D_MODEL) * 1e3
    model = make_model()
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all(), "NaN/Inf on large inputs"


# ---------------------------------------------------------------------------
# Test 11: Online softmax merge matches direct softmax on combined scores
# ---------------------------------------------------------------------------


def test_online_softmax_merge_correctness():
    """
    Construct two score vectors, compute (m, s, o) independently, merge with
    the paper's formula, and verify the merged normalised output equals the
    direct softmax over the concatenated scores.
    """
    torch.manual_seed(0)
    L = CHUNK
    h = HEAD_DIM
    T_left = 2 * CHUNK
    T_right = 2 * CHUNK

    S_left = torch.randn(L, T_left)
    S_right = torch.randn(L, T_right)
    V_left = torch.randn(T_left, h)
    V_right = torch.randn(T_right, h)

    # Direct reference
    S_full = torch.cat([S_left, S_right], dim=-1)
    V_full = torch.cat([V_left, V_right], dim=0)
    ref_out = torch.softmax(S_full, dim=-1) @ V_full  # (L, h)

    # Online-softmax left
    m_l = S_left.amax(dim=-1, keepdim=True)
    exp_l = torch.exp(S_left - m_l)
    s_l = exp_l.sum(dim=-1, keepdim=True)
    o_l = exp_l @ V_left  # unnormalised

    # Online-softmax right
    m_r = S_right.amax(dim=-1, keepdim=True)
    exp_r = torch.exp(S_right - m_r)
    s_r = exp_r.sum(dim=-1, keepdim=True)
    o_r = exp_r @ V_right  # unnormalised

    # Merge using paper formula
    m_merged, s_merged, o_merged = merge_two_chunks(m_l, s_l, o_l, m_r, s_r, o_r)

    assert torch.allclose(o_merged, ref_out, atol=1e-5), (
        f"Max merge diff: {(o_merged - ref_out).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 12: Multi-head (n_heads > 1) produces correct shape and finite outputs
# ---------------------------------------------------------------------------


def test_multi_head():
    torch.manual_seed(55)
    B, T = 2, 4 * CHUNK
    x = torch.randn(B, T, D_MODEL)
    model = TreeAttention(d_model=D_MODEL, n_heads=N_HEADS, chunk_size=CHUNK, causal=False)
    out = model(x)
    assert out.shape == (B, T, D_MODEL)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 13: causal=True does not crash and outputs are finite
# ---------------------------------------------------------------------------


def test_causal_no_crash():
    B, T = 1, 4 * CHUNK
    x = torch.randn(B, T, D_MODEL)
    model = make_model(causal=True)
    out = model(x)
    assert out.shape == (B, T, D_MODEL)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 14: Numerical stability — no overflow on large logits
# ---------------------------------------------------------------------------


def test_numerical_stability_large_logits():
    """
    Without the m_i (row-max) subtraction, exp(S_ij) overflows for logits > ~88.
    Verify that the log-sum-exp stabilisation keeps all outputs finite.
    """
    torch.manual_seed(3)
    B, T = 1, 4 * CHUNK
    x = torch.ones(B, T, D_MODEL) * 10.0

    model = make_model(causal=False)
    with torch.no_grad():
        nn.init.constant_(model.W_Q.weight, 1.0)
        nn.init.constant_(model.W_K.weight, 1.0)
        nn.init.constant_(model.W_V.weight, 1.0)
        nn.init.constant_(model.W_O.weight, 1.0)

    with torch.no_grad():
        out = model(x)

    assert torch.isfinite(out).all(), "Overflow detected on large logits"
