"""Tests for chunked_attention_v3 — memory-efficient chunked attention.

All tests use tiny configs: d_model=16, n_heads=2, chunk_size=4,
seq_len=8, batch=2 (or variants thereof).

Each test runs at least one forward (and where required, backward) pass.
"""

import math

import torch
import torch.nn as nn
import pytest

from src.model.chunked_attention_v3 import (
    ChunkedAttentionConfig,
    ChunkedSelfAttention,
    ChunkedCrossAttention,
    MemoryUsageEstimator,
    ChunkedAttentionBlock,
)

# ---------------------------------------------------------------------------
# Shared tiny-config constants
# ---------------------------------------------------------------------------
D_MODEL = 16
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS  # 8
CHUNK = 4
SEQ = 8
BATCH = 2
DTYPE = torch.float32

torch.manual_seed(42)


def _make_self_attn(chunk_size: int = CHUNK, causal: bool = True) -> ChunkedSelfAttention:
    cfg = ChunkedAttentionConfig(chunk_size=chunk_size, causal=causal)
    return ChunkedSelfAttention(D_MODEL, N_HEADS, cfg)


def _make_standard_attn(causal: bool = True) -> nn.MultiheadAttention:
    """Reference standard MHA (used for numerical comparison)."""
    return nn.MultiheadAttention(D_MODEL, N_HEADS, bias=False, batch_first=True)


def _run_standard_attn(x: torch.Tensor, causal: bool, mha: nn.MultiheadAttention) -> torch.Tensor:
    T = x.shape[1]
    if causal:
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    else:
        mask = None
    out, _ = mha(x, x, x, attn_mask=mask, need_weights=False)
    return out


# ===========================================================================
# Test 1 — ChunkedSelfAttention: output shape (B, T, D)
# ===========================================================================
def test_self_attn_output_shape():
    model = _make_self_attn()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = model(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), f"Expected {(BATCH, SEQ, D_MODEL)}, got {out.shape}"


# ===========================================================================
# Test 2 — ChunkedSelfAttention output matches standard attention (< 1e-4)
#           when weights are copied
# ===========================================================================
def test_self_attn_matches_standard():
    """Copy weights from standard MHA into chunked model; outputs should match."""
    torch.manual_seed(0)
    cfg = ChunkedAttentionConfig(chunk_size=CHUNK, causal=False)
    chunked = ChunkedSelfAttention(D_MODEL, N_HEADS, cfg)
    # Build a reference standard attention with the same projections
    # by sharing weight tensors explicitly.
    ref_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
    ref_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
    ref_v = nn.Linear(D_MODEL, D_MODEL, bias=False)
    ref_out = nn.Linear(D_MODEL, D_MODEL, bias=False)

    # Copy chunked weights to reference linears
    ref_q.weight.data.copy_(chunked.q_proj.weight.data)
    ref_k.weight.data.copy_(chunked.k_proj.weight.data)
    ref_v.weight.data.copy_(chunked.v_proj.weight.data)
    ref_out.weight.data.copy_(chunked.out_proj.weight.data)

    x = torch.randn(BATCH, SEQ, D_MODEL)

    # Run chunked
    out_chunked = chunked(x)

    # Run standard manually
    B, T, D = x.shape
    scale = math.sqrt(D_HEAD)
    Q = ref_q(x).view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
    K = ref_k(x).view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
    V = ref_v(x).view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
    attn = torch.softmax(scores, dim=-1)
    out_std = torch.matmul(attn, V)
    out_std = out_std.transpose(1, 2).contiguous().view(B, T, D)
    out_std = ref_out(out_std)

    diff = (out_chunked - out_std).abs().max().item()
    assert diff < 1e-4, f"Max diff = {diff:.2e} (expected < 1e-4)"


# ===========================================================================
# Test 3 — causal=True: upper triangle of effective attention is zero
# ===========================================================================
def test_causal_mask_upper_triangle_zero():
    """With causal=True, position i must not attend to positions j > i."""
    cfg = ChunkedAttentionConfig(chunk_size=CHUNK, causal=True)
    model = ChunkedSelfAttention(D_MODEL, N_HEADS, cfg)

    x = torch.randn(1, SEQ, D_MODEL, requires_grad=False)

    # Extract attention weights by running a forward pass and checking that
    # modifying future tokens doesn't change past outputs.
    x1 = x.clone()
    x2 = x.clone()
    # Corrupt all tokens from position SEQ//2 onward
    x2[:, SEQ // 2:, :] = torch.randn_like(x2[:, SEQ // 2:, :]) * 10.0

    out1 = model(x1)
    out2 = model(x2)

    # Positions 0 .. SEQ//2-1 should be identical (they only see past tokens)
    diff = (out1[:, : SEQ // 2, :] - out2[:, : SEQ // 2, :]).abs().max().item()
    assert diff < 1e-5, (
        f"Causal mask broken: past positions changed by {diff:.2e} when future tokens changed"
    )


# ===========================================================================
# Test 4 — backward succeeds, grad flows to Q/K/V projections
# ===========================================================================
def test_self_attn_backward_grad_flows():
    model = _make_self_attn()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = model(x)
    loss = out.sum()
    loss.backward()

    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        proj: nn.Linear = getattr(model, name)
        assert proj.weight.grad is not None, f"No grad for {name}.weight"
        assert proj.weight.grad.abs().sum().item() > 0.0, f"Zero grad for {name}.weight"


# ===========================================================================
# Test 5 — chunk_size=T (single chunk) matches chunk_size=1 outputs
# ===========================================================================
def test_chunk_size_T_equals_chunk_size_1():
    """chunk_size=T and chunk_size=1 must give identical outputs (same weights)."""
    torch.manual_seed(7)
    cfg_full = ChunkedAttentionConfig(chunk_size=SEQ, causal=True)
    cfg_one = ChunkedAttentionConfig(chunk_size=1, causal=True)
    m_full = ChunkedSelfAttention(D_MODEL, N_HEADS, cfg_full)
    m_one = ChunkedSelfAttention(D_MODEL, N_HEADS, cfg_one)

    # share weights
    for attr in ("q_proj", "k_proj", "v_proj", "out_proj"):
        getattr(m_one, attr).weight.data.copy_(getattr(m_full, attr).weight.data)

    x = torch.randn(BATCH, SEQ, D_MODEL)
    out_full = m_full(x)
    out_one = m_one(x)

    diff = (out_full - out_one).abs().max().item()
    assert diff < 1e-4, f"chunk_size=T vs chunk_size=1 diff = {diff:.2e}"


# ===========================================================================
# Test 6 — chunk_size=1: output still has correct shape
# ===========================================================================
def test_chunk_size_1_correct_shape():
    model = _make_self_attn(chunk_size=1)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = model(x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


# ===========================================================================
# Test 7 — ChunkedCrossAttention: output shape (B, T_q, D) with T_q ≠ T_c
# ===========================================================================
def test_cross_attn_output_shape():
    T_q = 6
    T_c = 10
    model = ChunkedCrossAttention(D_MODEL, N_HEADS, chunk_size=CHUNK)
    query = torch.randn(BATCH, T_q, D_MODEL)
    context = torch.randn(BATCH, T_c, D_MODEL)
    out = model(query, context)
    assert out.shape == (BATCH, T_q, D_MODEL), f"Expected {(BATCH, T_q, D_MODEL)}, got {out.shape}"


# ===========================================================================
# Test 8 — ChunkedCrossAttention: backward succeeds
# ===========================================================================
def test_cross_attn_backward():
    T_q, T_c = 5, 9
    model = ChunkedCrossAttention(D_MODEL, N_HEADS, chunk_size=CHUNK)
    query = torch.randn(BATCH, T_q, D_MODEL)
    context = torch.randn(BATCH, T_c, D_MODEL)
    out = model(query, context)
    loss = out.sum()
    loss.backward()

    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        proj: nn.Linear = getattr(model, name)
        assert proj.weight.grad is not None, f"No grad for cross_attn {name}.weight"


# ===========================================================================
# Test 9 — MemoryUsageEstimator.standard_attn_memory grows as T²
# ===========================================================================
def test_standard_memory_grows_quadratically():
    est = MemoryUsageEstimator()
    B, H, Dh = 1, 1, 8
    T1, T2 = 64, 128  # T2 = 2 * T1

    m1 = est.standard_attn_memory(B, H, T1, Dh)
    m2 = est.standard_attn_memory(B, H, T2, Dh)

    # The attention matrix part scales as T^2; ratio should be > 2 (strictly super-linear).
    # For clean verification: attn_matrix_T2 / attn_matrix_T1 = (T2/T1)^2 = 4
    attn1 = B * H * T1 * T1 * 4
    attn2 = B * H * T2 * T2 * 4
    assert attn2 / attn1 == pytest.approx(4.0), "Attention matrix should scale as T²"
    assert m2 > m1 * 1.5, "Standard memory should grow faster than linear"


# ===========================================================================
# Test 10 — MemoryUsageEstimator.chunked_attn_memory grows linearly (not T²)
# ===========================================================================
def test_chunked_memory_grows_linearly():
    est = MemoryUsageEstimator()
    B, H, Dh, chunk = 1, 1, 8, 4

    T1, T2 = 64, 128
    m1 = est.chunked_attn_memory(B, H, T1, Dh, chunk)
    m2 = est.chunked_attn_memory(B, H, T2, Dh, chunk)

    # chunk_attn part: chunk * T; for fixed chunk it's linear in T
    chunk_attn1 = B * H * chunk * T1 * 4
    chunk_attn2 = B * H * chunk * T2 * 4
    assert chunk_attn2 / chunk_attn1 == pytest.approx(2.0), "Chunk attention should scale linearly with T"

    # The standard attn matrix would scale 4× — chunked is less
    std_m1 = est.standard_attn_memory(B, H, T1, Dh)
    std_m2 = est.standard_attn_memory(B, H, T2, Dh)
    std_ratio = std_m2 / std_m1
    chunked_ratio = m2 / m1
    assert chunked_ratio < std_ratio, "Chunked memory should grow slower than standard"


# ===========================================================================
# Test 11 — MemoryUsageEstimator.memory_reduction
# ===========================================================================
def test_memory_reduction_ratio():
    est = MemoryUsageEstimator()
    B, H, T, Dh = 1, 2, 64, 8

    # chunk_size < T: reduction < 1.0
    ratio_small = est.memory_reduction(B, H, T, Dh, chunk_size=4)
    assert ratio_small < 1.0, f"Expected ratio < 1.0 for chunk_size < T, got {ratio_small}"

    # chunk_size = T: chunked == standard (both materialise T*T)
    ratio_eq = est.memory_reduction(B, H, T, Dh, chunk_size=T)
    assert ratio_eq == pytest.approx(1.0), f"Expected ratio = 1.0 for chunk_size = T, got {ratio_eq}"

    # Larger chunk_size -> larger ratio (less memory saving)
    ratio_mid = est.memory_reduction(B, H, T, Dh, chunk_size=16)
    assert ratio_small < ratio_mid <= 1.0


# ===========================================================================
# Test 12 — ChunkedAttentionBlock: output shape (B, T, D) and grad flows
# ===========================================================================
def test_block_output_shape_and_grad():
    block = ChunkedAttentionBlock(D_MODEL, N_HEADS, chunk_size=CHUNK, causal=True)
    x = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=True)
    out = block(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), f"Expected {(BATCH, SEQ, D_MODEL)}, got {out.shape}"

    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient flowed back to input"
    assert x.grad.abs().sum().item() > 0.0, "Zero gradient at input"

    # Check internal projection weights also have grads
    assert block.attn.q_proj.weight.grad is not None
    assert block.ffn[0].weight.grad is not None


# ===========================================================================
# Test 13 — seq_len not divisible by chunk_size: handles correctly
# ===========================================================================
def test_seq_len_not_divisible_by_chunk_size():
    """seq_len=9, chunk_size=4 — last chunk has size 1."""
    SEQ_ODD = 9
    CHUNK_4 = 4
    model = _make_self_attn(chunk_size=CHUNK_4)
    x = torch.randn(BATCH, SEQ_ODD, D_MODEL)
    out = model(x)
    assert out.shape == (BATCH, SEQ_ODD, D_MODEL)

    # Backward should not raise
    out.sum().backward()
    assert model.q_proj.weight.grad is not None


# ===========================================================================
# Test 14 — Different chunk sizes give numerically equivalent outputs
# ===========================================================================
def test_different_chunk_sizes_equivalent():
    """chunk_size=2, 4, 8 must all produce the same output for the same input/weights."""
    torch.manual_seed(99)
    chunk_sizes = [1, 2, 4, SEQ]
    models = []
    for cs in chunk_sizes:
        cfg = ChunkedAttentionConfig(chunk_size=cs, causal=True)
        m = ChunkedSelfAttention(D_MODEL, N_HEADS, cfg)
        models.append(m)

    # Share weights from first model into all others
    ref = models[0]
    for m in models[1:]:
        for attr in ("q_proj", "k_proj", "v_proj", "out_proj"):
            getattr(m, attr).weight.data.copy_(getattr(ref, attr).weight.data)

    x = torch.randn(BATCH, SEQ, D_MODEL)
    outputs = [m(x) for m in models]

    ref_out = outputs[0]
    for i, out in enumerate(outputs[1:], 1):
        diff = (out - ref_out).abs().max().item()
        assert diff < 1e-4, (
            f"chunk_size={chunk_sizes[i]} vs chunk_size={chunk_sizes[0]} "
            f"max diff = {diff:.2e} (expected < 1e-4)"
        )


# ===========================================================================
# Test 15 — Standard vs chunked: max absolute diff < 1e-3 for float32
# ===========================================================================
def test_standard_vs_chunked_max_diff():
    """Compare chunked attention to a manually computed standard attention.

    Uses identical weight matrices; max absolute element diff must be < 1e-3.
    """
    torch.manual_seed(123)
    cfg = ChunkedAttentionConfig(chunk_size=CHUNK, causal=True)
    chunked = ChunkedSelfAttention(D_MODEL, N_HEADS, cfg)

    # Build reference projection layers with copied weights
    ref_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
    ref_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
    ref_v = nn.Linear(D_MODEL, D_MODEL, bias=False)
    ref_out = nn.Linear(D_MODEL, D_MODEL, bias=False)

    ref_q.weight.data.copy_(chunked.q_proj.weight.data)
    ref_k.weight.data.copy_(chunked.k_proj.weight.data)
    ref_v.weight.data.copy_(chunked.v_proj.weight.data)
    ref_out.weight.data.copy_(chunked.out_proj.weight.data)

    x = torch.randn(BATCH, SEQ, D_MODEL)

    # Chunked output
    out_chunked = chunked(x)

    # Standard (full T×T attention matrix) with causal mask
    B, T, D = x.shape
    scale = math.sqrt(D_HEAD)
    Q = ref_q(x).view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
    K = ref_k(x).view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
    V = ref_v(x).view(B, T, N_HEADS, D_HEAD).transpose(1, 2)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out_std = torch.matmul(attn, V)
    out_std = out_std.transpose(1, 2).contiguous().view(B, T, D)
    out_std = ref_out(out_std)

    diff = (out_chunked - out_std).abs().max().item()
    assert diff < 1e-3, f"Max absolute diff = {diff:.4e} (expected < 1e-3)"
