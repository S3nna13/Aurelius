"""
Tests for InfiniAttention (arXiv:2404.07143).

Tiny config: d_model=64, n_heads=4, head_dim=16, segment_len=8.
Pure PyTorch only — no external ML libraries.
"""

import pytest
import torch

from src.model.infini_attention import InfiniAttention, _sigma

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------
D_MODEL = 64
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 16
SEG_LEN = 8
BATCH = 2
EPS = 1e-6


@pytest.fixture
def model():
    torch.manual_seed(0)
    return InfiniAttention(d_model=D_MODEL, n_heads=N_HEADS, segment_len=SEG_LEN)


def _rand_input(batch=BATCH, seq_len=SEG_LEN, seed=42):
    torch.manual_seed(seed)
    return torch.randn(batch, seq_len, D_MODEL)


# ---------------------------------------------------------------------------
# Test 1 — output shape matches input (B, T, d_model)
# ---------------------------------------------------------------------------
def test_output_shape(model):
    x = _rand_input(seq_len=SEG_LEN * 3)
    out, _ = model(x)
    assert out.shape == (BATCH, SEG_LEN * 3, D_MODEL), (
        f"Expected {(BATCH, SEG_LEN * 3, D_MODEL)}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2 — memory state shapes
# ---------------------------------------------------------------------------
def test_memory_state_shapes(model):
    x = _rand_input()
    _, (M, z) = model(x)
    assert M.shape == (N_HEADS, HEAD_DIM, HEAD_DIM), f"M shape wrong: {M.shape}"
    assert z.shape == (N_HEADS, HEAD_DIM), f"z shape wrong: {z.shape}"


# ---------------------------------------------------------------------------
# Test 3 — gradient flow: backward yields finite grads on all params incl. β
# ---------------------------------------------------------------------------
def test_gradient_flow(model):
    x = _rand_input()
    out, _ = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"
    # β specifically
    assert model.beta_param.grad is not None
    assert torch.isfinite(model.beta_param.grad).all()


# ---------------------------------------------------------------------------
# Test 4 — determinism under torch.manual_seed
# ---------------------------------------------------------------------------
def test_determinism(model):
    x = _rand_input()
    torch.manual_seed(7)
    out1, _ = model(x)
    torch.manual_seed(7)
    out2, _ = model(x)
    assert torch.allclose(out1, out2), "Non-deterministic outputs"


# ---------------------------------------------------------------------------
# Test 5 — single segment: batch=1, seq_len == segment_len
# ---------------------------------------------------------------------------
def test_single_segment():
    torch.manual_seed(1)
    m = InfiniAttention(d_model=D_MODEL, n_heads=N_HEADS, segment_len=SEG_LEN)
    x = torch.randn(1, SEG_LEN, D_MODEL)
    out, (M, z) = m(x)
    assert out.shape == (1, SEG_LEN, D_MODEL)
    assert M.shape == (N_HEADS, HEAD_DIM, HEAD_DIM)
    assert z.shape == (N_HEADS, HEAD_DIM)


# ---------------------------------------------------------------------------
# Test 6 — multiple segments (seq_len = 3 * segment_len), memory flows forward
# ---------------------------------------------------------------------------
def test_multiple_segments(model):
    x = _rand_input(seq_len=SEG_LEN * 3)
    out, (M, z) = model(x)
    assert out.shape == (BATCH, SEG_LEN * 3, D_MODEL)
    # Memory should have been updated (non-zero after processing tokens)
    assert not torch.all(M == 0), "Memory M is still all zeros after 3 segments"
    assert not torch.all(z == 0), "Memory z is still all zeros after 3 segments"


# ---------------------------------------------------------------------------
# Test 7 — None memory_state → zero init, no crash
# ---------------------------------------------------------------------------
def test_none_memory_state(model):
    x = _rand_input()
    try:
        out, state = model(x, memory_state=None)
    except Exception as exc:
        pytest.fail(f"Crashed with None memory_state: {exc}")
    assert out.shape == (BATCH, SEG_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# Test 8 — non-None memory_state is actually used (not ignored)
# ---------------------------------------------------------------------------
def test_memory_state_passed_in(model):
    x = _rand_input()
    # First pass: get a real memory state
    _, state1 = model(x)
    # Second pass with default (zero) memory
    out_zero, _ = model(x, memory_state=None)
    # Third pass with the accumulated memory — output must differ
    out_mem, _ = model(x, memory_state=state1)
    assert not torch.allclose(out_zero, out_mem, atol=1e-5), (
        "Output did not change when non-trivial memory_state was provided"
    )


# ---------------------------------------------------------------------------
# Test 9 — memory accumulates: segment 2 output differs from segment 1
# ---------------------------------------------------------------------------
def test_memory_context_effect(model):
    """Process two separate single-segment batches; the second should differ
    depending on whether we carry memory forward."""
    x1 = _rand_input(seq_len=SEG_LEN, seed=10)
    x2 = _rand_input(seq_len=SEG_LEN, seed=20)

    out2_no_mem, _ = model(x2, memory_state=None)

    _, state_after_1 = model(x1, memory_state=None)
    out2_with_mem, _ = model(x2, memory_state=state_after_1)

    assert not torch.allclose(out2_no_mem, out2_with_mem, atol=1e-5), (
        "Memory from segment 1 had no effect on segment 2 output"
    )


# ---------------------------------------------------------------------------
# Test 10 — β is in (0, 1) after sigmoid
# ---------------------------------------------------------------------------
def test_beta_in_range(model):
    beta = torch.sigmoid(model.beta_param)
    assert (beta > 0).all() and (beta < 1).all(), f"β out of (0,1): {beta}"


# ---------------------------------------------------------------------------
# Test 11 — numerical stability: no NaN/Inf on zero input
# ---------------------------------------------------------------------------
def test_no_nan_on_zero_input(model):
    x = torch.zeros(BATCH, SEG_LEN, D_MODEL)
    out, (M, z) = model(x)
    assert torch.isfinite(out).all(), "NaN/Inf on zero input"
    assert torch.isfinite(M).all(), "NaN/Inf in M on zero input"
    assert torch.isfinite(z).all(), "NaN/Inf in z on zero input"


# ---------------------------------------------------------------------------
# Test 12 — numerical stability: no NaN/Inf on large inputs (×100 scale)
# ---------------------------------------------------------------------------
def test_no_nan_on_large_input(model):
    x = _rand_input() * 100.0
    out, (M, z) = model(x)
    assert torch.isfinite(out).all(), "NaN/Inf on large input"
    assert torch.isfinite(M).all(), "NaN/Inf in M on large input"
    assert torch.isfinite(z).all(), "NaN/Inf in z on large input"


# ---------------------------------------------------------------------------
# Test 13 — σ(x) = ELU(x) + 1 is strictly positive
# ---------------------------------------------------------------------------
def test_sigma_positive():
    """σ(x) = ELU(x) + 1 must be non-negative everywhere.

    Mathematically σ(x) > 0 for all finite x, but float32 underflows to 0.0
    for very large negative inputs (ELU → -1, ELU+1 → 0 in fp32).  The
    memory retrieval formula divides by (σ(Q)·z + ε) so a zero kernel value
    is safe; the invariant that matters is σ(x) >= 0 (no negative values).
    """
    torch.manual_seed(0)
    x = torch.randn(100, 50) * 10  # wide range including very negative values
    result = _sigma(x)
    assert (result >= 0).all(), f"σ(x) has negative values; min={result.min().item():.6f}"


# ---------------------------------------------------------------------------
# Test 14 — seq_len not divisible by segment_len → padding handled without NaN
# ---------------------------------------------------------------------------
def test_non_divisible_seq_len(model):
    # 3*SEG_LEN + 3 = 27 tokens (not divisible by 8)
    T = SEG_LEN * 3 + 3
    x = _rand_input(seq_len=T)
    out, (M, z) = model(x)
    assert out.shape == (BATCH, T, D_MODEL), f"Shape mismatch for non-divisible T: {out.shape}"
    assert torch.isfinite(out).all(), "NaN/Inf when seq_len not divisible by segment_len"
    assert torch.isfinite(M).all()
    assert torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# Test 15 — memory state returned is detached from computation graph
# ---------------------------------------------------------------------------
def test_memory_state_detached(model):
    x = _rand_input()
    out, (M, z) = model(x)
    assert not M.requires_grad, "Returned M should be detached (no grad)"
    assert not z.requires_grad, "Returned z should be detached (no grad)"
