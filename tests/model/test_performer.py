"""Tests for src/model/performer.py — FAVOR+ Performer attention.

Test inventory (15 tests):
 1.  test_orf_shape                  — get_omegas returns (m, d_head)
 2.  test_orf_approximate_orthogonal — mean |dot| of distinct rows < 0.1 when m<=d
 3.  test_orf_redraw                 — redraw() causes different omegas on next call
 4.  test_attention_noncausal_shape  — PerformerAttention non-causal (B, T, d_head)
 5.  test_attention_noncausal_finite — output contains no NaN/Inf
 6.  test_attention_causal_shape     — PerformerAttention causal (B, T, d_head)
 7.  test_attention_causal_finite    — causal output contains no NaN/Inf
 8.  test_causal_causality           — out[:t] unchanged when v[t+1:] is randomised
 9.  test_layer_shape                — PerformerLayer output (B, T, d_model)
10.  test_layer_finite               — PerformerLayer output contains no NaN/Inf
11.  test_layer_batch1               — works with B=1
12.  test_layer_seqlen1              — works with T=1 (causal and non-causal)
13.  test_layer_gradients            — loss.backward() propagates gradients
14.  test_different_seeds            — different seeds produce different omegas
15.  test_more_features_no_crash     — m > d_head does not raise
"""

import torch
import pytest

from aurelius.model.performer import (
    PerformerOrthogonalRF,
    PerformerAttention,
    PerformerLayer,
)

# ---------------------------------------------------------------------------
# Tiny test dimensions (from task spec)
# ---------------------------------------------------------------------------
D_MODEL = 64
N_HEADS = 4
D_HEAD  = 16          # d_model // n_heads
B = 2
T = 8
M = 32               # num_features for most tests


# ===========================================================================
# 1. PerformerOrthogonalRF — shape
# ===========================================================================

def test_orf_shape():
    orf = PerformerOrthogonalRF(d_head=D_HEAD, num_features=M, seed=0)
    omega = orf.get_omegas(device=torch.device("cpu"), dtype=torch.float32)
    assert omega.shape == (M, D_HEAD), f"Expected ({M}, {D_HEAD}), got {omega.shape}"


# ===========================================================================
# 2. PerformerOrthogonalRF — approximate orthogonality (m <= d)
# ===========================================================================

def test_orf_approximate_orthogonal():
    """Mean absolute off-diagonal dot product of distinct rows should be small.

    This holds tightly when m <= d_head because the rows come from exactly one
    QR block that already gives an orthonormal basis.
    """
    d = D_HEAD
    m = d   # one full orthogonal block — exact orthogonality up to chi scaling
    orf = PerformerOrthogonalRF(d_head=d, num_features=m, seed=42)
    omega = orf.get_omegas(device=torch.device("cpu"), dtype=torch.float32)

    # Normalise rows to unit length before checking dot products
    norms = omega.norm(dim=1, keepdim=True).clamp(min=1e-9)
    omega_n = omega / norms                      # (m, d) — unit rows

    gram = omega_n @ omega_n.T                   # (m, m)
    # Zero out diagonal
    mask = 1.0 - torch.eye(m)
    off_diag = (gram * mask).abs()
    mean_off = off_diag.sum() / mask.sum()

    assert mean_off.item() < 0.1, (
        f"Rows not approximately orthogonal: mean |dot| = {mean_off:.4f}"
    )


# ===========================================================================
# 3. PerformerOrthogonalRF — redraw
# ===========================================================================

def test_orf_redraw():
    orf = PerformerOrthogonalRF(d_head=D_HEAD, num_features=M, seed=None)
    dev = torch.device("cpu")
    dt  = torch.float32
    omega1 = orf.get_omegas(dev, dt).clone()
    # Same call — should return cached copy
    omega2 = orf.get_omegas(dev, dt)
    assert torch.equal(omega1, omega2), "Second call should return cached omegas"
    # Trigger redraw
    orf.redraw()
    omega3 = orf.get_omegas(dev, dt)
    # With high probability the random draw will differ
    assert not torch.equal(omega1, omega3), "Omegas should change after redraw()"


# ===========================================================================
# 4. PerformerAttention — non-causal output shape
# ===========================================================================

def test_attention_noncausal_shape():
    attn = PerformerAttention(d_head=D_HEAD, num_features=M, causal=False, seed=0)
    q = torch.randn(B, T, D_HEAD)
    k = torch.randn(B, T, D_HEAD)
    v = torch.randn(B, T, D_HEAD)
    out = attn(q, k, v)
    assert out.shape == (B, T, D_HEAD), f"Expected {(B, T, D_HEAD)}, got {out.shape}"


# ===========================================================================
# 5. PerformerAttention — non-causal finite
# ===========================================================================

def test_attention_noncausal_finite():
    attn = PerformerAttention(d_head=D_HEAD, num_features=M, causal=False, seed=1)
    q = torch.randn(B, T, D_HEAD)
    k = torch.randn(B, T, D_HEAD)
    v = torch.randn(B, T, D_HEAD)
    out = attn(q, k, v)
    assert torch.isfinite(out).all(), "Non-causal output contains NaN/Inf"


# ===========================================================================
# 6. PerformerAttention — causal output shape
# ===========================================================================

def test_attention_causal_shape():
    attn = PerformerAttention(d_head=D_HEAD, num_features=M, causal=True, seed=2)
    q = torch.randn(B, T, D_HEAD)
    k = torch.randn(B, T, D_HEAD)
    v = torch.randn(B, T, D_HEAD)
    out = attn(q, k, v)
    assert out.shape == (B, T, D_HEAD), f"Expected {(B, T, D_HEAD)}, got {out.shape}"


# ===========================================================================
# 7. PerformerAttention — causal finite
# ===========================================================================

def test_attention_causal_finite():
    attn = PerformerAttention(d_head=D_HEAD, num_features=M, causal=True, seed=3)
    q = torch.randn(B, T, D_HEAD)
    k = torch.randn(B, T, D_HEAD)
    v = torch.randn(B, T, D_HEAD)
    out = attn(q, k, v)
    assert torch.isfinite(out).all(), "Causal output contains NaN/Inf"


# ===========================================================================
# 8. Causal causality check
# ===========================================================================

def test_causal_causality():
    """out[:, :t, :] must not change when v[:, t:, :] is replaced with noise."""
    t_split = T // 2   # split point
    attn = PerformerAttention(d_head=D_HEAD, num_features=M, causal=True, seed=4)
    # Put in inference mode to disable dropout / other stochastic ops
    attn.train(False)

    torch.manual_seed(7)
    q  = torch.randn(B, T, D_HEAD)
    k  = torch.randn(B, T, D_HEAD)
    v1 = torch.randn(B, T, D_HEAD)

    with torch.no_grad():
        out1 = attn(q, k, v1)

    # Replace values after t_split with different random values
    v2 = v1.clone()
    v2[:, t_split:, :] = torch.randn(B, T - t_split, D_HEAD)

    with torch.no_grad():
        out2 = attn(q, k, v2)

    # Outputs up to t_split must be identical
    assert torch.allclose(out1[:, :t_split, :], out2[:, :t_split, :], atol=1e-5), (
        "Causal mask violated: early outputs changed when future values were modified"
    )


# ===========================================================================
# 9. PerformerLayer — output shape
# ===========================================================================

def test_layer_shape():
    layer = PerformerLayer(d_model=D_MODEL, n_heads=N_HEADS, num_features=M, causal=False, seed=0)
    x = torch.randn(B, T, D_MODEL)
    out = layer(x)
    assert out.shape == (B, T, D_MODEL), f"Expected {(B, T, D_MODEL)}, got {out.shape}"


# ===========================================================================
# 10. PerformerLayer — finite outputs
# ===========================================================================

def test_layer_finite():
    layer = PerformerLayer(d_model=D_MODEL, n_heads=N_HEADS, num_features=M, causal=True, seed=1)
    x = torch.randn(B, T, D_MODEL)
    out = layer(x)
    assert torch.isfinite(out).all(), "PerformerLayer output contains NaN/Inf"


# ===========================================================================
# 11. PerformerLayer — batch=1
# ===========================================================================

def test_layer_batch1():
    layer = PerformerLayer(d_model=D_MODEL, n_heads=N_HEADS, num_features=M, causal=False)
    x = torch.randn(1, T, D_MODEL)
    out = layer(x)
    assert out.shape == (1, T, D_MODEL)
    assert torch.isfinite(out).all()


# ===========================================================================
# 12. PerformerLayer — seq_len=1 (causal and non-causal)
# ===========================================================================

def test_layer_seqlen1():
    for causal in (False, True):
        layer = PerformerLayer(d_model=D_MODEL, n_heads=N_HEADS, num_features=M, causal=causal)
        x = torch.randn(B, 1, D_MODEL)
        out = layer(x)
        assert out.shape == (B, 1, D_MODEL), (
            f"causal={causal}: expected {(B, 1, D_MODEL)}, got {out.shape}"
        )
        assert torch.isfinite(out).all(), f"causal={causal}: output contains NaN/Inf"


# ===========================================================================
# 13. Gradient flow through PerformerLayer
# ===========================================================================

def test_layer_gradients():
    layer = PerformerLayer(d_model=D_MODEL, n_heads=N_HEADS, num_features=M, causal=False, seed=5)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient at input"
    assert torch.isfinite(x.grad).all(), "Input gradient contains NaN/Inf"

    # At least one parameter must also have received a gradient
    param_grads = [p.grad for p in layer.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "No parameter gradients computed"


# ===========================================================================
# 14. Different seeds produce different omegas
# ===========================================================================

def test_different_seeds():
    orf_a = PerformerOrthogonalRF(d_head=D_HEAD, num_features=M, seed=10)
    orf_b = PerformerOrthogonalRF(d_head=D_HEAD, num_features=M, seed=99)
    dev, dt = torch.device("cpu"), torch.float32
    omega_a = orf_a.get_omegas(dev, dt)
    omega_b = orf_b.get_omegas(dev, dt)
    assert not torch.equal(omega_a, omega_b), (
        "Different seeds should produce different omega matrices"
    )


# ===========================================================================
# 15. More features (m > d_head) — no crash
# ===========================================================================

def test_more_features_no_crash():
    m_large = D_HEAD * 4    # 64 > d_head=16
    attn = PerformerAttention(d_head=D_HEAD, num_features=m_large, causal=False, seed=6)
    q = torch.randn(B, T, D_HEAD)
    k = torch.randn(B, T, D_HEAD)
    v = torch.randn(B, T, D_HEAD)
    out = attn(q, k, v)
    assert out.shape == (B, T, D_HEAD), f"Expected {(B, T, D_HEAD)}, got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf with m > d_head"
