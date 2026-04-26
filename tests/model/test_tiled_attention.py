"""
Tests for src/model/tiled_attention.py

Uses:
  d_model=16, n_heads=4 (d_head=4), T=8, B=2, tile_size=4
"""

import math

import torch

from src.model.tiled_attention import (
    AttentionEquivalenceChecker,
    OnlineSoftmax,
    TiledAttention,
    TiledAttentionConfig,
    TiledAttentionModule,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

B, H, T, D_HEAD = 2, 4, 8, 4
D_MODEL = H * D_HEAD  # 16
TILE = 4
SCALE = 1.0 / math.sqrt(D_HEAD)

_checker = AttentionEquivalenceChecker()


def _qkv(requires_grad: bool = False):
    """Return random Q, K, V of shape [B, H, T, D_HEAD]."""
    Q = torch.randn(B, H, T, D_HEAD, dtype=torch.float32)
    K = torch.randn(B, H, T, D_HEAD, dtype=torch.float32)
    V = torch.randn(B, H, T, D_HEAD, dtype=torch.float32)
    if requires_grad:
        Q.requires_grad_(True)
        K.requires_grad_(True)
        V.requires_grad_(True)
    return Q, K, V


# ===========================================================================
# 1  OnlineSoftmax update returns correct shapes for m, l, O
# ===========================================================================


def test_online_softmax_update_shapes():
    """OnlineSoftmax.update returns tensors with correct shapes."""
    tq, tk = 4, 4
    m = torch.full((B, H, tq, 1), -1e9)
    item = torch.zeros(B, H, tq, 1)
    O = torch.zeros(B, H, tq, D_HEAD)  # noqa: E741
    Q_i = torch.randn(B, H, tq, D_HEAD)
    K_b = torch.randn(B, H, tk, D_HEAD)
    V_b = torch.randn(B, H, tk, D_HEAD)

    m_new, l_new, O_new = OnlineSoftmax.update(m, item, O, Q_i, K_b, V_b, SCALE)  # noqa: E741

    assert m_new.shape == (B, H, tq, 1), f"m_new shape {m_new.shape}"
    assert l_new.shape == (B, H, tq, 1), f"l_new shape {l_new.shape}"
    assert O_new.shape == (B, H, tq, D_HEAD), f"O_new shape {O_new.shape}"


# ===========================================================================
# 2  OnlineSoftmax denominator is positive after update
# ===========================================================================


def test_online_softmax_l_positive():
    """Denominator l is strictly positive after an update."""
    tq, tk = 4, 4
    m = torch.full((B, H, tq, 1), -1e9)
    item = torch.zeros(B, H, tq, 1)
    O = torch.zeros(B, H, tq, D_HEAD)  # noqa: E741
    Q_i = torch.randn(B, H, tq, D_HEAD)
    K_b = torch.randn(B, H, tk, D_HEAD)
    V_b = torch.randn(B, H, tk, D_HEAD)

    _, l_new, _ = OnlineSoftmax.update(m, item, O, Q_i, K_b, V_b, SCALE)  # noqa: E741
    assert (l_new > 0).all(), "l should be positive after update"


# ===========================================================================
# 3  OnlineSoftmax single block matches standard softmax
# ===========================================================================


def test_online_softmax_single_block_matches_standard():
    """
    Running OnlineSoftmax over a single block should match standard
    row-wise softmax attention.
    """
    Q_i = torch.randn(B, H, T, D_HEAD)
    K_b = torch.randn(B, H, T, D_HEAD)
    V_b = torch.randn(B, H, T, D_HEAD)

    m = torch.full((B, H, T, 1), -1e9)
    item = torch.zeros(B, H, T, 1)
    O = torch.zeros(B, H, T, D_HEAD)  # noqa: E741

    _, _, O_online = OnlineSoftmax.update(m, item, O, Q_i, K_b, V_b, SCALE)  # noqa: E741

    scores = torch.matmul(Q_i, K_b.transpose(-2, -1)) * SCALE
    P_ref = torch.softmax(scores, dim=-1)
    O_ref = torch.matmul(P_ref, V_b)

    err = (O_online - O_ref).abs().max().item()
    assert err < 1e-5, f"Single-block online softmax error {err:.3e}"


# ===========================================================================
# 4  TiledAttention forward – output shape [B, H, T, d_head]
# ===========================================================================


def test_tiled_attention_forward_shape():
    """TiledAttention.apply returns shape [B, H, T, d_head]."""
    Q, K, V = _qkv()
    O = TiledAttention.apply(Q, K, V, SCALE, True, TILE)  # noqa: E741
    assert O.shape == (B, H, T, D_HEAD), f"Output shape {O.shape}"  # noqa: E741


# ===========================================================================
# 5  TiledAttention causal=True upper-triangular masked
# ===========================================================================


def test_tiled_attention_causal_mask():
    """
    With causal=True the output at position t must not depend on
    future positions.  Verify by changing future K/V and checking
    earlier positions are unchanged.
    """
    Q, K, V = _qkv()
    O1 = TiledAttention.apply(Q, K, V, SCALE, True, TILE)

    K2 = K.clone()
    V2 = V.clone()
    K2[:, :, -1, :] = torch.randn_like(K2[:, :, -1, :])
    V2[:, :, -1, :] = torch.randn_like(V2[:, :, -1, :])
    O2 = TiledAttention.apply(Q, K2, V2, SCALE, True, TILE)

    assert torch.allclose(O1[:, :, :-1, :], O2[:, :, :-1, :], atol=1e-5), (
        "Causal mask violated: earlier positions changed when future K/V changed"
    )


# ===========================================================================
# 6  TiledAttention vs standard_attention: max_abs_error < 1e-4
# ===========================================================================


def test_tiled_vs_standard_causal():
    """Tiled causal attention matches reference within 1e-4."""
    Q, K, V = _qkv()
    O_tiled = TiledAttention.apply(Q, K, V, SCALE, True, TILE)
    O_ref = _checker.standard_attention(Q, K, V, causal=True)
    err = _checker.max_abs_error(O_tiled, O_ref)
    assert err < 1e-4, f"Causal max_abs_error={err:.3e}"


# ===========================================================================
# 7  TiledAttention tile_size=1 (degenerate case) matches standard
# ===========================================================================


def test_tiled_attention_tile_size_1():
    """tile_size=1 must still produce correct output."""
    Q, K, V = _qkv()
    O_tiled = TiledAttention.apply(Q, K, V, SCALE, True, 1)
    O_ref = _checker.standard_attention(Q, K, V, causal=True)
    err = _checker.max_abs_error(O_tiled, O_ref)
    assert err < 1e-4, f"tile_size=1 max_abs_error={err:.3e}"


# ===========================================================================
# 8  TiledAttention tile_size=T (single tile) matches standard
# ===========================================================================


def test_tiled_attention_tile_size_T():
    """tile_size equal to T (single tile) must produce correct output."""
    Q, K, V = _qkv()
    O_tiled = TiledAttention.apply(Q, K, V, SCALE, True, T)
    O_ref = _checker.standard_attention(Q, K, V, causal=True)
    err = _checker.max_abs_error(O_tiled, O_ref)
    assert err < 1e-4, f"tile_size=T max_abs_error={err:.3e}"


# ===========================================================================
# 9  TiledAttention backward gradient flows (autograd)
# ===========================================================================


def test_tiled_attention_backward_gradient_flows():
    """Loss.backward() must succeed and produce non-None gradients."""
    Q, K, V = _qkv(requires_grad=True)
    O = TiledAttention.apply(Q, K, V, SCALE, True, TILE)  # noqa: E741
    loss = O.sum()  # noqa: E741
    loss.backward()
    assert Q.grad is not None, "dQ is None"
    assert K.grad is not None, "dK is None"
    assert V.grad is not None, "dV is None"


# ===========================================================================
# 10  TiledAttention backward dQ shape matches Q
# ===========================================================================


def test_tiled_attention_backward_dQ_shape():
    """dQ must have the same shape as Q."""
    Q, K, V = _qkv(requires_grad=True)
    O = TiledAttention.apply(Q, K, V, SCALE, True, TILE)  # noqa: E741
    O.sum().backward()  # noqa: E741
    assert Q.grad.shape == Q.shape, f"dQ shape {Q.grad.shape} != Q shape {Q.shape}"


# ===========================================================================
# 11  TiledAttentionModule forward output shape [B, T, d_model]
# ===========================================================================


def test_tiled_module_forward_shape():
    """TiledAttentionModule output shape is [B, T, d_model]."""
    model = TiledAttentionModule(D_MODEL, H, tile_size=TILE, causal=True)
    x = torch.randn(B, T, D_MODEL)
    y = model(x)
    assert y.shape == (B, T, D_MODEL), f"Output shape {y.shape}"


# ===========================================================================
# 12  TiledAttentionModule output matches naive MHA within tolerance
# ===========================================================================


def test_tiled_module_matches_naive_mha():
    """
    TiledAttentionModule must agree with a naive MHA that reuses the
    same projection weights.
    """
    torch.manual_seed(0)
    model = TiledAttentionModule(D_MODEL, H, tile_size=TILE, causal=True)
    model.train(False)

    x = torch.randn(B, T, D_MODEL)

    with torch.no_grad():
        y_tiled = model(x)

        Q = model._split_heads(model.W_q(x))
        K = model._split_heads(model.W_k(x))
        V = model._split_heads(model.W_v(x))
        O_ref = _checker.standard_attention(Q, K, V, causal=True)
        y_ref = model.W_o(model._merge_heads(O_ref))

    err = _checker.max_abs_error(y_tiled, y_ref)
    assert err < 1e-4, f"Module vs naive MHA max_abs_error={err:.3e}"


# ===========================================================================
# 13  TiledAttentionModule backward loss step runs
# ===========================================================================


def test_tiled_module_backward_loss_step():
    """A full forward + backward + optimizer step must complete."""
    model = TiledAttentionModule(D_MODEL, H, tile_size=TILE, causal=True)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(B, T, D_MODEL)

    optim.zero_grad()
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    optim.step()

    assert loss.item() >= 0, "Loss must be non-negative"


# ===========================================================================
# 14  AttentionEquivalenceChecker max_abs_error is 0 for identical inputs
# ===========================================================================


def test_equivalence_checker_zero_error_identical():
    """max_abs_error must be 0.0 when both tensors are identical."""
    t = torch.randn(B, H, T, D_HEAD)
    err = _checker.max_abs_error(t, t)
    assert err == 0.0, f"Expected 0.0, got {err}"


# ===========================================================================
# 15  AttentionEquivalenceChecker standard_attention shape correct
# ===========================================================================


def test_equivalence_checker_standard_attention_shape():
    """standard_attention must return shape [B, H, T, d_head]."""
    Q, K, V = _qkv()
    O = _checker.standard_attention(Q, K, V, causal=True)  # noqa: E741
    assert O.shape == (B, H, T, D_HEAD), f"Shape {O.shape}"  # noqa: E741


# ===========================================================================
# 16  TiledAttentionConfig defaults
# ===========================================================================


def test_tiled_attention_config_defaults():
    """TiledAttentionConfig must have the specified default values."""
    cfg = TiledAttentionConfig()
    assert cfg.d_model == 32
    assert cfg.n_heads == 4
    assert cfg.tile_size == 4
    assert cfg.causal is True


# ===========================================================================
# 17  TiledAttention non-causal mode (no mask)
# ===========================================================================


def test_tiled_attention_non_causal_matches_standard():
    """Non-causal tiled attention must match standard full attention."""
    Q, K, V = _qkv()
    O_tiled = TiledAttention.apply(Q, K, V, SCALE, False, TILE)
    O_ref = _checker.standard_attention(Q, K, V, causal=False)
    err = _checker.max_abs_error(O_tiled, O_ref)
    assert err < 1e-4, f"Non-causal max_abs_error={err:.3e}"


# ===========================================================================
# 18  (bonus) max_abs_error is > 0 for different tensors
# ===========================================================================


def test_equivalence_checker_nonzero_error_different():
    """max_abs_error must be > 0 when tensors differ."""
    a = torch.zeros(4)
    b = torch.ones(4)
    err = _checker.max_abs_error(a, b)
    assert err > 0.0, "Expected non-zero error for different tensors"
