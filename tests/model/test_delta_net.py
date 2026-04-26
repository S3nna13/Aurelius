"""Tests for DeltaNet — Delta Rule Linear Transformer (Yang et al. 2024).

Reference: arXiv:2406.06484.
"""

import torch

from src.model.delta_net import DeltaNetBlock, DeltaNetCell, DeltaNetLayer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

D = 16  # d_head for cell tests
B = 2  # batch size
T = 12  # sequence length
DM = 64  # d_model for layer/block tests
NH = 4  # n_heads for layer/block tests
DFF = 128  # d_ff for block tests


def make_cell() -> DeltaNetCell:
    return DeltaNetCell(d_head=D)


def make_layer(n_heads: int = NH) -> DeltaNetLayer:
    return DeltaNetLayer(d_model=DM, n_heads=n_heads)


def make_block() -> DeltaNetBlock:
    return DeltaNetBlock(d_model=DM, n_heads=NH, d_ff=DFF)


def zero_state(batch: int = 1, n_heads: int = 1, d_head: int = D) -> torch.Tensor:
    """Return a zero weight-matrix state."""
    return torch.zeros(batch, n_heads, d_head, d_head)


# ---------------------------------------------------------------------------
# DeltaNetCell tests
# ---------------------------------------------------------------------------


def test_cell_output_shape():
    """DeltaNetCell.step: o shape is (B, H, d_head)."""
    cell = make_cell()
    q = torch.randn(B, NH, D)
    k = torch.randn(B, NH, D)
    v = torch.randn(B, NH, D)
    beta = torch.sigmoid(torch.randn(B, NH))
    W = zero_state(B, NH, D)
    o, W_new = cell.step(q, k, v, beta, W)
    assert o.shape == (B, NH, D), f"o shape {o.shape} != {(B, NH, D)}"


def test_cell_w_new_shape():
    """DeltaNetCell.step: W_new shape is (B, H, d_head, d_head)."""
    cell = make_cell()
    q = torch.randn(B, NH, D)
    k = torch.randn(B, NH, D)
    v = torch.randn(B, NH, D)
    beta = torch.sigmoid(torch.randn(B, NH))
    W = zero_state(B, NH, D)
    _, W_new = cell.step(q, k, v, beta, W)
    assert W_new.shape == (B, NH, D, D), f"W_new shape {W_new.shape} != {(B, NH, D, D)}"


def test_cell_output_finite():
    """DeltaNetCell.step: output o must be finite (no NaN/Inf)."""
    cell = make_cell()
    q = torch.randn(B, NH, D)
    k = torch.nn.functional.normalize(torch.randn(B, NH, D), dim=-1)
    v = torch.randn(B, NH, D)
    beta = torch.sigmoid(torch.randn(B, NH))
    W = zero_state(B, NH, D)
    o, _ = cell.step(q, k, v, beta, W)
    assert torch.isfinite(o).all(), "Cell output contains NaN or Inf"


def test_cell_zero_state_output_zero():
    """With W_prev = 0, prediction = W k = 0, error = v.
    W_new = beta * (v outer k). Output o = W_new @ q.
    When k is orthogonal to q, dot(k, q) = 0 so o = 0."""
    cell = make_cell()
    # Craft k orthogonal to q
    q = torch.zeros(1, 1, D)
    q[..., 0] = 1.0  # q = e_0
    k = torch.zeros(1, 1, D)
    k[..., 1] = 1.0  # k = e_1  (orthogonal to q)
    v = torch.randn(1, 1, D)
    beta = torch.sigmoid(torch.randn(1, 1))
    W = zero_state(1, 1, D)
    o, _ = cell.step(q, k, v, beta, W)
    assert torch.allclose(o, torch.zeros_like(o), atol=1e-6), (
        "With W=0 and k orthogonal to q, output should be zero"
    )


# ---------------------------------------------------------------------------
# DeltaNetLayer tests
# ---------------------------------------------------------------------------


def test_layer_output_shape():
    """DeltaNetLayer.forward: output shape is (B, T, d_model)."""
    layer = make_layer()
    x = torch.randn(B, T, DM)
    out = layer(x)
    assert out.shape == (B, T, DM), f"Layer output shape {out.shape} != {(B, T, DM)}"


def test_layer_output_finite():
    """DeltaNetLayer.forward: output must be finite."""
    layer = make_layer()
    x = torch.randn(B, T, DM)
    out = layer(x)
    assert torch.isfinite(out).all(), "Layer output contains NaN or Inf"


def test_layer_causal():
    """DeltaNetLayer is causal: outputs for positions 0..T-1 must be identical
    whether or not extra future tokens are appended."""
    layer = make_layer()
    layer.train(False)  # inference mode

    torch.manual_seed(0)
    x_short = torch.randn(1, T, DM)
    x_long = torch.cat([x_short, torch.randn(1, 5, DM)], dim=1)

    with torch.no_grad():
        out_short = layer(x_short)  # (1, T, DM)
        out_long = layer(x_long)  # (1, T+5, DM)

    assert torch.allclose(out_short, out_long[:, :T, :], atol=1e-5), (
        "Layer is not causal: earlier outputs change when future tokens are appended"
    )


def test_layer_gradient_flow():
    """Backward pass through DeltaNetLayer must produce non-None, finite gradients."""
    layer = make_layer()
    x = torch.randn(B, T, DM, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None, "No gradient reached input"
    assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf"


def test_layer_batch_one():
    """DeltaNetLayer works with batch size 1."""
    layer = make_layer()
    x = torch.randn(1, T, DM)
    out = layer(x)
    assert out.shape == (1, T, DM)


def test_layer_seq_len_one():
    """DeltaNetLayer works with sequence length 1."""
    layer = make_layer()
    x = torch.randn(B, 1, DM)
    out = layer(x)
    assert out.shape == (B, 1, DM)


# ---------------------------------------------------------------------------
# DeltaNetBlock tests
# ---------------------------------------------------------------------------


def test_block_output_shape():
    """DeltaNetBlock.forward: output shape is (B, T, d_model)."""
    block = make_block()
    x = torch.randn(B, T, DM)
    out = block(x)
    assert out.shape == (B, T, DM), f"Block output shape {out.shape} != {(B, T, DM)}"


def test_block_output_finite():
    """DeltaNetBlock.forward: output must be finite."""
    block = make_block()
    x = torch.randn(B, T, DM)
    out = block(x)
    assert torch.isfinite(out).all(), "Block output contains NaN or Inf"


def test_block_residual_nontrivial():
    """DeltaNetBlock: output should not equal input (transform is active)."""
    block = make_block()
    x = torch.randn(B, T, DM)
    out = block(x)
    assert not torch.allclose(out, x), "Block output equals input — transformation appears trivial"


def test_block_gradient_flow():
    """Backward pass through DeltaNetBlock must produce non-None, finite gradients."""
    block = make_block()
    x = torch.randn(B, T, DM, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None, "No gradient reached input"
    assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf"


def test_layer_n_heads_one():
    """DeltaNetLayer edge case: n_heads=1 should work correctly."""
    layer = DeltaNetLayer(d_model=DM, n_heads=1)
    x = torch.randn(B, T, DM)
    out = layer(x)
    assert out.shape == (B, T, DM), f"n_heads=1 output shape {out.shape} != {(B, T, DM)}"
    assert torch.isfinite(out).all(), "n_heads=1 output contains NaN or Inf"
