"""Tests for src/model/s4_ssm.py -- S4D Structured State Space Model."""

import torch
from aurelius.model.s4_ssm import S4DBlock, S4DKernel, S4DLayer

# -- default dims for all tests ------------------------------------------------
D_MODEL = 16
D_STATE = 8
BATCH = 2
SEQ_LEN = 12


# -- S4DKernel tests -----------------------------------------------------------


def test_kernel_output_shape():
    """Kernel shape should be (d_model, L)."""
    kernel = S4DKernel(D_MODEL, D_STATE)
    K = kernel.get_kernel(SEQ_LEN)
    assert K.shape == (D_MODEL, SEQ_LEN), f"Expected ({D_MODEL}, {SEQ_LEN}), got {K.shape}"


def test_kernel_values_finite():
    """Kernel must contain no NaN or Inf."""
    kernel = S4DKernel(D_MODEL, D_STATE)
    K = kernel.get_kernel(SEQ_LEN)
    assert torch.isfinite(K).all(), "Kernel contains NaN or Inf"


def test_kernel_shape_varies_with_L():
    """Changing L should change kernel length accordingly."""
    kernel = S4DKernel(D_MODEL, D_STATE)
    for L in [1, 8, 32]:
        K = kernel.get_kernel(L)
        assert K.shape == (D_MODEL, L), f"Expected ({D_MODEL}, {L}), got {K.shape}"


def test_kernel_params_have_gradients():
    """All S4DKernel parameters should receive gradients after a full layer backward.

    Uses S4DLayer (which calls both get_kernel and the D skip connection) so
    every parameter in S4DKernel participates in the forward pass.
    """
    layer = S4DLayer(D_MODEL, D_STATE)
    u = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    y = layer(u)
    loss = y.sum()
    loss.backward()
    for name, param in layer.kernel.named_parameters():
        assert param.grad is not None, f"No gradient for kernel parameter: {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for kernel parameter: {name}"


# -- S4DLayer tests ------------------------------------------------------------


def test_layer_output_shape():
    """Layer output shape should be (B, L, d_model)."""
    layer = S4DLayer(D_MODEL, D_STATE)
    u = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    y = layer(u)
    assert y.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_layer_output_finite():
    """Layer output must contain no NaN or Inf."""
    layer = S4DLayer(D_MODEL, D_STATE)
    u = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    y = layer(u)
    assert torch.isfinite(y).all(), "Layer output contains NaN or Inf"


def test_layer_causality():
    """Output at position t must not depend on inputs at positions > t.

    Strategy: perturb input at position t+1 and verify position t is unchanged.
    """
    torch.manual_seed(42)
    layer = S4DLayer(D_MODEL, D_STATE)
    layer.train(False)  # inference mode

    u = torch.randn(1, SEQ_LEN, D_MODEL)
    y_orig = layer(u)

    # Perturb future position (index 5) and check that position 4 is unchanged
    perturb_pos = 5
    check_pos = 4
    u_perturbed = u.clone()
    u_perturbed[0, perturb_pos, :] += 100.0

    y_perturbed = layer(u_perturbed)

    assert torch.allclose(y_orig[0, check_pos, :], y_perturbed[0, check_pos, :], atol=1e-5), (
        "Layer is not causal: output at position t changed when future input was perturbed"
    )


def test_layer_gradient_flows():
    """Gradients should flow back through the layer to the input."""
    layer = S4DLayer(D_MODEL, D_STATE)
    u = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    y = layer(u)
    loss = y.sum()
    loss.backward()
    assert u.grad is not None, "No gradient at input"
    assert torch.isfinite(u.grad).all(), "Non-finite gradient at input"


def test_layer_length_one():
    """Layer must work with a sequence of length 1."""
    layer = S4DLayer(D_MODEL, D_STATE)
    u = torch.randn(BATCH, 1, D_MODEL)
    y = layer(u)
    assert y.shape == (BATCH, 1, D_MODEL)
    assert torch.isfinite(y).all()


# -- S4DBlock tests ------------------------------------------------------------


def test_block_output_shape():
    """Block output shape should be (B, L, d_model)."""
    block = S4DBlock(D_MODEL, D_STATE)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = block(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_block_output_finite():
    """Block output must contain no NaN or Inf."""
    block = S4DBlock(D_MODEL, D_STATE)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = block(x)
    assert torch.isfinite(out).all(), "Block output contains NaN or Inf"


def test_block_gradient_flows():
    """Gradients should flow back through the block to the input."""
    block = S4DBlock(D_MODEL, D_STATE)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient at block input"
    assert torch.isfinite(x.grad).all(), "Non-finite gradient at block input"


def test_different_inputs_different_outputs():
    """Two distinct inputs should produce distinct outputs."""
    block = S4DBlock(D_MODEL, D_STATE)
    block.train(False)
    x1 = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    x2 = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out1 = block(x1)
    out2 = block(x2)
    assert not torch.allclose(out1, out2), "Different inputs produced identical outputs"


def test_block_with_dropout():
    """Block should run successfully with dropout > 0 during training."""
    block = S4DBlock(D_MODEL, D_STATE, dropout=0.1)
    block.train(True)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = block(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert torch.isfinite(out).all()


def test_block_batch1_seq4():
    """Block must work with B=1, L=4."""
    block = S4DBlock(D_MODEL, D_STATE)
    x = torch.randn(1, 4, D_MODEL)
    out = block(x)
    assert out.shape == (1, 4, D_MODEL)
    assert torch.isfinite(out).all()
