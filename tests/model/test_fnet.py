"""Tests for FNet (Lee-Thorp et al., arXiv:2105.03824).

Covers FourierMixingLayer, FNetBlock, and FNetModel.
"""

from __future__ import annotations

import pytest
import torch
from aurelius.model.fnet import FNetBlock, FNetModel, FourierMixingLayer

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

B, T, D = 2, 16, 64  # batch, seq_len, d_model
D_FF = 128  # feed-forward dimension
N_LAYERS = 3
VOCAB = 256
MAX_SEQ = 64


@pytest.fixture()
def mixing_layer() -> FourierMixingLayer:
    return FourierMixingLayer()


@pytest.fixture()
def fnet_block() -> FNetBlock:
    return FNetBlock(d_model=D, d_ff=D_FF, dropout=0.0)


@pytest.fixture()
def fnet_model() -> FNetModel:
    return FNetModel(
        vocab_size=VOCAB,
        d_model=D,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        max_seq_len=MAX_SEQ,
        dropout=0.0,
    )


@pytest.fixture()
def sample_x() -> torch.Tensor:
    return torch.randn(B, T, D)


@pytest.fixture()
def sample_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB, (B, T))


# ---------------------------------------------------------------------------
# FourierMixingLayer tests (1–4)
# ---------------------------------------------------------------------------


def test_fourier_mixing_output_shape(mixing_layer, sample_x):
    """Test 1: output shape equals input shape."""
    out = mixing_layer(sample_x)
    assert out.shape == sample_x.shape, f"Expected {sample_x.shape}, got {out.shape}"


def test_fourier_mixing_output_is_float(mixing_layer, sample_x):
    """Test 2: output dtype is real floating-point."""
    out = mixing_layer(sample_x)
    assert not out.is_complex(), "Output should be real (not complex)"
    assert out.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16)


def test_fourier_mixing_deterministic(mixing_layer, sample_x):
    """Test 3: same input produces same output."""
    out1 = mixing_layer(sample_x)
    out2 = mixing_layer(sample_x)
    assert torch.allclose(out1, out2), "FourierMixingLayer is not deterministic"


def test_fourier_mixing_changes_input(mixing_layer, sample_x):
    """Test 4: mixing actually transforms the input (not the identity)."""
    out = mixing_layer(sample_x)
    assert not torch.allclose(out, sample_x), "FourierMixingLayer should not be the identity"


# ---------------------------------------------------------------------------
# FNetBlock tests (5–8)
# ---------------------------------------------------------------------------


def test_fnet_block_output_shape(fnet_block, sample_x):
    """Test 5: output shape is (B, T, d_model)."""
    out = fnet_block(sample_x)
    assert out.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {out.shape}"


def test_fnet_block_output_finite(fnet_block, sample_x):
    """Test 6: output contains no NaN or Inf."""
    out = fnet_block(sample_x)
    assert torch.isfinite(out).all(), "FNetBlock output contains non-finite values"


def test_fnet_block_gradient_flows(fnet_block, sample_x):
    """Test 7: gradient flows back through the block."""
    x = sample_x.requires_grad_(True)
    out = fnet_block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient reached the input"
    assert torch.isfinite(x.grad).all(), "Gradient contains non-finite values"


def test_fnet_block_single_token(fnet_block):
    """Test 8: batch=1, seq_len=1 does not crash."""
    x = torch.randn(1, 1, D)
    out = fnet_block(x)
    assert out.shape == (1, 1, D)


# ---------------------------------------------------------------------------
# FNetModel tests (9–14)
# ---------------------------------------------------------------------------


def test_fnet_model_output_shape(fnet_model, sample_ids):
    """Test 9: output shape is (B, T, d_model)."""
    out = fnet_model(sample_ids)
    assert out.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {out.shape}"


def test_fnet_model_output_finite(fnet_model, sample_ids):
    """Test 10: output contains no NaN or Inf."""
    out = fnet_model(sample_ids)
    assert torch.isfinite(out).all(), "FNetModel output contains non-finite values"


def test_fnet_model_gradient_flows(fnet_model, sample_ids):
    """Test 11: gradients flow back through the entire model."""
    out = fnet_model(sample_ids)
    loss = out.sum()
    loss.backward()
    for name, param in fnet_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for parameter '{name}'"


def test_fnet_model_return_hidden_states_length(fnet_model, sample_ids):
    """Test 12: return_hidden_states=True returns a list of length n_layers."""
    hidden = fnet_model(sample_ids, return_hidden_states=True)
    assert isinstance(hidden, list), "Expected a list"
    assert len(hidden) == N_LAYERS, f"Expected {N_LAYERS} states, got {len(hidden)}"


def test_fnet_model_hidden_state_shapes(fnet_model, sample_ids):
    """Test 13: each hidden state has shape (B, T, d_model)."""
    hidden = fnet_model(sample_ids, return_hidden_states=True)
    for i, h in enumerate(hidden):
        assert h.shape == (B, T, D), (
            f"Hidden state {i} has shape {h.shape}, expected ({B}, {T}, {D})"
        )


def test_fnet_model_different_inputs_different_outputs(fnet_model):
    """Test 14: different token sequences produce different outputs."""
    ids_a = torch.randint(0, VOCAB, (B, T))
    ids_b = torch.randint(0, VOCAB, (B, T))
    # Ensure they are actually different
    while torch.equal(ids_a, ids_b):
        ids_b = torch.randint(0, VOCAB, (B, T))
    out_a = fnet_model(ids_a)
    out_b = fnet_model(ids_b)
    assert not torch.allclose(out_a, out_b), "Different inputs produced identical outputs"
