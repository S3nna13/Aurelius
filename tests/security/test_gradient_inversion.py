"""Tests for the gradient inversion attack module (gradient_inversion.py)."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.security.gradient_inversion import GradientInverter

# ---------------------------------------------------------------------------
# Shared tiny config and fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

N_TOKENS = 4   # short sequences keep tests fast
SEED = 42


@pytest.fixture
def model():
    torch.manual_seed(SEED)
    m = AureliusTransformer(TINY_CFG)
    m.train(False)   # inference mode
    return m


@pytest.fixture
def inverter(model):
    return GradientInverter(model, loss_fn=None)


@pytest.fixture
def input_ids():
    torch.manual_seed(SEED)
    return torch.randint(0, TINY_CFG.vocab_size, (1, N_TOKENS))


@pytest.fixture
def labels(input_ids):
    return input_ids.clone()


@pytest.fixture
def target_grads(model, inverter, input_ids, labels):
    return inverter.compute_gradients(model, input_ids, labels)


# ---------------------------------------------------------------------------
# 1. compute_gradients returns a flat 1-D tensor of correct shape
# ---------------------------------------------------------------------------

def test_compute_gradients_shape(target_grads, model):
    total_params = sum(p.numel() for p in model.parameters())
    assert target_grads.ndim == 1
    assert target_grads.numel() == total_params


# ---------------------------------------------------------------------------
# 2. Gradient vector is non-zero
# ---------------------------------------------------------------------------

def test_compute_gradients_nonzero(target_grads):
    assert target_grads.norm().item() > 0.0


# ---------------------------------------------------------------------------
# 3. invert runs without error and returns tensor of shape (1, T, d_model)
# ---------------------------------------------------------------------------

def test_invert_output_shape(inverter, target_grads):
    result = inverter.invert(target_grads, n_tokens=N_TOKENS, n_steps=5, lr=0.01)
    assert result.shape == (1, N_TOKENS, TINY_CFG.d_model)


# ---------------------------------------------------------------------------
# 4. reconstruction_error returns a non-negative float
# ---------------------------------------------------------------------------

def test_reconstruction_error_nonnegative(inverter, target_grads):
    torch.manual_seed(0)
    x_true = torch.randn(1, N_TOKENS, TINY_CFG.d_model)
    x_rec = inverter.invert(target_grads, n_tokens=N_TOKENS, n_steps=5, lr=0.01)
    err = GradientInverter.reconstruction_error(x_true, x_rec)
    assert isinstance(err, float)
    assert err >= 0.0


# ---------------------------------------------------------------------------
# 5. reconstruction_error is 0.0 for identical inputs
# ---------------------------------------------------------------------------

def test_reconstruction_error_zero_for_identical():
    x = torch.randn(1, N_TOKENS, TINY_CFG.d_model)
    err = GradientInverter.reconstruction_error(x, x)
    assert abs(err) < 1e-6


# ---------------------------------------------------------------------------
# 6. More optimisation steps reduce the gradient-matching loss
# ---------------------------------------------------------------------------

def test_more_steps_reduces_loss(inverter, target_grads):
    """Verify that increasing n_steps makes the attack converge further.

    Uses the internal loss history from the same proxy to compare the
    attack objective at step 1 vs the final step after 50 steps.
    """
    torch.manual_seed(1)
    _, loss_history = inverter.invert(
        target_grads,
        n_tokens=N_TOKENS,
        n_steps=50,
        lr=0.05,
        _return_loss_history=True,
    )
    # Loss at step 0 should be >= loss at the final step
    assert loss_history[-1] <= loss_history[0], (
        f"Expected loss to decrease: first={loss_history[0]:.6f}, "
        f"last={loss_history[-1]:.6f}"
    )


# ---------------------------------------------------------------------------
# 7. Gradients from different inputs are different
# ---------------------------------------------------------------------------

def test_gradients_differ_for_different_inputs(inverter, model):
    torch.manual_seed(10)
    ids_a = torch.randint(0, TINY_CFG.vocab_size, (1, N_TOKENS))
    torch.manual_seed(20)
    ids_b = torch.randint(0, TINY_CFG.vocab_size, (1, N_TOKENS))

    grads_a = inverter.compute_gradients(model, ids_a, ids_a.clone())
    grads_b = inverter.compute_gradients(model, ids_b, ids_b.clone())

    assert not torch.allclose(grads_a, grads_b)


# ---------------------------------------------------------------------------
# 8. No NaN or Inf in reconstructed embeddings
# ---------------------------------------------------------------------------

def test_no_nan_inf_in_reconstruction(inverter, target_grads):
    result = inverter.invert(target_grads, n_tokens=N_TOKENS, n_steps=10, lr=0.01)
    assert torch.isfinite(result).all(), "Reconstructed embeddings contain NaN or Inf"


# ---------------------------------------------------------------------------
# 9. Attack works with batch size 1
# ---------------------------------------------------------------------------

def test_batch_size_one(model, inverter):
    torch.manual_seed(7)
    ids = torch.randint(0, TINY_CFG.vocab_size, (1, N_TOKENS))
    grads = inverter.compute_gradients(model, ids, ids.clone())
    result = inverter.invert(grads, n_tokens=N_TOKENS, n_steps=5, lr=0.01)
    assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# 10. Reconstructed embedding dtype is float32
# ---------------------------------------------------------------------------

def test_dtype_float32(inverter, target_grads):
    result = inverter.invert(target_grads, n_tokens=N_TOKENS, n_steps=5, lr=0.01)
    assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# 11. Deterministic with the same seed
# ---------------------------------------------------------------------------

def test_deterministic_with_same_seed(inverter, target_grads):
    torch.manual_seed(99)
    r1 = inverter.invert(target_grads, n_tokens=N_TOKENS, n_steps=10, lr=0.01)
    torch.manual_seed(99)
    r2 = inverter.invert(target_grads, n_tokens=N_TOKENS, n_steps=10, lr=0.01)
    assert torch.allclose(r1, r2)


# ---------------------------------------------------------------------------
# 12. GradientInverter can be instantiated
# ---------------------------------------------------------------------------

def test_gradient_inverter_instantiation(model):
    inverter = GradientInverter(model)
    assert isinstance(inverter, GradientInverter)
    assert inverter.d_model == TINY_CFG.d_model
    assert inverter.vocab_size == TINY_CFG.vocab_size


# ---------------------------------------------------------------------------
# 13. compute_gradients clears model gradients after returning
# ---------------------------------------------------------------------------

def test_compute_gradients_clears_model_grads(model, inverter, input_ids, labels):
    inverter.compute_gradients(model, input_ids, labels)
    for p in model.parameters():
        assert p.grad is None or p.grad.norm().item() == 0.0


# ---------------------------------------------------------------------------
# 14. reconstruction_error handles zero-norm true embedding
# ---------------------------------------------------------------------------

def test_reconstruction_error_zero_norm_true_embed():
    x_true = torch.zeros(1, N_TOKENS, TINY_CFG.d_model)
    x_rec = torch.randn(1, N_TOKENS, TINY_CFG.d_model)
    err = GradientInverter.reconstruction_error(x_true, x_rec)
    assert isinstance(err, float)
    assert err >= 0.0
