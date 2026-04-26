"""Tests for PolarQuant Stage 1 compression."""

import pytest
import torch

from src.inference.turboquant.polar_quant import PolarQuant, PolarQuantState


@pytest.fixture
def pq():
    return PolarQuant(dim=32, n_codes=16, seed=0)


def test_compress_returns_state_and_residual(pq):
    x = torch.randn(2, 4, 32)
    state, residual = pq.compress(x)
    assert isinstance(state, PolarQuantState)
    assert residual.shape == x.shape


def test_state_codes_shape(pq):
    x = torch.randn(2, 4, 32)
    state, _ = pq.compress(x)
    assert state.codes.shape == (2, 4, 32)


def test_state_mins_maxs_shape(pq):
    x = torch.randn(2, 4, 32)
    state, _ = pq.compress(x)
    assert state.mins.shape == (2, 4, 1)
    assert state.maxs.shape == (2, 4, 1)


def test_decompress_shape(pq):
    x = torch.randn(3, 8, 32)
    state, _ = pq.compress(x)
    x_hat = pq.decompress(state)
    assert x_hat.shape == x.shape


def test_residual_is_x_minus_reconstruction(pq):
    """residual + decompress(state) should equal original x (to float32 precision)."""
    torch.manual_seed(1)
    x = torch.randn(2, 4, 32)
    state, residual = pq.compress(x)
    x_hat = pq.decompress(state)
    assert torch.allclose(residual.float() + x_hat.float(), x.float(), atol=1e-5), (
        f"Max diff: {(residual.float() + x_hat.float() - x.float()).abs().max()}"
    )


def test_rotation_matrix_is_orthogonal(pq):
    """Q should satisfy Q @ Q^T = I."""
    Q = pq.Q
    eye_approx = Q @ Q.T
    assert torch.allclose(eye_approx, torch.eye(32), atol=1e-5)


def test_quantization_error_bounded(pq):
    """Reconstruction error should be < 1.0 for standard normal inputs (sanity check)."""
    torch.manual_seed(2)
    x = torch.randn(4, 8, 32)
    state, _ = pq.compress(x)
    x_hat = pq.decompress(state)
    mse = (x - x_hat).pow(2).mean().item()
    assert mse < 1.0, f"MSE {mse:.4f} too high for 16-code quantization"


def test_deterministic_across_calls(pq):
    """Same input always produces same output."""
    torch.manual_seed(3)
    x = torch.randn(2, 4, 32)
    state1, res1 = pq.compress(x)
    state2, res2 = pq.compress(x)
    assert torch.all(state1.codes == state2.codes)
    assert torch.allclose(res1, res2)
