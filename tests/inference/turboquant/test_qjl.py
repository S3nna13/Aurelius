"""Tests for QJL Gaussian sketch inner product estimator."""
import math
import torch
import pytest
from src.inference.turboquant.qjl import QJLSketch


@pytest.fixture
def sketch():
    return QJLSketch(dim=64, sketch_dim=32, seed=0)


def test_compress_keys_signs_shape(sketch):
    residual = torch.randn(2, 4, 64)
    signs, norms = sketch.compress_keys(residual)
    assert signs.shape == (2, 4, 32)


def test_compress_keys_norms_shape(sketch):
    residual = torch.randn(2, 4, 64)
    signs, norms = sketch.compress_keys(residual)
    assert norms.shape == (2, 4)


def test_signs_dtype(sketch):
    residual = torch.randn(2, 4, 64)
    signs, _ = sketch.compress_keys(residual)
    assert signs.dtype == torch.int8


def test_signs_are_binary(sketch):
    residual = torch.randn(2, 4, 64)
    signs, _ = sketch.compress_keys(residual)
    unique_vals = signs.float().unique()
    # sign() returns -1, 0, or 1; all should be in {-1, 0, 1}
    assert all(v.item() in {-1.0, 0.0, 1.0} for v in unique_vals)


def test_norms_positive(sketch):
    residual = torch.randn(2, 4, 64)
    _, norms = sketch.compress_keys(residual)
    assert (norms >= 0).all()


def test_estimator_unbiased():
    """QJL estimator should be approximately unbiased over many trials.

    Each trial uses a fresh S matrix (different seed) so the empirical mean
    converges to E[estimate] = <key, query>.
    """
    torch.manual_seed(42)
    dim, sketch_dim = 64, 128

    key = torch.randn(dim)
    query = torch.randn(dim)
    true_dot = (key * query).sum().item()

    n_trials = 200
    estimates = []
    for i in range(n_trials):
        # Fresh sketch each trial so we sample the expectation over S
        sketch = QJLSketch(dim=dim, sketch_dim=sketch_dim, seed=i)
        signs, norms = sketch.compress_keys(key.unsqueeze(0))
        est = sketch.estimate_attention(signs, norms, query.unsqueeze(0))
        estimates.append(est.item())

    mean_est = sum(estimates) / n_trials
    # With 200 independent sketches mean should be within 10% of true
    assert abs(mean_est - true_dot) < abs(true_dot) * 0.10 + 0.1, \
        f"Mean estimate {mean_est:.4f} too far from true {true_dot:.4f}"


def test_s_matrix_is_gaussian(sketch):
    """S matrix should be normally distributed (not ±1)."""
    # If S were ±1, std would be 1.0; for N(0,1) it's also 1.0
    # Better check: values are not all ±1
    s_flat = sketch.S.flatten()
    unique_count = len(s_flat.unique())
    assert unique_count > 100, "S matrix looks binary (too few unique values)"


def test_zero_residual_gives_zero_estimate(sketch):
    """Zero residual has zero norm, so estimate should be zero."""
    residual = torch.zeros(1, 1, 64)
    query = torch.randn(1, 1, 64)
    signs, norms = sketch.compress_keys(residual)
    est = sketch.estimate_attention(signs, norms, query)
    assert torch.allclose(est, torch.zeros_like(est), atol=1e-6)
