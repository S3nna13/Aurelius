"""Tests for sparse_autoencoder.py — 12 tests covering SAEConfig, SparseAutoencoder, sae_loss, SAETrainer, and utility functions."""

import pytest
import torch

from src.eval.sparse_autoencoder import (
    SAEConfig,
    SparseAutoencoder,
    sae_loss,
    SAETrainer,
    extract_features_from_model,
    find_top_activating_examples,
    compute_feature_statistics,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

D_MODEL = 64
N_FEATURES = 128
L1_COEFF = 0.001
BATCH = 4


@pytest.fixture
def config():
    return SAEConfig(d_model=D_MODEL, n_features=N_FEATURES, l1_coeff=L1_COEFF)


@pytest.fixture
def sae(config):
    torch.manual_seed(0)
    return SparseAutoencoder(config)


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(BATCH, D_MODEL)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sae_config_defaults():
    """SAEConfig has expected default values."""
    cfg = SAEConfig()
    assert cfg.d_model == 512
    assert cfg.n_features == 4096
    assert cfg.l1_coeff == 0.001
    assert cfg.learning_rate == 1e-3
    assert cfg.n_steps_warmup == 1000
    assert cfg.normalize_decoder is True


def test_sae_encode_shape(sae, x):
    """encode returns (B, n_features)."""
    features = sae.encode(x)
    assert features.shape == (BATCH, N_FEATURES)


def test_sae_encode_nonneg(sae, x):
    """encode output is non-negative (ReLU applied)."""
    features = sae.encode(x)
    assert (features >= 0).all(), "encode output should be non-negative (ReLU)"


def test_sae_decode_shape(sae, x):
    """decode returns (B, d_model)."""
    features = sae.encode(x)
    reconstructed = sae.decode(features)
    assert reconstructed.shape == (BATCH, D_MODEL)


def test_sae_forward_returns_tuple(sae, x):
    """forward returns a tuple of 3 tensors with correct shapes."""
    result = sae(x)
    assert isinstance(result, tuple)
    assert len(result) == 3
    reconstructed, features, x_centered = result
    assert reconstructed.shape == (BATCH, D_MODEL)
    assert features.shape == (BATCH, N_FEATURES)
    assert x_centered.shape == (BATCH, D_MODEL)


def test_sae_normalize_decoder(sae):
    """After normalize_decoder_weights, W_dec rows have unit norm."""
    sae.normalize_decoder_weights()
    row_norms = sae.W_dec.data.norm(dim=-1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5), (
        f"Row norms not 1.0: min={row_norms.min():.6f}, max={row_norms.max():.6f}"
    )


def test_sae_loss_scalar(sae, x):
    """sae_loss returns a scalar tensor."""
    reconstructed, features, _ = sae(x)
    total_loss, _ = sae_loss(reconstructed, x, features, L1_COEFF)
    assert total_loss.shape == torch.Size([])
    assert total_loss.item() >= 0.0


def test_sae_loss_keys(sae, x):
    """sae_loss metrics dict contains recon_loss, sparsity_loss, l0_norm."""
    reconstructed, features, _ = sae(x)
    _, metrics = sae_loss(reconstructed, x, features, L1_COEFF)
    assert "recon_loss" in metrics
    assert "sparsity_loss" in metrics
    assert "l0_norm" in metrics


def test_sae_trainer_step_keys(sae, config, x):
    """SAETrainer.train_step returns dict with total_loss, recon_loss, sparsity_loss, l0_norm."""
    trainer = SAETrainer(sae, config)
    metrics = trainer.train_step(x)
    assert "total_loss" in metrics
    assert "recon_loss" in metrics
    assert "sparsity_loss" in metrics
    assert "l0_norm" in metrics


def test_find_top_activating_examples_count():
    """find_top_activating_examples returns n_top indices."""
    torch.manual_seed(0)
    features = torch.randn(20, N_FEATURES).abs()
    n_top = 5
    indices = find_top_activating_examples(features, feature_idx=0, n_top=n_top)
    assert indices.shape == (n_top,)


def test_compute_feature_statistics_keys():
    """compute_feature_statistics returns dict with expected keys."""
    torch.manual_seed(0)
    features = torch.randn(16, N_FEATURES).abs()
    stats = compute_feature_statistics(features)
    assert "mean_l0" in stats
    assert "mean_activation" in stats
    assert "dead_features" in stats
    assert "max_activation" in stats


def test_sae_gradient_flow(sae, x):
    """Backward pass through SAE works without errors."""
    reconstructed, features, _ = sae(x)
    total_loss, _ = sae_loss(reconstructed, x, features, L1_COEFF)
    total_loss.backward()
    # Check that gradients exist
    assert sae.W_enc.grad is not None
    assert sae.W_dec.grad is not None
    assert sae.b_enc.grad is not None
    assert sae.b_dec.grad is not None
