"""Tests for SparseAutoencoder (SAE) for mechanistic interpretability."""

import pytest
import torch
import torch.optim as optim

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.eval.sparse_autoencoder import (
    SAEConfig,
    SparseAutoencoder,
    SAETrainer,
    extract_sae_features_from_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return SAEConfig(d_hidden=64, n_features=256, l1_coeff=1e-3, normalize_decoder=True)


@pytest.fixture
def sae(cfg):
    torch.manual_seed(0)
    return SparseAutoencoder(cfg)


@pytest.fixture
def small_model():
    model_cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    torch.manual_seed(0)
    return AureliusTransformer(model_cfg)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_sae_output_shapes(sae):
    """encode: (4,8,64)->(4,8,256); decode: (4,8,256)->(4,8,64)."""
    x = torch.randn(4, 8, 64)
    f = sae.encode(x)
    assert f.shape == (4, 8, 256), f"encode shape mismatch: {f.shape}"

    x_hat = sae.decode(f)
    assert x_hat.shape == (4, 8, 64), f"decode shape mismatch: {x_hat.shape}"


# ---------------------------------------------------------------------------
# Loss / metric value tests
# ---------------------------------------------------------------------------

def test_sae_reconstruction_loss_positive(sae):
    """forward returns positive reconstruction loss."""
    x = torch.randn(8, 64)
    _, _, metrics = sae(x)
    assert metrics["reconstruction_loss"] > 0.0


def test_sae_sparsity_between_0_and_1(sae):
    """sparsity metric is in [0, 1]."""
    x = torch.randn(16, 64)
    _, _, metrics = sae(x)
    assert 0.0 <= metrics["sparsity"] <= 1.0


def test_sae_l0_nonnegative(sae):
    """l0 (mean active features per token) >= 0."""
    x = torch.randn(16, 64)
    _, _, metrics = sae(x)
    assert metrics["l0"] >= 0.0


# ---------------------------------------------------------------------------
# Decoder normalization test
# ---------------------------------------------------------------------------

def test_normalize_decoder_unit_columns(sae):
    """After normalize_decoder_(), all decoder columns have norm ≈ 1.0."""
    sae.normalize_decoder_()
    # decoder.weight shape: (d_hidden, n_features); columns = dim=0
    col_norms = sae.decoder.weight.data.norm(dim=0)
    assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5), \
        f"Column norms not 1.0: min={col_norms.min():.6f}, max={col_norms.max():.6f}"


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

def test_sae_trainer_step_returns_metrics(sae):
    """train_step returns dict with required keys."""
    optimizer = optim.Adam(sae.parameters(), lr=1e-4)
    trainer = SAETrainer(sae, optimizer)
    hidden = torch.randn(4, 8, 64)
    metrics = trainer.train_step(hidden)
    required_keys = {"reconstruction_loss", "l1_loss", "total_loss", "sparsity", "l0"}
    assert required_keys.issubset(metrics.keys()), \
        f"Missing keys: {required_keys - metrics.keys()}"


def test_sae_trainer_loss_decreases(sae):
    """After 20 steps on fixed activations, final loss < initial loss."""
    torch.manual_seed(42)
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)
    trainer = SAETrainer(sae, optimizer, normalize_every=100)
    activations = [torch.randn(32, 64)]

    initial_metrics = trainer.train_step(activations[0])
    initial_loss = initial_metrics["total_loss"]

    for _ in range(19):
        final_metrics = trainer.train_step(activations[0])

    final_loss = final_metrics["total_loss"]
    assert final_loss < initial_loss, \
        f"Loss did not decrease: initial={initial_loss:.6f}, final={final_loss:.6f}"


# ---------------------------------------------------------------------------
# Feature activation stats
# ---------------------------------------------------------------------------

def test_feature_activation_stats_shapes(sae):
    """mean_activation has shape (n_features,)."""
    x = torch.randn(20, 64)
    stats = sae.feature_activation_stats(x)
    assert stats["mean_activation"].shape == (256,)
    assert stats["activation_frequency"].shape == (256,)
    assert stats["max_activation"].shape == (256,)


# ---------------------------------------------------------------------------
# Hook-based extraction
# ---------------------------------------------------------------------------

def test_extract_features_from_model_shape(small_model, sae):
    """Hook-based extraction returns (B, T, n_features)."""
    B, T = 2, 8
    input_ids = torch.randint(0, 256, (B, T))
    features = extract_sae_features_from_model(small_model, sae, input_ids, layer_idx=0)
    assert features.shape == (B, T, 256), f"Shape mismatch: {features.shape}"


# ---------------------------------------------------------------------------
# train_on_activations
# ---------------------------------------------------------------------------

def test_sae_train_on_activations_returns_list(sae):
    """Returns list of length n_steps."""
    optimizer = optim.Adam(sae.parameters(), lr=1e-4)
    trainer = SAETrainer(sae, optimizer)
    activations = [torch.randn(16, 64), torch.randn(16, 64)]
    n_steps = 10
    history = trainer.train_on_activations(activations, n_steps=n_steps)
    assert isinstance(history, list)
    assert len(history) == n_steps
