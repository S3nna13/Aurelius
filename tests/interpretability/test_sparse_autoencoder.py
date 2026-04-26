"""Tests for src/interpretability/sparse_autoencoder.py"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.interpretability.sparse_autoencoder import (
    SAEConfig,
    SparseAutoencoder,
)


@pytest.fixture
def config() -> SAEConfig:
    return SAEConfig(input_dim=32, hidden_dim=64, sparsity_coef=1e-3, lr=1e-3)


@pytest.fixture
def sae(config) -> SparseAutoencoder:
    return SparseAutoencoder(config)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_sae_is_nn_module(sae):
    assert isinstance(sae, nn.Module)


def test_sae_has_encoder_decoder(sae):
    assert hasattr(sae, "encoder")
    assert hasattr(sae, "decoder")


def test_encoder_shape(config, sae):
    assert sae.encoder.in_features == config.input_dim
    assert sae.encoder.out_features == config.hidden_dim


def test_decoder_shape(config, sae):
    assert sae.decoder.in_features == config.hidden_dim
    assert sae.decoder.out_features == config.input_dim


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


def test_forward_output_shapes(config, sae):
    x = torch.randn(8, config.input_dim)
    recon, hidden = sae(x)
    assert recon.shape == x.shape
    assert hidden.shape == (8, config.hidden_dim)


def test_forward_recon_is_real(config, sae):
    x = torch.randn(4, config.input_dim)
    recon, hidden = sae(x)
    assert not torch.isnan(recon).any()
    assert not torch.isnan(hidden).any()


def test_hidden_nonnegative_relu(config, sae):
    """Hidden activations must be >= 0 because of ReLU."""
    x = torch.randn(16, config.input_dim)
    _, hidden = sae(x)
    assert (hidden >= 0).all()


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------


def test_loss_is_scalar(config, sae):
    x = torch.randn(8, config.input_dim)
    recon, hidden = sae(x)
    loss_val = sae.loss(x, recon, hidden)
    assert loss_val.shape == torch.Size([])


def test_loss_is_positive(config, sae):
    x = torch.randn(8, config.input_dim)
    recon, hidden = sae(x)
    loss_val = sae.loss(x, recon, hidden)
    assert loss_val.item() >= 0.0


def test_loss_decreases_with_training(config):
    sae = SparseAutoencoder(config)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)
    torch.manual_seed(0)
    x = torch.randn(32, config.input_dim)

    sae.train()
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        recon, hidden = sae(x)
        loss_val = sae.loss(x, recon, hidden)
        loss_val.backward()
        optimizer.step()
        losses.append(loss_val.item())

    assert losses[-1] < losses[0], "Loss should decrease during training"


# ---------------------------------------------------------------------------
# get_live_features
# ---------------------------------------------------------------------------


def test_get_live_features_returns_tensor(config, sae):
    x = torch.randn(16, config.input_dim)
    live = sae.get_live_features(x)
    assert isinstance(live, torch.Tensor)


def test_get_live_features_indices_in_range(config, sae):
    x = torch.randn(16, config.input_dim)
    live = sae.get_live_features(x, threshold=0.0)
    assert (live >= 0).all()
    assert (live < config.hidden_dim).all()


def test_get_live_features_high_threshold_fewer_features(config):
    sae = SparseAutoencoder(config)
    x = torch.randn(32, config.input_dim)
    live_low = sae.get_live_features(x, threshold=0.0)
    live_high = sae.get_live_features(x, threshold=10.0)
    assert len(live_high) <= len(live_low)
