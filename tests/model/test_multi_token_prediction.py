"""Tests for Multi-Token Prediction (MTP) module.

Project test config: AureliusConfig(
    n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
    head_dim=16, d_ff=128, vocab_size=256, max_seq_len=64
)
"""

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.multi_token_prediction import (
    MTPConfig,
    MultiTokenPredictionHead,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def mtp_cfg():
    return MTPConfig(depth=4, lambda_mtp=0.3)


# ---------------------------------------------------------------------------
# MTPConfig
# ---------------------------------------------------------------------------


def test_mtp_config_defaults():
    config = MTPConfig()
    assert config.depth == 1
    assert config.lambda_mtp == 0.3


def test_mtp_config_custom_values():
    config = MTPConfig(depth=3, lambda_mtp=0.5)
    assert config.depth == 3
    assert config.lambda_mtp == 0.5


# ---------------------------------------------------------------------------
# MultiTokenPredictionHead — output shape
# ---------------------------------------------------------------------------


def test_mtp_head_output_shape(cfg, mtp_cfg):
    """MultiTokenPredictionHead: (B, T, D) -> list of depth logits."""
    B, T = 2, 16
    head = MultiTokenPredictionHead(cfg, mtp_cfg)
    hidden = torch.randn(B, T, cfg.d_model)
    loss, all_logits = head(hidden)
    assert loss is None
    assert len(all_logits) == mtp_cfg.depth
    for logits in all_logits:
        assert logits.shape == (B, T, cfg.vocab_size)


# ---------------------------------------------------------------------------
# MultiTokenPredictionHead — forward with labels
# ---------------------------------------------------------------------------


def test_mtp_head_forward_with_labels(cfg, mtp_cfg):
    """With labels: loss > 0."""
    B, T = 2, 16
    head = MultiTokenPredictionHead(cfg, mtp_cfg)
    hidden = torch.randn(B, T, cfg.d_model)
    labels = torch.randint(0, cfg.vocab_size, (B, T))
    loss, all_logits = head(hidden, labels=labels)
    assert loss is not None
    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"
    assert len(all_logits) == mtp_cfg.depth


# ---------------------------------------------------------------------------
# MultiTokenPredictionHead — compute_loss
# ---------------------------------------------------------------------------


def test_mtp_head_compute_loss(cfg, mtp_cfg):
    """compute_loss returns a scalar tensor."""
    B, T = 2, 16
    head = MultiTokenPredictionHead(cfg, mtp_cfg)
    hidden = torch.randn(B, T, cfg.d_model)
    labels = torch.randint(0, cfg.vocab_size, (B, T))
    loss = head.compute_loss(hidden, labels)
    assert loss.shape == torch.Size([])
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# MultiTokenPredictionHead — varying depth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_mtp_head_varying_depth(cfg, depth):
    """Different depths produce the correct number of logit tensors."""
    B, T = 2, 16
    config = MTPConfig(depth=depth, lambda_mtp=0.3)
    head = MultiTokenPredictionHead(cfg, config)
    hidden = torch.randn(B, T, cfg.d_model)
    labels = torch.randint(0, cfg.vocab_size, (B, T))
    loss, all_logits = head(hidden, labels=labels)
    assert len(all_logits) == depth
    assert loss is not None
    assert loss.item() > 0
