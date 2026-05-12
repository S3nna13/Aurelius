"""Tests for Multi-Token Prediction heads."""
from __future__ import annotations

import torch

from src.model.config import AureliusConfig
from src.model.multi_token_prediction import MTPConfig, MultiTokenPredictionHead


def test_mtp_forward_returns_logits_list():
    cfg = AureliusConfig()
    mtp = MultiTokenPredictionHead(cfg, MTPConfig(depth=2))
    x = torch.randn(2, 16, cfg.d_model)
    loss, logits = mtp(x, labels=None)
    assert loss is None
    loss, logits = mtp(x, labels=torch.randint(0, cfg.vocab_size, (2, 16)))
    assert isinstance(logits, list)
    assert len(logits) == 2


def test_mtp_compute_loss():
    cfg = AureliusConfig()
    mtp = MultiTokenPredictionHead(cfg, MTPConfig(depth=1))
    x = torch.randn(2, 16, cfg.d_model)
    labels = torch.randint(0, cfg.vocab_size, (2, 16))
    loss = mtp.compute_loss(x, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_mtp_head_output_shape():
    cfg = AureliusConfig()
    mtp = MultiTokenPredictionHead(cfg, MTPConfig(depth=2))
    x = torch.randn(2, 8, cfg.d_model)
    loss, logits = mtp(x, labels=torch.randint(0, cfg.vocab_size, (2, 8)))
    for logit in logits:
        assert logit.shape[0] == 2  # batch size
        assert logit.shape[2] == cfg.vocab_size  # vocab