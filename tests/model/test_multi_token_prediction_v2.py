"""Tests for multi_token_prediction_v2.py

Covers:
  - MTPConfig field defaults and overrides
  - MTPHead output shape and gradient flow
  - MTPTrunk output shape and gradient flow
  - MultiTokenPredictor list length, tensor shapes, share_trunk=False
  - MTPLoss scalar output, dict keys, finite values, gradient flow
  - MTPDecoder.draft_tokens shape, vocab range, n_heads=1
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from aurelius.model.multi_token_prediction_v2 import (
    MTPConfig,
    MTPDecoder,
    MTPHead,
    MTPLoss,
    MTPTrunk,
    MultiTokenPredictor,
)

# ---------------------------------------------------------------------------
# Shared test dimensions
# ---------------------------------------------------------------------------

B = 2
T = 8
D = 32
V = 64
K = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> MTPConfig:
    return MTPConfig(d_model=D, vocab_size=V, n_heads=K)


@pytest.fixture
def cfg_no_trunk() -> MTPConfig:
    return MTPConfig(d_model=D, vocab_size=V, n_heads=K, share_trunk=False)


@pytest.fixture
def hidden() -> torch.Tensor:
    return torch.randn(B, T, D)


@pytest.fixture
def labels() -> torch.Tensor:
    return torch.randint(0, V, (B, T))


# ---------------------------------------------------------------------------
# MTPConfig tests
# ---------------------------------------------------------------------------


class TestMTPConfig:
    def test_required_fields(self):
        c = MTPConfig(d_model=128, vocab_size=512)
        assert c.d_model == 128
        assert c.vocab_size == 512

    def test_default_n_heads(self):
        c = MTPConfig(d_model=64, vocab_size=256)
        assert c.n_heads == 4

    def test_default_share_trunk(self):
        c = MTPConfig(d_model=64, vocab_size=256)
        assert c.share_trunk is True

    def test_default_trunk_expansion(self):
        c = MTPConfig(d_model=64, vocab_size=256)
        assert c.trunk_expansion == 2

    def test_override_fields(self, cfg):
        assert cfg.d_model == D
        assert cfg.vocab_size == V
        assert cfg.n_heads == K


# ---------------------------------------------------------------------------
# MTPHead tests
# ---------------------------------------------------------------------------


class TestMTPHead:
    def test_output_shape(self, hidden):
        head = MTPHead(D, V)
        out = head(hidden)
        assert out.shape == (B, T, V), f"Expected {(B, T, V)}, got {out.shape}"

    def test_gradient_flows(self, hidden):
        head = MTPHead(D, V)
        x = hidden.requires_grad_(True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# MTPTrunk tests
# ---------------------------------------------------------------------------


class TestMTPTrunk:
    def test_output_shape_unchanged(self, hidden):
        trunk = MTPTrunk(D)
        out = trunk(hidden)
        assert out.shape == hidden.shape, (
            f"Trunk should preserve shape {hidden.shape}, got {out.shape}"
        )

    def test_expansion_parameter(self, hidden):
        trunk = MTPTrunk(D, expansion=4)
        out = trunk(hidden)
        assert out.shape == hidden.shape

    def test_gradient_flows(self, hidden):
        trunk = MTPTrunk(D)
        x = hidden.requires_grad_(True)
        out = trunk(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# MultiTokenPredictor tests
# ---------------------------------------------------------------------------


class TestMultiTokenPredictor:
    def test_returns_list_of_n_heads(self, cfg, hidden):
        predictor = MultiTokenPredictor(cfg)
        outputs = predictor(hidden)
        assert isinstance(outputs, list)
        assert len(outputs) == K

    def test_all_output_shapes(self, cfg, hidden):
        predictor = MultiTokenPredictor(cfg)
        outputs = predictor(hidden)
        for i, out in enumerate(outputs):
            assert out.shape == (B, T, V), (
                f"Head {i}: expected {(B, T, V)}, got {out.shape}"
            )

    def test_trunk_is_present_when_share_trunk_true(self, cfg):
        predictor = MultiTokenPredictor(cfg)
        assert predictor.trunk is not None

    def test_trunk_is_none_when_share_trunk_false(self, cfg_no_trunk):
        predictor = MultiTokenPredictor(cfg_no_trunk)
        assert predictor.trunk is None

    def test_no_trunk_output_shape(self, cfg_no_trunk, hidden):
        predictor = MultiTokenPredictor(cfg_no_trunk)
        outputs = predictor(hidden)
        assert len(outputs) == K
        for out in outputs:
            assert out.shape == (B, T, V)

    def test_heads_are_module_list(self, cfg):
        predictor = MultiTokenPredictor(cfg)
        assert isinstance(predictor.heads, nn.ModuleList)
        assert len(predictor.heads) == K


# ---------------------------------------------------------------------------
# MTPLoss tests
# ---------------------------------------------------------------------------


class TestMTPLoss:
    def test_returns_scalar(self, cfg, hidden, labels):
        predictor = MultiTokenPredictor(cfg)
        criterion = MTPLoss(n_heads=K)
        logits = predictor(hidden)
        loss, _ = criterion(logits, labels)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"

    def test_returns_correct_keys(self, cfg, hidden, labels):
        predictor = MultiTokenPredictor(cfg)
        criterion = MTPLoss(n_heads=K)
        logits = predictor(hidden)
        _, metrics = criterion(logits, labels)
        for k in range(K):
            assert f"head_{k}_loss" in metrics, f"Missing key head_{k}_loss"
        assert "total_loss" in metrics

    def test_loss_values_are_finite(self, cfg, hidden, labels):
        predictor = MultiTokenPredictor(cfg)
        criterion = MTPLoss(n_heads=K)
        logits = predictor(hidden)
        loss, metrics = criterion(logits, labels)
        assert torch.isfinite(loss), "total_loss is not finite"
        for key, val in metrics.items():
            assert torch.isfinite(val), f"{key} is not finite"

    def test_gradient_flows_through_loss(self, cfg, hidden, labels):
        predictor = MultiTokenPredictor(cfg)
        criterion = MTPLoss(n_heads=K)
        logits = predictor(hidden)
        loss, _ = criterion(logits, labels)
        loss.backward()
        # Check at least one parameter has a gradient
        has_grad = any(p.grad is not None for p in predictor.parameters())
        assert has_grad, "No gradients flowed through MTPLoss"

    def test_ignore_index_respected(self, cfg, hidden):
        """Labels with -100 at some positions should still produce finite loss."""
        labels_with_ignore = torch.full((B, T), -100, dtype=torch.long)
        # Only set a few valid positions
        labels_with_ignore[:, :4] = torch.randint(0, V, (B, 4))
        predictor = MultiTokenPredictor(cfg)
        criterion = MTPLoss(n_heads=K)
        logits = predictor(hidden)
        loss, metrics = criterion(logits, labels_with_ignore)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# MTPDecoder tests
# ---------------------------------------------------------------------------


class TestMTPDecoder:
    def test_draft_tokens_shape(self, cfg):
        predictor = MultiTokenPredictor(cfg)
        decoder = MTPDecoder(predictor)
        last_hidden = torch.randn(B, 1, D)
        drafts = decoder.draft_tokens(last_hidden)
        assert drafts.shape == (B, K), (
            f"Expected ({B}, {K}), got {drafts.shape}"
        )

    def test_draft_tokens_in_vocab_range(self, cfg):
        predictor = MultiTokenPredictor(cfg)
        decoder = MTPDecoder(predictor)
        last_hidden = torch.randn(B, 1, D)
        drafts = decoder.draft_tokens(last_hidden)
        assert (drafts >= 0).all(), "Some draft token ids are negative"
        assert (drafts < V).all(), "Some draft token ids exceed vocab_size"

    def test_draft_tokens_dtype_is_long(self, cfg):
        predictor = MultiTokenPredictor(cfg)
        decoder = MTPDecoder(predictor)
        last_hidden = torch.randn(B, 1, D)
        drafts = decoder.draft_tokens(last_hidden)
        assert drafts.dtype == torch.long, f"Expected torch.long, got {drafts.dtype}"

    def test_n_heads_1_works(self):
        """Single-head MTP (n_heads=1) should work end-to-end."""
        cfg1 = MTPConfig(d_model=D, vocab_size=V, n_heads=1)
        predictor = MultiTokenPredictor(cfg1)
        decoder = MTPDecoder(predictor)
        last_hidden = torch.randn(B, 1, D)
        drafts = decoder.draft_tokens(last_hidden)
        assert drafts.shape == (B, 1)
        assert (drafts >= 0).all() and (drafts < V).all()
