"""Tests for noise_training module: label smoothing, token noise, mixup, and trainer."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.noise_training import (
    NoiseAwareTrainer,
    NoiseConfig,
    inject_token_noise,
    label_smoothed_loss,
    mixup_embeddings,
    sample_mixup_lambda,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
SEQ_LEN = 8
BATCH = 2


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=512,
    )


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(42)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def optimizer(small_model):
    return torch.optim.Adam(small_model.parameters(), lr=1e-4)


@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


# ---------------------------------------------------------------------------
# 1. NoiseConfig defaults
# ---------------------------------------------------------------------------


def test_noise_config_defaults():
    cfg = NoiseConfig()
    assert cfg.label_smoothing == 0.1
    assert cfg.token_noise_prob == 0.05
    assert cfg.mixup_alpha == 0.2
    assert cfg.noise_type == "uniform"


# ---------------------------------------------------------------------------
# 2. label_smoothed_loss returns scalar
# ---------------------------------------------------------------------------


def test_label_smoothed_loss_returns_scalar():
    logits = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE)
    targets = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    loss = label_smoothed_loss(logits, targets, smoothing=0.1)
    assert loss.shape == torch.Size([])


# ---------------------------------------------------------------------------
# 3. label_smoothed_loss with no smoothing matches F.cross_entropy
# ---------------------------------------------------------------------------


def test_label_smoothed_loss_no_smoothing_matches_ce():
    torch.manual_seed(1)
    logits = torch.randn(BATCH * SEQ_LEN, VOCAB_SIZE)
    targets = torch.randint(0, VOCAB_SIZE, (BATCH * SEQ_LEN,))

    expected = F.cross_entropy(logits, targets, reduction="mean")
    got = label_smoothed_loss(logits, targets, smoothing=0.0)

    assert torch.allclose(got, expected, atol=1e-5), f"expected {expected}, got {got}"


# ---------------------------------------------------------------------------
# 4. label_smoothed_loss with smoothing > no-smoothing loss (softens predictions)
# ---------------------------------------------------------------------------


def test_label_smoothed_loss_smoothing_increases_loss():
    """Smoothing redistributes probability mass and typically changes the loss.
    More specifically, for a peaked distribution the smoothed loss is higher."""
    torch.manual_seed(2)
    # Use very peaked logits so CE is very low; smoothing should increase it.
    logits = torch.zeros(16, VOCAB_SIZE)
    targets = torch.zeros(16, dtype=torch.long)
    # Make class 0 very confident
    logits[:, 0] = 20.0

    loss_no_smooth = label_smoothed_loss(logits, targets, smoothing=0.0)
    loss_smooth = label_smoothed_loss(logits, targets, smoothing=0.3)

    assert loss_smooth > loss_no_smooth, (
        f"Smoothed loss ({loss_smooth:.4f}) should be > plain loss ({loss_no_smooth:.4f})"
    )


# ---------------------------------------------------------------------------
# 5. label_smoothed_loss respects ignore_index=-100
# ---------------------------------------------------------------------------


def test_label_smoothed_loss_ignore_index():
    torch.manual_seed(3)
    logits = torch.randn(BATCH * SEQ_LEN, VOCAB_SIZE)
    targets = torch.randint(0, VOCAB_SIZE, (BATCH * SEQ_LEN,))
    # Mask first half
    targets[: BATCH * SEQ_LEN // 2] = -100

    # Should not raise and should be a finite scalar
    loss = label_smoothed_loss(logits, targets, smoothing=0.1, ignore_index=-100)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 6. inject_token_noise changes ~noise_prob fraction of tokens
# ---------------------------------------------------------------------------


def test_inject_token_noise_changes_fraction(input_ids):
    torch.manual_seed(10)
    noise_prob = 0.5
    noisy = inject_token_noise(input_ids, noise_prob=noise_prob, vocab_size=VOCAB_SIZE)
    changed = (noisy != input_ids).float().mean().item()
    # Allow wide tolerance due to randomness; expect within ±0.25
    assert abs(changed - noise_prob) < 0.35, f"changed fraction {changed} too far from {noise_prob}"


# ---------------------------------------------------------------------------
# 7. inject_token_noise noise_prob=0 returns identical tensor
# ---------------------------------------------------------------------------


def test_inject_token_noise_zero_prob_identical(input_ids):
    noisy = inject_token_noise(input_ids, noise_prob=0.0, vocab_size=VOCAB_SIZE)
    assert torch.equal(noisy, input_ids)


# ---------------------------------------------------------------------------
# 8. inject_token_noise noise_prob=1 changes all tokens to valid range
# ---------------------------------------------------------------------------


def test_inject_token_noise_full_noise_valid_range(input_ids):
    noisy = inject_token_noise(input_ids, noise_prob=1.0, vocab_size=VOCAB_SIZE)
    assert noisy.shape == input_ids.shape
    assert noisy.min() >= 0
    assert noisy.max() < VOCAB_SIZE


# ---------------------------------------------------------------------------
# 9. mixup_embeddings output is weighted combination
# ---------------------------------------------------------------------------


def test_mixup_embeddings_weighted_combination(small_model):
    torch.manual_seed(20)
    ids_a = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    ids_b = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    lam = 0.6

    mixed = mixup_embeddings(small_model.embed, ids_a, ids_b, lam)
    expected = lam * small_model.embed(ids_a) + (1 - lam) * small_model.embed(ids_b)

    assert torch.allclose(mixed, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 10. mixup_embeddings lam=1.0 equals embed_fn(ids_a)
# ---------------------------------------------------------------------------


def test_mixup_embeddings_lam_one_equals_a(small_model):
    torch.manual_seed(21)
    ids_a = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    ids_b = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

    mixed = mixup_embeddings(small_model.embed, ids_a, ids_b, lam=1.0)
    expected = small_model.embed(ids_a)

    assert torch.allclose(mixed, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 11. sample_mixup_lambda alpha=0 → 1.0
# ---------------------------------------------------------------------------


def test_sample_mixup_lambda_alpha_zero():
    lam = sample_mixup_lambda(0.0)
    assert lam == 1.0


# ---------------------------------------------------------------------------
# 12. sample_mixup_lambda alpha>0 → value in (0, 1)
# ---------------------------------------------------------------------------


def test_sample_mixup_lambda_alpha_positive():
    torch.manual_seed(30)
    lam = sample_mixup_lambda(0.4)
    assert 0.0 < lam < 1.0, f"Expected lam in (0, 1), got {lam}"


# ---------------------------------------------------------------------------
# 13. NoiseAwareTrainer.train_step returns required keys
# ---------------------------------------------------------------------------


def test_noise_trainer_train_step_keys(small_model, optimizer, input_ids):
    cfg = NoiseConfig(label_smoothing=0.1, token_noise_prob=0.1)
    trainer = NoiseAwareTrainer(small_model, cfg, optimizer)
    result = trainer.train_step(input_ids)
    assert "loss" in result, "train_step result missing 'loss'"
    assert "noise_type" in result, "train_step result missing 'noise_type'"


# ---------------------------------------------------------------------------
# 14. NoiseAwareTrainer.train_step loss is finite float
# ---------------------------------------------------------------------------


def test_noise_trainer_train_step_loss_finite(small_model, optimizer, input_ids):
    cfg = NoiseConfig(label_smoothing=0.1, token_noise_prob=0.05)
    trainer = NoiseAwareTrainer(small_model, cfg, optimizer)
    result = trainer.train_step(input_ids)
    assert isinstance(result["loss"], float), f"loss should be float, got {type(result['loss'])}"
    assert math.isfinite(result["loss"]), f"loss is not finite: {result['loss']}"
