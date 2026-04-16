"""Tests for neural_watermark.py (Kirchenbauer et al. 2023 green/red list watermarking)."""

import math

import pytest
import torch
import torch.nn as nn

from src.inference.neural_watermark import (
    WatermarkConfig,
    GreenListWatermark,
    watermark_sample,
    generate_watermarked,
    detect_watermark,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Tiny model fixture
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256

def make_tiny_config():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=64,
    )


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    cfg = make_tiny_config()
    model = AureliusTransformer(cfg)
    model.eval()
    return model


@pytest.fixture
def wm_config():
    """Watermark config with gamma=0.5 and strong delta for statistical tests."""
    return WatermarkConfig(
        key=42,
        gamma=0.5,
        delta=10.0,
        vocab_size=VOCAB_SIZE,
        seeding_scheme="simple",
        detection_threshold=4.0,
    )


@pytest.fixture
def watermark(wm_config):
    return GreenListWatermark(wm_config)


# ---------------------------------------------------------------------------
# 1. WatermarkConfig defaults are sensible
# ---------------------------------------------------------------------------

def test_watermark_config_defaults():
    cfg = WatermarkConfig()
    assert cfg.key == 42
    assert 0.0 < cfg.gamma < 1.0, "gamma must be in (0, 1)"
    assert cfg.delta > 0.0, "delta should be positive"
    assert cfg.vocab_size > 0
    assert isinstance(cfg.seeding_scheme, str)
    assert cfg.detection_threshold > 0.0


# ---------------------------------------------------------------------------
# 2. get_green_list returns (vocab_size,) bool tensor
# ---------------------------------------------------------------------------

def test_get_green_list_shape(watermark, wm_config):
    mask = watermark.get_green_list(prev_token=5)
    assert mask.shape == (wm_config.vocab_size,)
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 3. Fraction of green tokens approximately equals gamma (within +-0.1)
# ---------------------------------------------------------------------------

def test_get_green_list_fraction(watermark, wm_config):
    mask = watermark.get_green_list(prev_token=42)
    green_frac = mask.float().mean().item()
    assert abs(green_frac - wm_config.gamma) <= 0.1, (
        f"Expected ~{wm_config.gamma} green fraction, got {green_frac}"
    )


# ---------------------------------------------------------------------------
# 4. Same prev_token always produces the same green list (deterministic)
# ---------------------------------------------------------------------------

def test_get_green_list_deterministic(watermark):
    mask_a = watermark.get_green_list(prev_token=7)
    mask_b = watermark.get_green_list(prev_token=7)
    assert torch.equal(mask_a, mask_b), "Green list must be deterministic for same prev_token"


# ---------------------------------------------------------------------------
# 5. Different prev_tokens produce different green lists
# ---------------------------------------------------------------------------

def test_get_green_list_different_prev_tokens(watermark):
    mask_a = watermark.get_green_list(prev_token=1)
    mask_b = watermark.get_green_list(prev_token=2)
    assert not torch.equal(mask_a, mask_b), (
        "Different prev_tokens should (almost certainly) produce different green lists"
    )


# ---------------------------------------------------------------------------
# 6. apply() increases logits for green tokens
# ---------------------------------------------------------------------------

def test_apply_increases_green_logits(watermark, wm_config):
    torch.manual_seed(0)
    logits = torch.randn(wm_config.vocab_size)
    prev_token = 10

    green_mask = watermark.get_green_list(prev_token)
    modified = watermark.apply(logits, prev_token)

    diff = modified - logits
    # Green tokens should have increased by delta
    assert (diff[green_mask] == wm_config.delta).all(), (
        "Green-list logits should increase by exactly delta"
    )


# ---------------------------------------------------------------------------
# 7. apply() doesn't change red-token logits
# ---------------------------------------------------------------------------

def test_apply_does_not_change_red_logits(watermark, wm_config):
    torch.manual_seed(1)
    logits = torch.randn(wm_config.vocab_size)
    prev_token = 10

    green_mask = watermark.get_green_list(prev_token)
    red_mask = ~green_mask
    modified = watermark.apply(logits, prev_token)

    assert torch.equal(modified[red_mask], logits[red_mask]), (
        "Red-list logits must not be changed by apply()"
    )


# ---------------------------------------------------------------------------
# 8. score_token returns 0 or 1
# ---------------------------------------------------------------------------

def test_score_token_binary(watermark):
    for token in range(10):
        score = watermark.score_token(token=token, prev_token=0)
        assert score in (0, 1), f"score_token must return 0 or 1, got {score}"


# ---------------------------------------------------------------------------
# 9. score_sequence returns dict with required keys
# ---------------------------------------------------------------------------

def test_score_sequence_keys(watermark):
    tokens = torch.arange(20)
    result = watermark.score_sequence(tokens)
    assert "green_fraction" in result
    assert "z_score" in result
    assert "is_watermarked" in result


# ---------------------------------------------------------------------------
# 10. score_sequence green_fraction is in [0, 1]
# ---------------------------------------------------------------------------

def test_score_sequence_green_fraction_range(watermark):
    torch.manual_seed(2)
    tokens = torch.randint(0, VOCAB_SIZE, (50,))
    result = watermark.score_sequence(tokens)
    assert 0.0 <= result["green_fraction"] <= 1.0, (
        f"green_fraction must be in [0, 1], got {result['green_fraction']}"
    )


# ---------------------------------------------------------------------------
# 11. watermark_sample returns valid token id in [0, vocab_size)
# ---------------------------------------------------------------------------

def test_watermark_sample_valid_token(watermark, wm_config):
    torch.manual_seed(3)
    logits = torch.randn(wm_config.vocab_size)
    token = watermark_sample(logits, watermark, prev_token=0)
    assert isinstance(token, int)
    assert 0 <= token < wm_config.vocab_size, (
        f"sampled token {token} out of range [0, {wm_config.vocab_size})"
    )


# ---------------------------------------------------------------------------
# 12. generate_watermarked returns tensor of length max_new_tokens
# ---------------------------------------------------------------------------

def test_generate_watermarked_length(tiny_model, watermark):
    torch.manual_seed(4)
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    max_new = 8
    generated = generate_watermarked(
        tiny_model, prompt, watermark, max_new_tokens=max_new, temperature=1.0
    )
    assert generated.shape == (max_new,), (
        f"Expected ({max_new},) tensor, got shape {generated.shape}"
    )


# ---------------------------------------------------------------------------
# 13. Long watermarked sequence has z_score > 0 (biased toward green)
# ---------------------------------------------------------------------------

def test_watermarked_sequence_positive_z_score(tiny_model):
    """With strong delta=10.0 and gamma=0.5, watermarked text should be clearly green-biased."""
    torch.manual_seed(5)
    # Use a strong config so the effect is unmistakable
    cfg = WatermarkConfig(
        key=42,
        gamma=0.5,
        delta=10.0,
        vocab_size=VOCAB_SIZE,
        detection_threshold=4.0,
    )
    wm = GreenListWatermark(cfg)

    prompt = torch.tensor([1, 5, 10], dtype=torch.long)
    generated = generate_watermarked(
        tiny_model, prompt, wm, max_new_tokens=50, temperature=1.0
    )

    # Combine prompt + generated for a longer sequence to score
    full_seq = torch.cat([prompt, generated])
    result = detect_watermark(full_seq, wm)
    assert result["z_score"] > 0, (
        f"Watermarked sequence should have positive z_score, got {result['z_score']}"
    )
