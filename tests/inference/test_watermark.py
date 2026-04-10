"""Tests for Kirchenbauer et al. style LLM watermarking."""

import math

import pytest
import torch
import torch.nn as nn

from src.inference.watermark import (
    WatermarkConfig,
    WatermarkDetector,
    WatermarkGenerator,
    apply_watermark,
    compute_green_list,
    detect_watermark,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return WatermarkConfig()


@pytest.fixture
def config_large():
    """Larger vocab for statistical tests."""
    return WatermarkConfig(gamma=0.25, delta=2.0, vocab_size=1000, seed=42)


class DummyModel(nn.Module):
    """Minimal model that returns (loss=None, logits, pkv=[])."""

    def __init__(self, vocab_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        B, S = input_ids.shape
        logits = torch.randn(B, S, self.vocab_size)
        return None, logits, []


# ---------------------------------------------------------------------------
# WatermarkConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = WatermarkConfig()
    assert cfg.gamma == 0.25
    assert cfg.delta == 2.0
    assert cfg.vocab_size == 256
    assert cfg.seed == 42


def test_config_custom_values():
    cfg = WatermarkConfig(gamma=0.5, delta=3.0, vocab_size=512, seed=99)
    assert cfg.gamma == 0.5
    assert cfg.delta == 3.0
    assert cfg.vocab_size == 512
    assert cfg.seed == 99


# ---------------------------------------------------------------------------
# compute_green_list
# ---------------------------------------------------------------------------

def test_green_list_size_default(config):
    gl = compute_green_list(0, config.vocab_size, config.gamma, config.seed)
    expected = int(config.vocab_size * config.gamma)
    assert len(gl) == expected


def test_green_list_size_large(config_large):
    gl = compute_green_list(10, config_large.vocab_size, config_large.gamma, config_large.seed)
    expected = int(config_large.vocab_size * config_large.gamma)
    assert len(gl) == expected


def test_green_list_deterministic(config):
    g1 = compute_green_list(7, config.vocab_size, config.gamma, config.seed)
    g2 = compute_green_list(7, config.vocab_size, config.gamma, config.seed)
    assert g1 == g2


def test_green_list_varies_by_prev_token(config):
    g1 = compute_green_list(0, config.vocab_size, config.gamma, config.seed)
    g2 = compute_green_list(1, config.vocab_size, config.gamma, config.seed)
    assert g1 != g2


def test_green_list_varies_by_seed(config):
    g1 = compute_green_list(0, config.vocab_size, config.gamma, seed=1)
    g2 = compute_green_list(0, config.vocab_size, config.gamma, seed=2)
    assert g1 != g2


def test_green_list_indices_in_range(config):
    gl = compute_green_list(42, config.vocab_size, config.gamma, config.seed)
    assert all(0 <= idx < config.vocab_size for idx in gl)


# ---------------------------------------------------------------------------
# apply_watermark
# ---------------------------------------------------------------------------

def test_apply_watermark_increases_green_logits(config):
    logits = torch.zeros(config.vocab_size)
    result = apply_watermark(logits, 10, config)
    green = compute_green_list(10, config.vocab_size, config.gamma, config.seed)
    for idx in green:
        assert result[idx].item() == pytest.approx(config.delta)
    for idx in range(config.vocab_size):
        if idx not in green:
            assert result[idx].item() == pytest.approx(0.0)


def test_apply_watermark_preserves_shape(config):
    logits = torch.randn(config.vocab_size)
    result = apply_watermark(logits, 5, config)
    assert result.shape == logits.shape


def test_apply_watermark_preserves_shape_batched(config):
    logits = torch.randn(4, config.vocab_size)
    result = apply_watermark(logits, 5, config)
    assert result.shape == logits.shape


def test_apply_watermark_does_not_modify_input(config):
    logits = torch.randn(config.vocab_size)
    original = logits.clone()
    _ = apply_watermark(logits, 5, config)
    assert torch.equal(logits, original)


# ---------------------------------------------------------------------------
# detect_watermark
# ---------------------------------------------------------------------------

def test_detect_watermark_high_green_fraction(config_large):
    """All-green tokens should give green_fraction close to 1.0."""
    tokens = [100]
    for _ in range(100):
        green = compute_green_list(
            tokens[-1], config_large.vocab_size, config_large.gamma, config_large.seed
        )
        tokens.append(next(iter(green)))

    result = detect_watermark(tokens, config_large)
    assert result["green_fraction"] > 0.9
    assert result["is_watermarked"] is True
    assert result["z_score"] > 2.0


def test_detect_watermark_random_text(config_large):
    """Random tokens should have green_fraction near gamma."""
    torch.manual_seed(0)
    tokens = torch.randint(0, config_large.vocab_size, (500,)).tolist()
    result = detect_watermark(tokens, config_large)
    # Should be within a reasonable range of gamma
    assert abs(result["green_fraction"] - config_large.gamma) < 0.1


def test_detect_watermark_z_score_higher_for_watermarked(config_large):
    """Watermarked text should have much higher z-score than random."""
    # Watermarked
    tokens_wm = [50]
    for _ in range(100):
        green = compute_green_list(
            tokens_wm[-1], config_large.vocab_size, config_large.gamma, config_large.seed
        )
        tokens_wm.append(next(iter(green)))

    # Random
    torch.manual_seed(123)
    tokens_rand = torch.randint(0, config_large.vocab_size, (101,)).tolist()

    z_wm = detect_watermark(tokens_wm, config_large)["z_score"]
    z_rand = detect_watermark(tokens_rand, config_large)["z_score"]
    assert z_wm > z_rand


def test_detect_watermark_empty_sequence(config):
    result = detect_watermark([], config)
    assert result["green_fraction"] == 0.0
    assert result["z_score"] == 0.0
    assert result["is_watermarked"] is False


def test_detect_watermark_single_token(config):
    result = detect_watermark([42], config)
    assert result["green_fraction"] == 0.0
    assert result["is_watermarked"] is False


def test_detect_watermark_accepts_tensor(config_large):
    tokens = [100]
    for _ in range(20):
        green = compute_green_list(
            tokens[-1], config_large.vocab_size, config_large.gamma, config_large.seed
        )
        tokens.append(next(iter(green)))
    tensor_ids = torch.tensor(tokens)
    result = detect_watermark(tensor_ids, config_large)
    assert "z_score" in result


def test_detect_watermark_returns_required_keys(config):
    result = detect_watermark([1, 2, 3, 4, 5], config)
    assert "green_fraction" in result
    assert "z_score" in result
    assert "is_watermarked" in result


# ---------------------------------------------------------------------------
# WatermarkGenerator
# ---------------------------------------------------------------------------

def test_generator_returns_tensor():
    cfg = WatermarkConfig(vocab_size=256)
    model = DummyModel(vocab_size=256)
    gen = WatermarkGenerator(model, cfg)
    input_ids = torch.tensor([1, 2, 3])
    out = gen.generate(input_ids, max_new_tokens=5)
    assert isinstance(out, torch.Tensor)


def test_generator_output_length():
    cfg = WatermarkConfig(vocab_size=256)
    model = DummyModel(vocab_size=256)
    gen = WatermarkGenerator(model, cfg)
    input_ids = torch.tensor([1, 2, 3])
    out = gen.generate(input_ids, max_new_tokens=10)
    assert out.shape[0] == 3 + 10


def test_generator_output_is_watermarked():
    """Generated tokens should be detected as watermarked."""
    cfg = WatermarkConfig(gamma=0.25, delta=5.0, vocab_size=256, seed=42)
    model = DummyModel(vocab_size=256)
    gen = WatermarkGenerator(model, cfg)
    input_ids = torch.tensor([1, 2, 3])
    out = gen.generate(input_ids, max_new_tokens=50)
    result = detect_watermark(out.tolist(), cfg)
    assert result["green_fraction"] > 0.5


def test_generator_accepts_2d_input():
    cfg = WatermarkConfig(vocab_size=256)
    model = DummyModel(vocab_size=256)
    gen = WatermarkGenerator(model, cfg)
    input_ids = torch.tensor([[1, 2, 3]])
    out = gen.generate(input_ids, max_new_tokens=5)
    assert isinstance(out, torch.Tensor)
    assert out.dim() == 1


# ---------------------------------------------------------------------------
# WatermarkDetector class
# ---------------------------------------------------------------------------

def test_detector_returns_dict(config):
    det = WatermarkDetector(config)
    result = det.detect([1, 2, 3, 4, 5])
    assert isinstance(result, dict)
    assert "green_fraction" in result
    assert "z_score" in result
    assert "is_watermarked" in result


def test_detector_detect_matches_function(config_large):
    """WatermarkDetector.detect should match detect_watermark."""
    tokens = [10, 20, 30, 40, 50]
    det = WatermarkDetector(config_large)
    r1 = det.detect(tokens)
    r2 = detect_watermark(tokens, config_large)
    assert r1["z_score"] == pytest.approx(r2["z_score"])
    assert r1["green_fraction"] == pytest.approx(r2["green_fraction"])
    assert r1["is_watermarked"] == r2["is_watermarked"]
