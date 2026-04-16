"""Tests for src/inference/watermarking.py

Uses tiny configurations (VOCAB=64) to keep execution fast.
"""

import math

import pytest
import torch

from src.inference.watermarking import (
    WatermarkConfig,
    WatermarkDetector,
    WatermarkLogitProcessor,
    apply_watermark_bias,
    compute_watermark_strength,
    detect_watermark_score,
    get_green_list,
)

# ---------------------------------------------------------------------------
# Shared tiny test constants
# ---------------------------------------------------------------------------
VOCAB = 64
DELTA = 2.0
GAMMA = 0.25
KEY = 42
T = 20  # sequence length for statistical tests


# ---------------------------------------------------------------------------
# 1. WatermarkConfig defaults
# ---------------------------------------------------------------------------
def test_config_defaults():
    cfg = WatermarkConfig()
    assert cfg.vocab_size == 50257
    assert cfg.delta == 2.0
    assert cfg.gamma == 0.25
    assert cfg.seeding_scheme == "hash"
    assert cfg.key == 42


# ---------------------------------------------------------------------------
# 2. get_green_list length ≈ gamma * vocab (within 10%)
# ---------------------------------------------------------------------------
def test_green_list_length_approx():
    green = get_green_list(prev_token=7, vocab_size=VOCAB, gamma=GAMMA, key=KEY)
    expected = VOCAB * GAMMA
    assert len(green) == pytest.approx(expected, abs=expected * 0.10 + 1)


# ---------------------------------------------------------------------------
# 3. get_green_list is deterministic: same inputs → same output
# ---------------------------------------------------------------------------
def test_green_list_deterministic():
    g1 = get_green_list(prev_token=5, vocab_size=VOCAB, gamma=GAMMA, key=KEY)
    g2 = get_green_list(prev_token=5, vocab_size=VOCAB, gamma=GAMMA, key=KEY)
    assert g1 == g2


# ---------------------------------------------------------------------------
# 4. Different prev tokens produce different green lists
# ---------------------------------------------------------------------------
def test_green_list_different_prev_tokens():
    g0 = get_green_list(prev_token=0, vocab_size=VOCAB, gamma=GAMMA, key=KEY)
    g1 = get_green_list(prev_token=1, vocab_size=VOCAB, gamma=GAMMA, key=KEY)
    assert g0 != g1


# ---------------------------------------------------------------------------
# 5. apply_watermark_bias: green tokens have strictly higher logits than before
# ---------------------------------------------------------------------------
def test_apply_watermark_bias_green_tokens_boosted():
    logits = torch.zeros(VOCAB)
    green = get_green_list(prev_token=3, vocab_size=VOCAB, gamma=GAMMA, key=KEY)
    modified = apply_watermark_bias(logits, green, DELTA)
    for idx in green:
        assert modified[idx].item() == pytest.approx(DELTA, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. apply_watermark_bias: output shape equals input shape
# ---------------------------------------------------------------------------
def test_apply_watermark_bias_shape_preserved():
    logits = torch.randn(VOCAB)
    green = get_green_list(prev_token=0, vocab_size=VOCAB, gamma=GAMMA, key=KEY)
    modified = apply_watermark_bias(logits, green, DELTA)
    assert modified.shape == logits.shape


# ---------------------------------------------------------------------------
# 7. detect_watermark_score returns a tuple of exactly 2 floats
# ---------------------------------------------------------------------------
def test_detect_watermark_score_return_type():
    tokens = list(range(T + 1))
    result = detect_watermark_score(tokens, VOCAB, GAMMA, KEY)
    assert isinstance(result, tuple)
    assert len(result) == 2
    z, frac = result
    assert isinstance(z, float)
    assert isinstance(frac, float)


# ---------------------------------------------------------------------------
# 8. WatermarkLogitProcessor output shape is (VOCAB,)
# ---------------------------------------------------------------------------
def test_logit_processor_output_shape():
    cfg = WatermarkConfig(vocab_size=VOCAB, delta=DELTA, gamma=GAMMA, key=KEY)
    processor = WatermarkLogitProcessor(cfg)
    logits = torch.zeros(VOCAB)
    out = processor(logits, prev_token=10)
    assert out.shape == (VOCAB,)


# ---------------------------------------------------------------------------
# 9. WatermarkLogitProcessor: green tokens are boosted relative to red tokens
# ---------------------------------------------------------------------------
def test_logit_processor_green_tokens_boosted():
    cfg = WatermarkConfig(vocab_size=VOCAB, delta=DELTA, gamma=GAMMA, key=KEY)
    processor = WatermarkLogitProcessor(cfg)
    logits = torch.zeros(VOCAB)
    prev = 10
    out = processor(logits, prev_token=prev)
    green = set(get_green_list(prev_token=prev, vocab_size=VOCAB, gamma=GAMMA, key=KEY))
    red = set(range(VOCAB)) - green
    if red:
        green_mean = out[list(green)].mean().item()
        red_mean = out[list(red)].mean().item()
        assert green_mean > red_mean


# ---------------------------------------------------------------------------
# 10. WatermarkDetector.detect returns dict with required keys
# ---------------------------------------------------------------------------
def test_detector_detect_required_keys():
    cfg = WatermarkConfig(vocab_size=VOCAB, delta=DELTA, gamma=GAMMA, key=KEY)
    detector = WatermarkDetector(cfg, z_threshold=4.0)
    tokens = list(range(T + 1))
    result = detector.detect(tokens)
    assert "z_score" in result
    assert "green_fraction" in result
    assert "is_watermarked" in result


# ---------------------------------------------------------------------------
# 11. WatermarkDetector.detect: is_watermarked is bool-like (0/1 or True/False)
# ---------------------------------------------------------------------------
def test_detector_detect_is_watermarked_bool():
    cfg = WatermarkConfig(vocab_size=VOCAB, delta=DELTA, gamma=GAMMA, key=KEY)
    detector = WatermarkDetector(cfg, z_threshold=4.0)
    tokens = list(range(T + 1))
    result = detector.detect(tokens)
    # is_watermarked should be truthy/falsy (bool, int, or float 0/1)
    assert result["is_watermarked"] in (True, False, 0, 0.0, 1, 1.0)


# ---------------------------------------------------------------------------
# 12. batch_detect returns list of same length as input
# ---------------------------------------------------------------------------
def test_batch_detect_length():
    cfg = WatermarkConfig(vocab_size=VOCAB, delta=DELTA, gamma=GAMMA, key=KEY)
    detector = WatermarkDetector(cfg, z_threshold=4.0)
    seqs = [list(range(T + 1)), list(range(T, 0, -1)), [0] * (T + 1)]
    results = detector.batch_detect(seqs)
    assert isinstance(results, list)
    assert len(results) == len(seqs)


# ---------------------------------------------------------------------------
# 13. All-green sequence has z_score > 0
# ---------------------------------------------------------------------------
def test_all_green_sequence_positive_z():
    """Build a sequence where every token lands in its predecessor's green list."""
    cfg = WatermarkConfig(vocab_size=VOCAB, delta=DELTA, gamma=GAMMA, key=KEY)
    tokens = [0]
    for _ in range(T):
        green = get_green_list(tokens[-1], VOCAB, GAMMA, KEY)
        tokens.append(green[0])  # always pick first green token

    z, frac = detect_watermark_score(tokens, VOCAB, GAMMA, KEY)
    assert z > 0.0
    assert frac > 0.0


# ---------------------------------------------------------------------------
# 14. compute_watermark_strength > 0 when delta > 0
# ---------------------------------------------------------------------------
def test_compute_watermark_strength_positive():
    logits = torch.zeros(VOCAB)
    green = get_green_list(prev_token=0, vocab_size=VOCAB, gamma=GAMMA, key=KEY)
    watermarked = apply_watermark_bias(logits, green, DELTA)
    strength = compute_watermark_strength(logits, watermarked)
    assert isinstance(strength, float)
    assert strength > 0.0
