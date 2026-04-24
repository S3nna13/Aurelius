"""Tests for src/safety/watermark_detector.py"""
from __future__ import annotations

import math
import random

import pytest

from src.safety.watermark_detector import (
    WatermarkConfig,
    WatermarkDetector,
    WatermarkScheme,
    SAFETY_REGISTRY,
)


@pytest.fixture
def detector() -> WatermarkDetector:
    return WatermarkDetector()


def _make_config(scheme: WatermarkScheme = WatermarkScheme.GREEN_LIST) -> WatermarkConfig:
    return WatermarkConfig(scheme=scheme, key=42, gamma=0.25, delta=2.0)


# ---------------------------------------------------------------------------
# Basic interface
# ---------------------------------------------------------------------------

def test_detect_returns_float(detector):
    config = _make_config()
    result = detector.detect([1, 2, 3, 4, 5], vocab_size=1000, config=config)
    assert isinstance(result, float)


def test_empty_sequence_returns_zero(detector):
    config = _make_config()
    assert detector.detect([], vocab_size=1000, config=config) == 0.0


def test_is_watermarked_returns_bool(detector):
    config = _make_config()
    result = detector.is_watermarked([1, 2, 3], vocab_size=1000, config=config)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

def test_random_sequence_low_z_score(detector):
    """Random tokens should yield a z-score close to 0 on average."""
    config = _make_config()
    rng = random.Random(0)
    token_ids = [rng.randint(0, 999) for _ in range(500)]
    z = detector.detect(token_ids, vocab_size=1000, config=config)
    # For a large random sample, |z| should be well below the 4.0 threshold
    assert abs(z) < 4.0


def test_all_green_list_tokens_high_z_score(detector):
    """A sequence biased toward green-list tokens should yield high z-score."""
    config = _make_config()
    vocab_size = 1000
    # Find tokens that ARE in the green list
    from src.safety.watermark_detector import _extended_green_list_member
    green_tokens = [
        t for t in range(vocab_size)
        if _extended_green_list_member(t, config.key, vocab_size, config.gamma)
    ]
    assert len(green_tokens) > 0
    # Use only green tokens — 100% green fraction
    token_ids = (green_tokens * 50)[:200]
    z = detector.detect(token_ids, vocab_size=vocab_size, config=config)
    assert z > 4.0


def test_z_score_increases_with_more_green_tokens(detector):
    config = _make_config()
    vocab_size = 1000
    from src.safety.watermark_detector import _extended_green_list_member
    green_tokens = [
        t for t in range(vocab_size)
        if _extended_green_list_member(t, config.key, vocab_size, config.gamma)
    ]
    red_tokens = [
        t for t in range(vocab_size)
        if not _extended_green_list_member(t, config.key, vocab_size, config.gamma)
    ]
    # mostly red
    low_green = (red_tokens * 10)[:100]
    z_low = detector.detect(low_green, vocab_size=vocab_size, config=config)
    # mostly green
    high_green = (green_tokens * 10)[:100]
    z_high = detector.detect(high_green, vocab_size=vocab_size, config=config)
    assert z_high > z_low


# ---------------------------------------------------------------------------
# Schemes
# ---------------------------------------------------------------------------

def test_all_schemes_accept_tokens(detector):
    token_ids = list(range(50))
    for scheme in WatermarkScheme:
        config = WatermarkConfig(scheme=scheme, key=7, gamma=0.25, delta=2.0)
        z = detector.detect(token_ids, vocab_size=500, config=config)
        assert isinstance(z, float)


def test_multi_hash_scheme(detector):
    config = WatermarkConfig(scheme=WatermarkScheme.MULTI_HASH, key=42, gamma=0.25)
    z = detector.detect([10, 20, 30, 40, 50], vocab_size=100, config=config)
    assert isinstance(z, float)


# ---------------------------------------------------------------------------
# is_watermarked threshold
# ---------------------------------------------------------------------------

def test_not_watermarked_below_threshold(detector):
    config = _make_config()
    rng = random.Random(1)
    token_ids = [rng.randint(0, 999) for _ in range(300)]
    assert not detector.is_watermarked(token_ids, vocab_size=1000, config=config, z_threshold=4.0)


def test_is_watermarked_custom_threshold(detector):
    config = _make_config()
    vocab_size = 1000
    from src.safety.watermark_detector import _extended_green_list_member
    green_tokens = [
        t for t in range(vocab_size)
        if _extended_green_list_member(t, config.key, vocab_size, config.gamma)
    ]
    token_ids = (green_tokens * 50)[:200]
    assert detector.is_watermarked(token_ids, vocab_size=vocab_size, config=config, z_threshold=2.0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_safety_registry_contains_detector():
    assert "watermark_detector" in SAFETY_REGISTRY
    assert isinstance(SAFETY_REGISTRY["watermark_detector"], WatermarkDetector)
