"""Tests for the statistical prompt injection detector."""

from __future__ import annotations

import math

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.security.prompt_injection_detector import InjectionPattern, PromptInjectionDetector

# ---------------------------------------------------------------------------
# Shared tiny config and fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

SEED = 42
SEQ_LEN = 10


@pytest.fixture(scope="module")
def model() -> AureliusTransformer:
    torch.manual_seed(SEED)
    m = AureliusTransformer(TINY_CFG)
    m.eval()
    return m


@pytest.fixture(scope="module")
def detector(model) -> PromptInjectionDetector:
    return PromptInjectionDetector(model)


@pytest.fixture(scope="module")
def system_ids() -> list:
    torch.manual_seed(SEED)
    return torch.randint(60, 100, (SEQ_LEN,)).tolist()


@pytest.fixture(scope="module")
def user_ids() -> list:
    torch.manual_seed(SEED + 1)
    return torch.randint(60, 100, (SEQ_LEN,)).tolist()


# ---------------------------------------------------------------------------
# Test 1: instantiates without error (no patterns arg)
# ---------------------------------------------------------------------------


def test_instantiates_default_patterns(model):
    """PromptInjectionDetector instantiates without error using default patterns."""
    det = PromptInjectionDetector(model)
    assert det is not None
    assert len(det.patterns) > 0


# ---------------------------------------------------------------------------
# Test 2: instantiates with explicit patterns
# ---------------------------------------------------------------------------


def test_instantiates_with_explicit_patterns(model):
    """PromptInjectionDetector instantiates correctly with explicit patterns."""
    custom_patterns = [
        InjectionPattern(pattern_ids=[1, 2, 3, 4, 5], weight=1.5, label="custom_a"),
        InjectionPattern(pattern_ids=[6, 7, 8, 9, 10], weight=0.5, label="custom_b"),
    ]
    det = PromptInjectionDetector(model, patterns=custom_patterns)
    assert len(det.patterns) == 2
    assert det.patterns[0].label == "custom_a"
    assert det.patterns[1].label == "custom_b"


# ---------------------------------------------------------------------------
# Test 3: ngram_overlap returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_ngram_overlap_range():
    """ngram_overlap returns a float in [0, 1]."""
    ids_a = [1, 2, 3, 4, 5, 6, 7]
    ids_b = [3, 4, 5, 8, 9, 10, 11]
    result = PromptInjectionDetector.ngram_overlap(ids_a, ids_b, n=3)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Test 4: identical inputs give ngram_overlap = 1.0
# ---------------------------------------------------------------------------


def test_ngram_overlap_identical():
    """Identical token sequences give an ngram_overlap of exactly 1.0."""
    ids = [10, 20, 30, 40, 50, 60, 70]
    result = PromptInjectionDetector.ngram_overlap(ids, ids, n=3)
    assert result == 1.0


# ---------------------------------------------------------------------------
# Test 5: completely different inputs give ngram_overlap = 0.0
# ---------------------------------------------------------------------------


def test_ngram_overlap_disjoint():
    """Completely disjoint token sequences give an ngram_overlap of 0.0."""
    ids_a = [1, 2, 3, 4, 5]
    ids_b = [100, 101, 102, 103, 104]
    result = PromptInjectionDetector.ngram_overlap(ids_a, ids_b, n=3)
    assert result == 0.0


# ---------------------------------------------------------------------------
# Test 6: perplexity_ratio returns positive float
# ---------------------------------------------------------------------------


def test_perplexity_ratio_positive(detector, system_ids, user_ids):
    """perplexity_ratio returns a positive float."""
    ratio = detector.perplexity_ratio(system_ids, user_ids)
    assert isinstance(ratio, float)
    assert ratio > 0.0


# ---------------------------------------------------------------------------
# Test 7: perplexity_ratio is finite
# ---------------------------------------------------------------------------


def test_perplexity_ratio_finite(detector, system_ids, user_ids):
    """perplexity_ratio is a finite number (no inf or nan)."""
    ratio = detector.perplexity_ratio(system_ids, user_ids)
    assert math.isfinite(ratio)


# ---------------------------------------------------------------------------
# Test 8: pattern_score returns float >= 0
# ---------------------------------------------------------------------------


def test_pattern_score_non_negative(detector, user_ids):
    """pattern_score returns a non-negative float."""
    score = detector.pattern_score(user_ids)
    assert isinstance(score, float)
    assert score >= 0.0


# ---------------------------------------------------------------------------
# Test 9: detect returns dict with expected keys
# ---------------------------------------------------------------------------


def test_detect_returns_expected_keys(detector, system_ids, user_ids):
    """detect returns a dict containing all four expected keys."""
    result = detector.detect(system_ids, user_ids)
    assert isinstance(result, dict)
    assert "is_injection" in result
    assert "pattern_score" in result
    assert "perplexity_ratio" in result
    assert "overlap" in result


# ---------------------------------------------------------------------------
# Test 10: detect 'is_injection' value is bool
# ---------------------------------------------------------------------------


def test_detect_is_injection_is_bool(detector, system_ids, user_ids):
    """detect returns a Python bool for the 'is_injection' key."""
    result = detector.detect(system_ids, user_ids)
    assert isinstance(result["is_injection"], bool)


# ---------------------------------------------------------------------------
# Test 11: very high thresholds -> is_injection = False
# ---------------------------------------------------------------------------


def test_detect_high_thresholds_not_injection(detector, system_ids, user_ids):
    """With very high thresholds, detect never flags as injection."""
    result = detector.detect(
        system_ids,
        user_ids,
        overlap_threshold=1e9,
        perplexity_ratio_threshold=1e9,
    )
    assert result["is_injection"] is False


# ---------------------------------------------------------------------------
# Test 12: batch_detect returns list of correct length
# ---------------------------------------------------------------------------


def test_batch_detect_correct_length(detector, system_ids):
    """batch_detect returns a list whose length matches the number of user messages."""
    torch.manual_seed(SEED + 10)
    user_batch = [torch.randint(60, 100, (SEQ_LEN,)).tolist() for _ in range(5)]
    results = detector.batch_detect(system_ids, user_batch)
    assert isinstance(results, list)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Test 13: each element of batch_detect output is a dict with expected keys
# ---------------------------------------------------------------------------


def test_batch_detect_elements_have_expected_keys(detector, system_ids):
    """Every element of batch_detect output is a dict with the four detection keys."""
    torch.manual_seed(SEED + 20)
    user_batch = [torch.randint(60, 100, (SEQ_LEN,)).tolist() for _ in range(3)]
    results = detector.batch_detect(system_ids, user_batch)
    expected_keys = {"is_injection", "pattern_score", "perplexity_ratio", "overlap"}
    for i, item in enumerate(results):
        assert isinstance(item, dict), f"Element {i} is not a dict"
        assert set(item.keys()) == expected_keys, f"Element {i} missing keys"
