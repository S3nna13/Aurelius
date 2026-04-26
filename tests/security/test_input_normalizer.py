"""Tests for input_normalizer — evasion-resistant text canonicalization.

Security surface: STRIDE Spoofing / Tampering.
"""
from __future__ import annotations

import pytest

from src.security.input_normalizer import (
    InputNormalizer,
    NORMALIZER_REGISTRY,
    DEFAULT_INPUT_NORMALIZER,
)


# ---------------------------------------------------------------------------
# Zero-width character removal
# ---------------------------------------------------------------------------


def test_strips_zero_width_space():
    n = InputNormalizer()
    text = "hello\u200bworld"
    result = n.normalize(text)
    assert result.normalized == "helloworld"
    assert result.zero_width_removed == 1


def test_strips_multiple_zero_width_chars():
    n = InputNormalizer()
    text = "h\u200be\u200dl\u200cl\u200bo"
    result = n.normalize(text)
    assert result.normalized == "hello"
    assert result.zero_width_removed == 4


# ---------------------------------------------------------------------------
# Homoglyph replacement
# ---------------------------------------------------------------------------


def test_replaces_cyrillic_homoglyphs():
    n = InputNormalizer()
    # "hello" with Cyrillic а, е, о
    text = "hеllо"  # U+0435, U+043E
    result = n.normalize(text)
    assert result.normalized == "hello"
    assert result.homoglyphs_replaced >= 2


def test_no_change_for_pure_ascii():
    n = InputNormalizer()
    text = "hello world"
    result = n.normalize(text)
    assert result.normalized == text
    assert result.changes == []


# ---------------------------------------------------------------------------
# NFC normalization
# ---------------------------------------------------------------------------


def test_nfc_combines_accents():
    n = InputNormalizer()
    # e + combining acute → é (precomposed)
    text = "e\u0301"
    result = n.normalize(text)
    assert result.normalized == "\u00e9"
    assert "nfc_normalize" in result.changes


# ---------------------------------------------------------------------------
# Combined attacks
# ---------------------------------------------------------------------------


def test_combined_zero_width_and_homoglyphs():
    n = InputNormalizer()
    text = "h\u200bеllо"  # zero-width + Cyrillic
    result = n.normalize(text)
    assert result.normalized == "hello"
    assert result.zero_width_removed == 1
    assert result.homoglyphs_replaced >= 2


# ---------------------------------------------------------------------------
# Convenience normalize_text
# ---------------------------------------------------------------------------


def test_normalize_text_returns_string():
    n = InputNormalizer()
    assert n.normalize_text("h\u200bеllо") == "hello"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in NORMALIZER_REGISTRY
    assert isinstance(NORMALIZER_REGISTRY["default"], InputNormalizer)


def test_default_is_normalizer():
    assert isinstance(DEFAULT_INPUT_NORMALIZER, InputNormalizer)
