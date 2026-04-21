"""Unit tests for reasoning_level_controller (GPT-OSS-120B, arXiv:2508.10925).

Covers parse_reasoning_level and apply_reasoning_level.
SWE-bench Verified: low=47.9%, medium=52.6%, high=62.4%.
"""
from __future__ import annotations

import pytest

from src.inference.reasoning_level_controller import (
    LEVEL_CONFIGS,
    apply_reasoning_level,
    parse_reasoning_level,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _low() -> dict:
    return parse_reasoning_level("Reasoning: low")


def _medium() -> dict:
    return parse_reasoning_level("Reasoning: medium")


def _high() -> dict:
    return parse_reasoning_level("Reasoning: high")


# ---------------------------------------------------------------------------
# 1. low temperature
# ---------------------------------------------------------------------------

def test_low_temperature() -> None:
    assert _low()["temperature"] == 0.3


# ---------------------------------------------------------------------------
# 2. low max_tokens
# ---------------------------------------------------------------------------

def test_low_max_tokens() -> None:
    assert _low()["max_tokens"] == 512


# ---------------------------------------------------------------------------
# 3. medium temperature
# ---------------------------------------------------------------------------

def test_medium_temperature() -> None:
    assert _medium()["temperature"] == 0.6


# ---------------------------------------------------------------------------
# 4. medium max_tokens
# ---------------------------------------------------------------------------

def test_medium_max_tokens() -> None:
    assert _medium()["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# 5. high temperature
# ---------------------------------------------------------------------------

def test_high_temperature() -> None:
    assert _high()["temperature"] == 1.0


# ---------------------------------------------------------------------------
# 6. high max_tokens
# ---------------------------------------------------------------------------

def test_high_max_tokens() -> None:
    assert _high()["max_tokens"] == 8192


# ---------------------------------------------------------------------------
# 7. missing prefix defaults to medium
# ---------------------------------------------------------------------------

def test_missing_prefix_defaults_medium() -> None:
    result = parse_reasoning_level("You are a helpful assistant.")
    assert result == dict(LEVEL_CONFIGS["medium"])


# ---------------------------------------------------------------------------
# 8. None prompt defaults to medium
# ---------------------------------------------------------------------------

def test_none_prompt_defaults_medium() -> None:
    result = parse_reasoning_level(None)
    assert result == dict(LEVEL_CONFIGS["medium"])


# ---------------------------------------------------------------------------
# 9. empty string defaults to medium
# ---------------------------------------------------------------------------

def test_empty_string_defaults_medium() -> None:
    result = parse_reasoning_level("")
    assert result == dict(LEVEL_CONFIGS["medium"])


# ---------------------------------------------------------------------------
# 10. case-insensitive uppercase
# ---------------------------------------------------------------------------

def test_case_insensitive_upper() -> None:
    result = parse_reasoning_level("REASONING: HIGH")
    assert result == dict(LEVEL_CONFIGS["high"])


# ---------------------------------------------------------------------------
# 11. case-insensitive mixed
# ---------------------------------------------------------------------------

def test_case_insensitive_mixed() -> None:
    result = parse_reasoning_level("Reasoning: Low")
    assert result == dict(LEVEL_CONFIGS["low"])


# ---------------------------------------------------------------------------
# 12. returns a copy (no shared state)
# ---------------------------------------------------------------------------

def test_returns_copy() -> None:
    a = parse_reasoning_level("Reasoning: high")
    b = parse_reasoning_level("Reasoning: high")
    assert a is not b
    # Mutating one should not affect the other or the canonical config.
    a["temperature"] = 99.0
    assert b["temperature"] == 1.0
    assert LEVEL_CONFIGS["high"]["temperature"] == 1.0


# ---------------------------------------------------------------------------
# 13. reasoning_level key always present
# ---------------------------------------------------------------------------

def test_reasoning_level_key_present() -> None:
    for prompt in (
        "Reasoning: low",
        "Reasoning: medium",
        "Reasoning: high",
        "no token at all",
        None,
        "",
    ):
        result = parse_reasoning_level(prompt)
        assert "reasoning_level" in result, f"key missing for prompt={prompt!r}"


# ---------------------------------------------------------------------------
# 14. apply_reasoning_level does not override explicit values
# ---------------------------------------------------------------------------

def test_apply_does_not_override_explicit() -> None:
    merged = apply_reasoning_level(
        "Reasoning: low",
        {"temperature": 0.9},
    )
    # Caller's explicit temperature must survive.
    assert merged["temperature"] == 0.9


# ---------------------------------------------------------------------------
# 15. apply_reasoning_level fills missing keys
# ---------------------------------------------------------------------------

def test_apply_adds_missing_keys() -> None:
    merged = apply_reasoning_level(
        "Reasoning: high",
        {"beam_size": 4},  # deliberately missing generation keys
    )
    # Keys from the high config are filled in.
    assert merged["max_tokens"] == 8192
    assert merged["top_p"] == 0.95
    assert merged["reasoning_level"] == "high"
    # Original caller key is preserved.
    assert merged["beam_size"] == 4


# ---------------------------------------------------------------------------
# Bonus: verify top_p values
# ---------------------------------------------------------------------------

def test_low_top_p() -> None:
    assert _low()["top_p"] == 0.9


def test_medium_top_p() -> None:
    assert _medium()["top_p"] == 0.95


def test_high_top_p() -> None:
    assert _high()["top_p"] == 0.95


# ---------------------------------------------------------------------------
# Bonus: reasoning_level field correctness per level
# ---------------------------------------------------------------------------

def test_reasoning_level_field_low() -> None:
    assert _low()["reasoning_level"] == "low"


def test_reasoning_level_field_medium() -> None:
    assert _medium()["reasoning_level"] == "medium"


def test_reasoning_level_field_high() -> None:
    assert _high()["reasoning_level"] == "high"
