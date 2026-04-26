"""Tests for src/data/rephrase.py."""

from src.data.rephrase import (
    RephraseConfig,
    build_rephrase_prompt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text(length: int) -> str:
    """Return a string of exactly *length* 'a' characters."""
    return "a" * length


# ---------------------------------------------------------------------------
# build_rephrase_prompt
# ---------------------------------------------------------------------------


def test_build_rephrase_prompt_basic():
    """Prompt must contain the input text."""
    text = _make_text(100)  # well within default 50–2048 window
    prompt = build_rephrase_prompt(text)
    assert text in prompt
    assert len(prompt) > len(text)


def test_build_rephrase_prompt_too_short():
    """Text shorter than min_input_chars must return empty string."""
    config = RephraseConfig(min_input_chars=50)
    short_text = _make_text(49)
    assert build_rephrase_prompt(short_text, config) == ""


def test_build_rephrase_prompt_too_long():
    """Text longer than max_input_chars must return empty string."""
    config = RephraseConfig(max_input_chars=2048)
    long_text = _make_text(2049)
    assert build_rephrase_prompt(long_text, config) == ""


def test_build_rephrase_prompt_custom_template():
    """Custom template is used when provided."""
    config = RephraseConfig(template="REWRITE: {text}", min_input_chars=5)
    text = "hello world"
    prompt = build_rephrase_prompt(text, config)
    assert prompt == "REWRITE: hello world"
