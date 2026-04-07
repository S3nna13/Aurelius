"""Tests for src/data/rephrase.py."""

import pytest

from src.data.rephrase import (
    RephraseConfig,
    RephrasedExample,
    build_rephrase_prompt,
    rephrase_batch,
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


def test_build_rephrase_prompt_boundary_min():
    """Text of exactly min_input_chars should produce a non-empty prompt."""
    config = RephraseConfig(min_input_chars=50)
    text = _make_text(50)
    prompt = build_rephrase_prompt(text, config)
    assert text in prompt


def test_build_rephrase_prompt_boundary_max():
    """Text of exactly max_input_chars should produce a non-empty prompt."""
    config = RephraseConfig(max_input_chars=2048)
    text = _make_text(2048)
    prompt = build_rephrase_prompt(text, config)
    assert text in prompt


def test_build_rephrase_prompt_custom_template():
    """Custom template is used when provided."""
    config = RephraseConfig(template="REWRITE: {text}", min_input_chars=5)
    text = "hello world"
    prompt = build_rephrase_prompt(text, config)
    assert prompt == "REWRITE: hello world"


# ---------------------------------------------------------------------------
# rephrase_batch
# ---------------------------------------------------------------------------

def test_rephrase_batch_pairs_correctly():
    """rephrase_batch must pair originals with rephrased outputs correctly."""
    config = RephraseConfig(min_input_chars=5)
    texts = ["hello world", "foo bar baz"]
    rephrased_outputs = ["hi there world", ""]

    examples = rephrase_batch(texts, rephrased_outputs, config)

    assert len(examples) == 2

    # First pair: non-empty rephrased
    assert examples[0].original == "hello world"
    assert examples[0].rephrased == "hi there world"
    assert examples[0].was_rephrased is True
    assert "hello world" in examples[0].prompt

    # Second pair: empty rephrased
    assert examples[1].original == "foo bar baz"
    assert examples[1].rephrased == ""
    assert examples[1].was_rephrased is False


def test_rephrase_batch_length_mismatch_raises():
    """Mismatched list lengths must raise ValueError."""
    with pytest.raises(ValueError):
        rephrase_batch(["a", "b"], ["only one"])


def test_rephrase_batch_returns_rephrasedexample_instances():
    """Each item returned must be a RephrasedExample."""
    config = RephraseConfig(min_input_chars=1)
    examples = rephrase_batch(["x"], ["y"], config)
    assert isinstance(examples[0], RephrasedExample)
