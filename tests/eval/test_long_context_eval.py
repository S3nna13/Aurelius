"""Tests for long-context evaluation module."""

from __future__ import annotations

import math
import random

import pytest
import torch

from src.eval.long_context_eval import (
    LongContextConfig,
    LongContextEvaluator,
    create_needle_prompt,
    create_passkey_prompt,
    extract_passkey_from_output,
    generate_distractor_text,
    greedy_generate_text,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def byte_encode(text: str) -> list[int]:
    """Byte-level tokenizer: text -> utf-8 bytes -> list[int]."""
    return list(text.encode("utf-8"))


def byte_decode(ids: list[int]) -> str:
    """Byte-level detokenizer: list[int] -> str (replace unmappable bytes)."""
    return bytes(ids).decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def small_model():
    """Tiny AureliusTransformer for fast test execution."""
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


@pytest.fixture
def default_config():
    return LongContextConfig()


@pytest.fixture
def small_config():
    """Config scaled down for fast tests (fewer distractors)."""
    return LongContextConfig(
        max_context_len=512,
        n_distractors=4,
        passkey_length=5,
        eval_positions=[0.25, 0.75],
        seed=0,
    )


@pytest.fixture
def evaluator(small_model, small_config):
    return LongContextEvaluator(
        model=small_model,
        tokenizer_encode=byte_encode,
        tokenizer_decode=byte_decode,
        config=small_config,
    )


# ---------------------------------------------------------------------------
# 1. LongContextConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults(default_config):
    assert default_config.max_context_len == 8192
    assert default_config.n_distractors == 10
    assert default_config.passkey_length == 5
    assert default_config.eval_positions == [0.1, 0.25, 0.5, 0.75, 0.9]
    assert default_config.seed == 42


# ---------------------------------------------------------------------------
# 2. generate_distractor_text returns string of reasonable length
# ---------------------------------------------------------------------------


def test_generate_distractor_text_reasonable_length():
    rng = random.Random(42)
    n = 10
    text = generate_distractor_text(rng, n)
    assert isinstance(text, str)
    # Each sentence ~50+ chars; allow generous lower bound
    assert len(text) >= n * 20, f"Expected at least {n * 20} chars, got {len(text)}"


# ---------------------------------------------------------------------------
# 3. generate_distractor_text different seeds give different text
# ---------------------------------------------------------------------------


def test_generate_distractor_text_different_seeds():
    text1 = generate_distractor_text(random.Random(1), 5)
    text2 = generate_distractor_text(random.Random(999), 5)
    # With different seeds, results should differ (extremely likely with 20 templates)
    assert text1 != text2


# ---------------------------------------------------------------------------
# 4. create_passkey_prompt returns tuple (prompt, passkey)
# ---------------------------------------------------------------------------


def test_create_passkey_prompt_returns_tuple(default_config):
    rng = random.Random(0)
    result = create_passkey_prompt(rng, default_config, 0.5)
    assert isinstance(result, tuple)
    assert len(result) == 2
    prompt, passkey = result
    assert isinstance(prompt, str)
    assert isinstance(passkey, str)


# ---------------------------------------------------------------------------
# 5. create_passkey_prompt passkey is numeric and correct length
# ---------------------------------------------------------------------------


def test_create_passkey_prompt_passkey_numeric_correct_length(default_config):
    rng = random.Random(7)
    _, passkey = create_passkey_prompt(rng, default_config, 0.5)
    assert passkey.isdigit(), f"Expected all digits, got: {passkey!r}"
    assert len(passkey) == default_config.passkey_length


# ---------------------------------------------------------------------------
# 6. create_passkey_prompt passkey appears in prompt
# ---------------------------------------------------------------------------


def test_create_passkey_prompt_passkey_in_prompt(default_config):
    rng = random.Random(13)
    prompt, passkey = create_passkey_prompt(rng, default_config, 0.5)
    assert passkey in prompt, f"Passkey {passkey!r} not found in prompt"


# ---------------------------------------------------------------------------
# 7. create_needle_prompt needle appears in returned prompt
# ---------------------------------------------------------------------------


def test_create_needle_prompt_needle_in_prompt():
    rng = random.Random(42)
    needle = "XRAY-FOXTROT-7749"
    prompt, returned_needle = create_needle_prompt(rng, needle, 0.5, 6)
    assert needle in prompt
    assert returned_needle == needle


# ---------------------------------------------------------------------------
# 8. extract_passkey_from_output finds correct passkey
# ---------------------------------------------------------------------------


def test_extract_passkey_from_output_finds_correct():
    output = "The answer is 83742 based on the context provided."
    result = extract_passkey_from_output(output, 5)
    assert result == "83742"


# ---------------------------------------------------------------------------
# 9. extract_passkey_from_output returns None when not present
# ---------------------------------------------------------------------------


def test_extract_passkey_from_output_none_when_missing():
    output = "I do not know the answer."
    result = extract_passkey_from_output(output, 5)
    assert result is None


# ---------------------------------------------------------------------------
# 10. extract_passkey_from_output handles wrong length digits
# ---------------------------------------------------------------------------


def test_extract_passkey_from_output_wrong_length():
    # Output has a 3-digit and a 7-digit number, but we want 5-digit
    output = "The values are 123 and 1234567."
    result = extract_passkey_from_output(output, 5)
    assert result is None


# ---------------------------------------------------------------------------
# 11. greedy_generate_text returns a string
# ---------------------------------------------------------------------------


def test_greedy_generate_text_returns_string(small_model):
    output = greedy_generate_text(
        small_model,
        byte_encode,
        byte_decode,
        "Hello world",
        max_new=5,
    )
    assert isinstance(output, str)


# ---------------------------------------------------------------------------
# 12. LongContextEvaluator.evaluate_passkey_retrieval returns correct keys
# ---------------------------------------------------------------------------


def test_evaluate_passkey_retrieval_keys(evaluator, small_config):
    results = evaluator.evaluate_passkey_retrieval(n_eval=2)
    assert "mean_accuracy" in results
    for pos in small_config.eval_positions:
        key = f"acc_at_{pos}"
        assert key in results, f"Missing key: {key}"


def test_evaluate_passkey_retrieval_values_in_range(evaluator, small_config):
    results = evaluator.evaluate_passkey_retrieval(n_eval=2)
    for k, v in results.items():
        assert 0.0 <= v <= 1.0, f"Value out of range for key {k}: {v}"


# ---------------------------------------------------------------------------
# 13. LongContextEvaluator.compute_perplexity_at_length returns list of floats
# ---------------------------------------------------------------------------


def test_compute_perplexity_at_length_returns_list_of_floats(evaluator):
    # Generate enough text to form at least one chunk
    text = "The cat sat on the mat. " * 40
    results = evaluator.compute_perplexity_at_length(text, chunk_size=64)
    assert isinstance(results, list)
    assert len(results) > 0
    for ppl in results:
        assert isinstance(ppl, float)
        assert math.isfinite(ppl)
        assert ppl > 0.0


def test_compute_perplexity_at_length_short_text_empty(evaluator):
    """Single character text should return empty list (not enough tokens)."""
    results = evaluator.compute_perplexity_at_length("A", chunk_size=64)
    assert isinstance(results, list)
    # Single byte = 1 token, not enough for input+target pair
    assert results == []
