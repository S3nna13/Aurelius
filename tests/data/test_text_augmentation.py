"""Tests for src/data/text_augmentation.py."""

from __future__ import annotations

import random

import pytest

from src.data.text_augmentation import (
    SYNONYM_DICT,
    InstructionAugmenter,
    TextAugConfig,
    apply_synonym_substitution,
    augment_dataset,
    generate_instruction_variants,
    paraphrase_with_template,
)


# ---------------------------------------------------------------------------
# TextAugConfig
# ---------------------------------------------------------------------------

def test_textaugconfig_defaults():
    config = TextAugConfig()
    assert config.p_paraphrase == 0.3
    assert config.p_synonym == 0.2
    assert config.p_instruction_variant == 0.5
    assert config.max_synonyms == 3
    assert config.seed == 42


# ---------------------------------------------------------------------------
# SYNONYM_DICT
# ---------------------------------------------------------------------------

def test_synonym_dict_has_20_plus_entries():
    assert len(SYNONYM_DICT) >= 20


def test_synonym_dict_values_are_lists():
    for key, value in SYNONYM_DICT.items():
        assert isinstance(value, list), f"Expected list for key '{key}'"
        assert len(value) > 0, f"Empty synonym list for key '{key}'"


# ---------------------------------------------------------------------------
# apply_synonym_substitution
# ---------------------------------------------------------------------------

def test_apply_synonym_substitution_p1_changes_words():
    """With p=1.0 every word that has a synonym should be replaced."""
    rng = random.Random(0)
    # Use words we know are in the dict
    text = "fast large small"
    result = apply_synonym_substitution(text, 1.0, SYNONYM_DICT, rng)
    assert result != text


def test_apply_synonym_substitution_p0_unchanged():
    rng = random.Random(0)
    text = "fast large small good"
    result = apply_synonym_substitution(text, 0.0, SYNONYM_DICT, rng)
    assert result == text


def test_apply_synonym_substitution_output_is_string():
    rng = random.Random(0)
    result = apply_synonym_substitution("fast good use", 0.5, SYNONYM_DICT, rng)
    assert isinstance(result, str)


def test_apply_synonym_substitution_empty_text():
    rng = random.Random(0)
    result = apply_synonym_substitution("", 1.0, SYNONYM_DICT, rng)
    assert result == ""


def test_apply_synonym_substitution_preserves_first_word_capitalization():
    rng = random.Random(0)
    text = "Fast cars are good."
    result = apply_synonym_substitution(text, 1.0, SYNONYM_DICT, rng)
    # First word should still be capitalized
    assert result[0].isupper()


# ---------------------------------------------------------------------------
# paraphrase_with_template
# ---------------------------------------------------------------------------

def test_paraphrase_with_template_contains_original_text():
    rng = random.Random(0)
    text = "the sky is blue"
    result = paraphrase_with_template(text, rng)
    assert text in result


def test_paraphrase_with_template_returns_different_string():
    rng = random.Random(0)
    text = "the sky is blue"
    result = paraphrase_with_template(text, rng)
    assert result != text


def test_paraphrase_with_template_is_string():
    rng = random.Random(0)
    result = paraphrase_with_template("hello world", rng)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# generate_instruction_variants
# ---------------------------------------------------------------------------

def test_generate_instruction_variants_returns_3_items():
    rng = random.Random(0)
    variants = generate_instruction_variants("Summarize this text.", rng)
    assert len(variants) == 3


def test_generate_instruction_variants_all_are_strings():
    rng = random.Random(0)
    variants = generate_instruction_variants("Explain the concept.", rng)
    for v in variants:
        assert isinstance(v, str)


def test_generate_instruction_variants_contain_instruction():
    rng = random.Random(0)
    instruction = "Describe the process."
    variants = generate_instruction_variants(instruction, rng)
    # At least two variants should contain the original instruction text
    containing = [v for v in variants if instruction in v]
    assert len(containing) >= 2


# ---------------------------------------------------------------------------
# augment_dataset
# ---------------------------------------------------------------------------

def test_augment_dataset_same_length_as_input():
    texts = ["Hello world.", "This is a test.", "Fast and large."]
    config = TextAugConfig()
    result = augment_dataset(texts, config)
    assert len(result) == len(texts)


def test_augment_dataset_deterministic_with_same_seed():
    texts = ["Hello world.", "This is a test.", "Fast and large."]
    config = TextAugConfig(seed=99)
    result1 = augment_dataset(texts, config)
    result2 = augment_dataset(texts, config)
    assert result1 == result2


def test_augment_dataset_returns_list_of_strings():
    texts = ["fast use find"]
    config = TextAugConfig()
    result = augment_dataset(texts, config)
    assert isinstance(result, list)
    assert all(isinstance(t, str) for t in result)


# ---------------------------------------------------------------------------
# InstructionAugmenter
# ---------------------------------------------------------------------------

def test_instruction_augmenter_augment_returns_list_of_tuples():
    config = TextAugConfig()
    augmenter = InstructionAugmenter(config)
    result = augmenter.augment("Explain this.", "Sure, here is an explanation.")
    assert isinstance(result, list)
    assert len(result) > 0
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2


def test_instruction_augmenter_augment_preserves_response():
    config = TextAugConfig()
    augmenter = InstructionAugmenter(config)
    response = "This is the response."
    result = augmenter.augment("Do something.", response)
    for _, resp in result:
        assert resp == response


def test_instruction_augmenter_augment_batch():
    config = TextAugConfig()
    augmenter = InstructionAugmenter(config)
    pairs = [
        ("Summarize this.", "Summary here."),
        ("Explain that.", "Explanation here."),
    ]
    result = augmenter.augment_batch(pairs)
    assert isinstance(result, list)
    # Each pair produces 3 variants, so 2 pairs -> 6 items
    assert len(result) == 6
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2
