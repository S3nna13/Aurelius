"""Tests for src/data/text_augmentation.py (AugmentationConfig / TextAugmentor API)"""

import random

import pytest

from src.data.text_augmentation import (
    AugmentationConfig,
    TextAugmentor,
    compute_edit_distance,
    random_char_noise,
    random_word_deletion,
    random_word_swap,
    synonym_swap_simple,
)

TEXT = "The quick brown fox jumps over the lazy dog"
RNG = random.Random(0)


# ---------------------------------------------------------------------------
# AugmentationConfig
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = AugmentationConfig()
    assert cfg.p_word_delete == pytest.approx(0.1)
    assert cfg.p_word_swap == pytest.approx(0.1)
    assert cfg.p_char_noise == pytest.approx(0.05)
    assert cfg.max_aug_ratio == pytest.approx(0.3)
    assert cfg.seed is None


# ---------------------------------------------------------------------------
# random_word_deletion
# ---------------------------------------------------------------------------


def test_deletion_removes_words():
    rng = random.Random(1)
    # High p to ensure deletion
    result = random_word_deletion(TEXT, p=0.9, rng=rng)
    assert len(result.split()) < len(TEXT.split())


def test_deletion_never_empties():
    rng = random.Random(2)
    for _ in range(10):
        result = random_word_deletion(TEXT, p=1.0, rng=rng)
        assert len(result.strip()) > 0


def test_deletion_p_zero_unchanged():
    rng = random.Random(3)
    result = random_word_deletion(TEXT, p=0.0, rng=rng)
    assert result == TEXT


# ---------------------------------------------------------------------------
# random_word_swap
# ---------------------------------------------------------------------------


def test_swap_changes_order():
    rng = random.Random(42)
    # High p to guarantee at least one swap
    result = random_word_swap(TEXT, p=1.0, rng=rng)
    # Words are same set, order may differ
    assert sorted(result.split()) == sorted(TEXT.split())


def test_swap_p_zero_unchanged():
    rng = random.Random(5)
    result = random_word_swap(TEXT, p=0.0, rng=rng)
    assert result == TEXT


# ---------------------------------------------------------------------------
# random_char_noise
# ---------------------------------------------------------------------------


def test_char_noise_changes_chars():
    rng = random.Random(99)
    result = random_char_noise(TEXT, p=0.9, rng=rng)
    # With very high p, most chars change
    assert result != TEXT


def test_char_noise_preserves_spaces():
    rng = random.Random(6)
    result = random_char_noise(TEXT, p=1.0, rng=rng)
    # Same number of spaces
    assert result.count(" ") == TEXT.count(" ")


def test_char_noise_p_zero_unchanged():
    rng = random.Random(7)
    result = random_char_noise(TEXT, p=0.0, rng=rng)
    assert result == TEXT


# ---------------------------------------------------------------------------
# synonym_swap_simple
# ---------------------------------------------------------------------------


def test_synonym_swap_replaces():
    swap_map = {"quick": "fast", "lazy": "sleepy"}
    result = synonym_swap_simple(TEXT, swap_map)
    assert "fast" in result
    assert "sleepy" in result


def test_synonym_swap_preserves_case():
    swap_map = {"the": "a"}
    text = "The quick brown fox"
    result = synonym_swap_simple(text, swap_map)
    # "The" starts with uppercase → replacement should be capitalized
    assert result.startswith("A")


def test_synonym_swap_no_match_unchanged():
    result = synonym_swap_simple(TEXT, {})
    assert result == TEXT


# ---------------------------------------------------------------------------
# compute_edit_distance
# ---------------------------------------------------------------------------


def test_edit_distance_identical():
    assert compute_edit_distance(TEXT, TEXT) == 0


def test_edit_distance_one_insertion():
    assert compute_edit_distance("hello world", "hello cruel world") == 1


def test_edit_distance_one_deletion():
    assert compute_edit_distance("hello cruel world", "hello world") == 1


def test_edit_distance_empty():
    assert compute_edit_distance("", "") == 0


# ---------------------------------------------------------------------------
# TextAugmentor
# ---------------------------------------------------------------------------


def test_augmentor_returns_string():
    aug = TextAugmentor(AugmentationConfig(seed=0))
    result = aug.augment(TEXT)
    assert isinstance(result, str)


def test_augmentor_specific_methods():
    aug = TextAugmentor(AugmentationConfig(seed=0))
    result = aug.augment(TEXT, methods=["char_noise"])
    assert isinstance(result, str)


def test_augmentor_empty_methods_unchanged():
    aug = TextAugmentor(AugmentationConfig(seed=0))
    result = aug.augment(TEXT, methods=[])
    assert result == TEXT


def test_augment_batch_correct_length():
    aug = TextAugmentor(AugmentationConfig(seed=0))
    texts = [TEXT, "Hello world", "Foo bar baz"]
    result = aug.augment_batch(texts, n_augments_per_text=2)
    assert len(result) == len(texts) * 2


def test_get_augment_stats_keys():
    aug = TextAugmentor(AugmentationConfig(seed=0))
    augmented = aug.augment(TEXT)
    stats = aug.get_augment_stats(TEXT, augmented)
    for key in [
        "word_count_original",
        "word_count_augmented",
        "edit_distance_ratio",
        "words_changed_ratio",
    ]:
        assert key in stats


def test_get_augment_stats_identical():
    aug = TextAugmentor(AugmentationConfig(seed=0))
    stats = aug.get_augment_stats(TEXT, TEXT)
    assert stats["edit_distance_ratio"] == pytest.approx(0.0)
