"""
Tests for src/data/synthetic_augmentation.py
Uses only stdlib + PyTorch; no HuggingFace / scipy / sklearn / nltk.
"""

from __future__ import annotations

import random

import pytest

from src.data.synthetic_augmentation import (
    SyntheticAugConfig,
    SyntheticAugmentor,
    compute_token_overlap,
    mask_tokens,
    random_token_insertion,
    random_token_replacement,
    sentence_order_permutation,
    span_masking,
)

TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]
RNG = random.Random(42)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------
def test_config_defaults():
    cfg = SyntheticAugConfig()
    assert cfg.p_mask == 0.15
    assert cfg.p_insert == 0.1
    assert cfg.p_replace == 0.1
    assert cfg.vocab_size == 32000
    assert cfg.mask_token_id == 103
    assert cfg.n_augments == 3
    assert cfg.seed is None


# ---------------------------------------------------------------------------
# 2. mask_tokens — length unchanged
# ---------------------------------------------------------------------------
def test_mask_tokens_length_unchanged():
    rng = random.Random(0)
    result = mask_tokens(TOKENS, p=0.5, mask_id=103, rng=rng)
    assert len(result) == len(TOKENS)


# ---------------------------------------------------------------------------
# 3. mask_tokens — never returns empty list
# ---------------------------------------------------------------------------
def test_mask_tokens_never_empty():
    rng = random.Random(0)
    result = mask_tokens(TOKENS, p=1.0, mask_id=103, rng=rng)
    # At least one token must NOT be the mask id
    assert len(result) > 0
    non_masked = [t for t in result if t != 103]
    assert len(non_masked) >= 1


# ---------------------------------------------------------------------------
# 4. mask_tokens — at least one token masked when p=1
# ---------------------------------------------------------------------------
def test_mask_tokens_at_least_one_masked_when_p1():
    rng = random.Random(0)
    result = mask_tokens(TOKENS, p=1.0, mask_id=103, rng=rng)
    masked_count = sum(1 for t in result if t == 103)
    # With p=1 all tokens targeted, but one is kept → n-1 masked
    assert masked_count == len(TOKENS) - 1


# ---------------------------------------------------------------------------
# 5. random_token_insertion — longer than original when p=1
# ---------------------------------------------------------------------------
def test_random_token_insertion_longer_when_p1():
    rng = random.Random(0)
    result = random_token_insertion(TOKENS, p=1.0, vocab_size=32000, rng=rng)
    # Every token gets one insertion → length should be 2x original
    assert len(result) == len(TOKENS) * 2


# ---------------------------------------------------------------------------
# 6. random_token_replacement — same length
# ---------------------------------------------------------------------------
def test_random_token_replacement_same_length():
    rng = random.Random(0)
    result = random_token_replacement(TOKENS, p=0.5, vocab_size=32000, rng=rng)
    assert len(result) == len(TOKENS)


# ---------------------------------------------------------------------------
# 7. sentence_order_permutation — returns a list
# ---------------------------------------------------------------------------
def test_sentence_order_permutation_returns_list():
    # Use sep_id=4 so we get two segments [1,2,3] and [5,6,7,8]
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    result = sentence_order_permutation(tokens, sep_id=4)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 8. sentence_order_permutation — unchanged if fewer than 2 sentences
# ---------------------------------------------------------------------------
def test_sentence_order_permutation_unchanged_single_sentence():
    tokens = [1, 2, 3, 4, 5]
    # sep_id=99 not present → single sentence → returned unchanged
    result = sentence_order_permutation(tokens, sep_id=99)
    assert result == tokens


# ---------------------------------------------------------------------------
# 9. span_masking — length unchanged
# ---------------------------------------------------------------------------
def test_span_masking_length_unchanged():
    rng = random.Random(0)
    result = span_masking(TOKENS, mask_ratio=0.3, max_span_len=3, mask_id=103, rng=rng)
    assert len(result) == len(TOKENS)


# ---------------------------------------------------------------------------
# 10. compute_token_overlap — identical sequences → 1.0
# ---------------------------------------------------------------------------
def test_compute_token_overlap_identical():
    overlap = compute_token_overlap(TOKENS, TOKENS)
    assert overlap == 1.0


# ---------------------------------------------------------------------------
# 11. compute_token_overlap — disjoint sequences → 0.0
# ---------------------------------------------------------------------------
def test_compute_token_overlap_disjoint():
    seq_a = [1, 2, 3]
    seq_b = [4, 5, 6]
    overlap = compute_token_overlap(seq_a, seq_b)
    assert overlap == 0.0


# ---------------------------------------------------------------------------
# 12. SyntheticAugmentor.augment — returns a list
# ---------------------------------------------------------------------------
def test_augmentor_augment_returns_list():
    cfg = SyntheticAugConfig(seed=0)
    aug = SyntheticAugmentor(cfg)
    result = aug.augment(TOKENS)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 13. Augmentor result differs from original when p > 0
# ---------------------------------------------------------------------------
def test_augmentor_result_differs_from_original():
    # Use high probabilities to make it extremely likely that something changes
    cfg = SyntheticAugConfig(p_mask=0.9, p_replace=0.9, seed=1)
    aug = SyntheticAugmentor(cfg)
    # Try multiple times in case of rare RNG hits
    changed = False
    for _ in range(20):
        result = aug.augment(list(TOKENS), methods=["mask", "replace"])
        if result != TOKENS:
            changed = True
            break
    assert changed, "Expected augmented result to differ from original with high p"


# ---------------------------------------------------------------------------
# 14. augment_batch — correct total length
# ---------------------------------------------------------------------------
def test_augment_batch_correct_total_length():
    cfg = SyntheticAugConfig(seed=0)
    aug = SyntheticAugmentor(cfg)
    batch = [TOKENS, [10, 20, 30]]
    n_augments = 2
    result = aug.augment_batch(batch, n_augments=n_augments)
    expected_len = len(batch) * (1 + n_augments)
    assert len(result) == expected_len


# ---------------------------------------------------------------------------
# 15. get_stats — has required keys
# ---------------------------------------------------------------------------
def test_get_stats_has_required_keys():
    cfg = SyntheticAugConfig(seed=0)
    aug = SyntheticAugmentor(cfg)
    augmented = aug.augment(TOKENS)
    stats = aug.get_stats(TOKENS, augmented)
    for key in ("original_len", "augmented_len", "overlap", "change_ratio"):
        assert key in stats, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 16. get_stats — overlap in [0, 1]
# ---------------------------------------------------------------------------
def test_get_stats_overlap_in_range():
    cfg = SyntheticAugConfig(seed=0)
    aug = SyntheticAugmentor(cfg)
    augmented = aug.augment(TOKENS)
    stats = aug.get_stats(TOKENS, augmented)
    assert 0.0 <= stats["overlap"] <= 1.0


# ---------------------------------------------------------------------------
# 17. augment with unknown method raises ValueError
# ---------------------------------------------------------------------------
def test_augment_unknown_method_raises():
    cfg = SyntheticAugConfig(seed=0)
    aug = SyntheticAugmentor(cfg)
    with pytest.raises(ValueError, match="Unknown augmentation method"):
        aug.augment(TOKENS, methods=["nonexistent"])


# ---------------------------------------------------------------------------
# 18. span_masking actually masks some tokens when ratio > 0
# ---------------------------------------------------------------------------
def test_span_masking_masks_tokens():
    rng = random.Random(7)
    result = span_masking(TOKENS, mask_ratio=0.5, max_span_len=3, mask_id=103, rng=rng)
    assert 103 in result


# ---------------------------------------------------------------------------
# 19. mask_tokens with p=0 — nothing masked
# ---------------------------------------------------------------------------
def test_mask_tokens_p0_nothing_masked():
    rng = random.Random(0)
    result = mask_tokens(TOKENS, p=0.0, mask_id=103, rng=rng)
    assert result == TOKENS


# ---------------------------------------------------------------------------
# 20. augment_batch preserves originals as the first N entries
# ---------------------------------------------------------------------------
def test_augment_batch_preserves_originals():
    cfg = SyntheticAugConfig(seed=0)
    aug = SyntheticAugmentor(cfg)
    batch = [[1, 2, 3], [4, 5, 6]]
    result = aug.augment_batch(batch, n_augments=1)
    # First len(batch) entries should be the originals
    for orig, res in zip(batch, result[: len(batch)]):
        assert orig == res
