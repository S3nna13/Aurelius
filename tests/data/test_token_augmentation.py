"""
Tests for token-level data augmentation module.
"""

import random

import pytest
import torch

from src.data.token_augmentation import (
    AugmentationConfig,
    TokenAugmentor,
    mixup_sequences,
    mlm_masking,
    random_deletion,
    random_insertion,
    random_swap,
    span_masking,
    token_cutout,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_tokens():
    """A typical token sequence: BOS + body + EOS."""
    return [1, 10, 20, 30, 40, 50, 60, 70, 80, 2]


@pytest.fixture
def rng():
    return random.Random(42)


# ── 1. random_deletion: output shorter than or equal input ──────────────────


def test_random_deletion_shorter_or_equal(sample_tokens, rng):
    result = random_deletion(sample_tokens, prob=0.5, keep_special_tokens=True, rng=rng)
    assert len(result) <= len(sample_tokens)


# ── 2. random_deletion with prob=0.0: returns same tokens ──────────────────


def test_random_deletion_prob_zero(sample_tokens, rng):
    result = random_deletion(sample_tokens, prob=0.0, keep_special_tokens=True, rng=rng)
    assert result == sample_tokens


# ── 3. random_deletion with prob=1.0 ───────────────────────────────────────


def test_random_deletion_prob_one_keep_special(sample_tokens, rng):
    # With keep_special_tokens=True, first and last must survive.
    result = random_deletion(sample_tokens, prob=1.0, keep_special_tokens=True, rng=rng)
    assert len(result) == 2
    assert result[0] == sample_tokens[0]
    assert result[-1] == sample_tokens[-1]


def test_random_deletion_prob_one_no_special(sample_tokens, rng):
    # With keep_special_tokens=False everything gets deleted.
    result = random_deletion(sample_tokens, prob=1.0, keep_special_tokens=False, rng=rng)
    assert result == []


# ── 4. random_swap: output same length as input ─────────────────────────────


def test_random_swap_same_length(sample_tokens, rng):
    result = random_swap(sample_tokens, prob=0.5, keep_special_tokens=True, rng=rng)
    assert len(result) == len(sample_tokens)


# ── 5. random_swap with prob=0.0: returns same tokens ──────────────────────


def test_random_swap_prob_zero(sample_tokens, rng):
    result = random_swap(sample_tokens, prob=0.0, keep_special_tokens=True, rng=rng)
    assert result == sample_tokens


# ── 6. random_insertion: output longer than or equal input ─────────────────


def test_random_insertion_longer_or_equal(sample_tokens, rng):
    result = random_insertion(
        sample_tokens, prob=0.9, vocab_size=128, keep_special_tokens=True, rng=rng
    )
    assert len(result) >= len(sample_tokens)


def test_random_insertion_prob_zero(sample_tokens, rng):
    result = random_insertion(
        sample_tokens, prob=0.0, vocab_size=128, keep_special_tokens=True, rng=rng
    )
    assert result == sample_tokens


# ── 7. mlm_masking: labels has -100 for unmasked positions ─────────────────


def test_mlm_masking_unmasked_labels_are_neg100(sample_tokens, rng):
    masked, labels = mlm_masking(
        sample_tokens,
        mask_token_id=0,
        mask_prob=0.5,
        vocab_size=128,
        keep_special_tokens=True,
        rng=rng,
    )
    for i, (orig, label) in enumerate(zip(sample_tokens, labels)):
        if label != -100:
            # label must equal the original token when it was selected for masking
            assert label == orig
        # unmasked positions: label == -100 and token is unchanged
        if label == -100:
            assert masked[i] == sample_tokens[i]


# ── 8. mlm_masking: masked tokens are mask_token_id or random at masked positions


def test_mlm_masking_masked_values(rng):
    tokens = list(range(2, 50))  # large enough to get some masks
    mask_id = 0
    vocab_size = 128
    masked, labels = mlm_masking(
        tokens,
        mask_token_id=mask_id,
        mask_prob=0.5,
        vocab_size=vocab_size,
        keep_special_tokens=False,
        rng=rng,
    )
    assert len(masked) == len(tokens)
    assert len(labels) == len(tokens)
    for i, label in enumerate(labels):
        if label != -100:
            # masked position: value must be mask_id, a random token, or original
            assert masked[i] == mask_id or (0 <= masked[i] < vocab_size) or masked[i] == label


# ── 9. mlm_masking with prob=0.0: no masks applied ─────────────────────────


def test_mlm_masking_prob_zero(sample_tokens, rng):
    masked, labels = mlm_masking(
        sample_tokens,
        mask_token_id=0,
        mask_prob=0.0,
        vocab_size=128,
        keep_special_tokens=True,
        rng=rng,
    )
    assert all(line == -100 for line in labels)
    assert masked == sample_tokens


# ── 10. span_masking: returns (masked, labels) of same length as input ──────


def test_span_masking_same_length(sample_tokens, rng):
    masked, labels = span_masking(
        sample_tokens,
        mask_token_id=0,
        avg_span_length=2.0,
        mask_ratio=0.3,
        keep_special_tokens=True,
        rng=rng,
    )
    assert len(masked) == len(sample_tokens)
    assert len(labels) == len(sample_tokens)


def test_span_masking_labels_semantics(sample_tokens, rng):
    masked, labels = span_masking(
        sample_tokens,
        mask_token_id=0,
        avg_span_length=2.0,
        mask_ratio=0.3,
        keep_special_tokens=True,
        rng=rng,
    )
    for i, (orig, label) in enumerate(zip(sample_tokens, labels)):
        if label != -100:
            assert label == orig
            assert masked[i] == 0  # mask_token_id


# ── 11. token_cutout: output same length as input ───────────────────────────


def test_token_cutout_same_length(sample_tokens, rng):
    result = token_cutout(sample_tokens, cutout_len=3, mask_token_id=0, rng=rng)
    assert len(result) == len(sample_tokens)


# ── 12. token_cutout: contains mask_token_id in cutout region ───────────────


def test_token_cutout_contains_mask(rng):
    tokens = [5] * 20  # none are 0
    mask_id = 0
    result = token_cutout(tokens, cutout_len=5, mask_token_id=mask_id, rng=rng)
    assert mask_id in result
    # Exactly 5 positions should be masked
    assert result.count(mask_id) == 5


# ── 13. mixup_sequences: output length == max(len_a, len_b) ─────────────────


def test_mixup_sequences_length():
    a = [1, 2, 3, 4, 5]
    b = [10, 20, 30]
    result = mixup_sequences(a, b, alpha=0.5)
    assert len(result) == max(len(a), len(b))


def test_mixup_sequences_equal_length():
    a = [1, 2, 3]
    b = [4, 5, 6]
    result = mixup_sequences(a, b, alpha=0.5)
    assert len(result) == 3


def test_mixup_sequences_alpha_one():
    """alpha=1.0 should always pick from a."""
    a = [1, 2, 3, 4]
    b = [10, 20, 30, 40]
    result = mixup_sequences(a, b, alpha=1.0)
    assert result == a


def test_mixup_sequences_alpha_zero():
    """alpha=0.0 should always pick from b."""
    a = [1, 2, 3, 4]
    b = [10, 20, 30, 40]
    result = mixup_sequences(a, b, alpha=0.0)
    assert result == b


# ── 14. TokenAugmentor.augment returns list of ints ─────────────────────────


def test_augmentor_augment_returns_list_of_ints(sample_tokens):
    cfg = AugmentationConfig(
        random_deletion_prob=0.1,
        random_swap_prob=0.1,
        seed=0,
    )
    augmentor = TokenAugmentor(cfg)
    result = augmentor.augment(sample_tokens)
    assert isinstance(result, list)
    assert all(isinstance(t, int) for t in result)


# ── 15. TokenAugmentor.augment_tensor works on 2D (B, T) tensor ─────────────


def test_augmentor_augment_tensor_2d():
    cfg = AugmentationConfig(
        random_deletion_prob=0.0,  # keep length stable for easy shape check
        random_swap_prob=0.1,
        seed=7,
    )
    augmentor = TokenAugmentor(cfg)
    batch = torch.randint(1, 100, (4, 10))
    result = augmentor.augment_tensor(batch)
    assert result.dim() == 2
    assert result.shape[0] == 4  # batch size preserved


def test_augmentor_augment_tensor_1d():
    cfg = AugmentationConfig(random_swap_prob=0.1, seed=7)
    augmentor = TokenAugmentor(cfg)
    tokens = torch.randint(1, 100, (10,))
    result = augmentor.augment_tensor(tokens)
    assert result.dim() == 1


# ── 16. TokenAugmentor.mlm_augment returns tensors with correct label values ─


def test_augmentor_mlm_augment_label_values(sample_tokens):
    cfg = AugmentationConfig(
        mask_token_id=0,
        mask_prob=0.5,
        vocab_size=128,
        seed=99,
    )
    augmentor = TokenAugmentor(cfg)
    masked_input, labels = augmentor.mlm_augment(sample_tokens)

    assert isinstance(masked_input, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert masked_input.shape == labels.shape

    # Unmasked positions must have label -100
    unmasked_mask = labels == -100
    assert torch.all(masked_input[unmasked_mask] == torch.tensor(sample_tokens)[unmasked_mask])

    # Masked positions must have label == original token
    for i in range(len(sample_tokens)):
        if labels[i].item() != -100:
            assert labels[i].item() == sample_tokens[i]
