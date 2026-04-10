"""Tests for src/data/augmentation.py — covers both the original class-based
transforms and the new functional API."""

import pytest
import torch

from src.data.augmentation import (
    # Original transforms
    RandomTokenMask,
    TokenDropout,
    SpanCorruption,
    # New functional API
    AugmentationConfig,
    token_masking,
    token_replacement,
    token_deletion,
    token_insertion,
    adjacent_swap,
    span_masking,
    TokenAugmenter,
    AugmentedDataset,
)


# ---------------------------------------------------------------------------
# Original class-based transform tests (9 tests)
# ---------------------------------------------------------------------------


def test_random_mask_shape_preserved():
    mask = RandomTokenMask(p=0.15, mask_id=1, vocab_size=256)
    x = torch.randint(2, 256, (20,))
    out = mask(x)
    assert out.shape == x.shape


def test_random_mask_uses_mask_id():
    mask = RandomTokenMask(p=1.0, mask_id=99, vocab_size=256)
    x = torch.zeros(10, dtype=torch.long)
    out = mask(x)
    assert (out == 99).all()


def test_random_mask_does_not_modify_input():
    mask = RandomTokenMask(p=0.5, mask_id=0, vocab_size=256)
    x = torch.randint(1, 256, (20,))
    original = x.clone()
    mask(x)
    assert torch.equal(x, original)


def test_token_dropout_shortens():
    drop = TokenDropout(p=0.5)
    torch.manual_seed(42)
    x = torch.randint(0, 100, (50,))
    out = drop(x)
    assert len(out) < len(x)
    assert len(out) >= 1


def test_token_dropout_p0_unchanged():
    drop = TokenDropout(p=0.0)
    x = torch.randint(0, 100, (20,))
    out = drop(x)
    assert torch.equal(out, x)


def test_token_dropout_p1_keeps_one():
    drop = TokenDropout(p=1.0)
    x = torch.randint(0, 100, (10,))
    out = drop(x)
    assert len(out) == 1


def test_span_corruption_shortens():
    sc = SpanCorruption(p=0.15, mean_span_length=3, sentinel_start=32000)
    torch.manual_seed(0)
    x = torch.randint(0, 1000, (100,))
    out = sc(x)
    assert len(out) < len(x)


def test_span_corruption_contains_sentinels():
    sc = SpanCorruption(p=0.5, mean_span_length=3, sentinel_start=32000)
    torch.manual_seed(1)
    x = torch.randint(0, 1000, (50,))
    out = sc(x)
    assert (out >= 32000).any()


def test_span_corruption_empty_safe():
    sc = SpanCorruption(p=0.15, sentinel_start=32000)
    x = torch.tensor([5], dtype=torch.long)
    out = sc(x)
    assert len(out) >= 1


# ---------------------------------------------------------------------------
# New functional API tests (17 tests)
# ---------------------------------------------------------------------------


# 1. AugmentationConfig defaults
def test_augmentation_config_defaults():
    cfg = AugmentationConfig()
    assert cfg.mask_prob == 0.15
    assert cfg.replace_prob == 0.1
    assert cfg.delete_prob == 0.05
    assert cfg.insert_prob == 0.05
    assert cfg.swap_prob == 0.05
    assert cfg.vocab_size == 256
    assert cfg.mask_token_id == 1
    assert cfg.seed is None


# 2. token_masking output shape matches input
def test_token_masking_shape():
    x = torch.randint(2, 256, (4, 32))
    masked, mask = token_masking(x, mask_prob=0.15, mask_token_id=1, vocab_size=256)
    assert masked.shape == x.shape


# 3. token_masking mask is boolean and has correct density
def test_token_masking_mask_dtype_and_density():
    torch.manual_seed(0)
    x = torch.randint(2, 256, (1, 1000))
    _, mask = token_masking(x, mask_prob=0.15, mask_token_id=1, vocab_size=256)
    assert mask.dtype == torch.bool
    density = mask.float().mean().item()
    # allow ±5% tolerance around 15%
    assert 0.10 <= density <= 0.20, f"density={density}"


# 4. token_masking masked positions contain mask_token_id
def test_token_masking_mask_token_id():
    x = torch.randint(2, 256, (2, 50))
    masked, mask = token_masking(x, mask_prob=0.5, mask_token_id=99, vocab_size=256)
    assert (masked[mask] == 99).all()


# 5. token_replacement output shape matches input
def test_token_replacement_shape():
    x = torch.randint(0, 256, (3, 20))
    out = token_replacement(x, replace_prob=0.1, vocab_size=256)
    assert out.shape == x.shape


# 6. token_replacement changes some tokens but not all (at high prob)
def test_token_replacement_changes_some_not_all():
    torch.manual_seed(1)
    x = torch.ones(1, 200, dtype=torch.long) * 128
    out = token_replacement(x, replace_prob=0.5, vocab_size=256)
    changed = (out != x).sum().item()
    assert 0 < changed < 200


# 7. token_deletion output length <= input length
def test_token_deletion_shorter():
    x = torch.randint(0, 256, (1, 50))
    out = token_deletion(x, delete_prob=0.3)
    assert out.shape[1] <= 50
    assert out.shape[0] == 1


# 8. token_deletion with delete_prob=0 returns unchanged sequence
def test_token_deletion_zero_prob():
    x = torch.randint(0, 256, (1, 30))
    out = token_deletion(x, delete_prob=0.0)
    assert torch.equal(out, x)


# 9. token_insertion output length >= input length
def test_token_insertion_longer():
    torch.manual_seed(2)
    x = torch.randint(0, 256, (1, 40))
    out = token_insertion(x, insert_prob=0.3, vocab_size=256)
    assert out.shape[1] >= 40
    assert out.shape[0] == 1


# 10. adjacent_swap output shape matches input
def test_adjacent_swap_shape():
    x = torch.randint(0, 256, (4, 20))
    out = adjacent_swap(x, swap_prob=0.3)
    assert out.shape == x.shape


# 11. adjacent_swap is a permutation (same multiset of tokens)
def test_adjacent_swap_same_multiset():
    torch.manual_seed(3)
    x = torch.randint(0, 256, (1, 30))
    out = adjacent_swap(x, swap_prob=0.5)
    # same sorted values
    assert torch.equal(x.sort().values, out.sort().values)


# 12. span_masking output shape matches input
def test_span_masking_shape():
    x = torch.randint(2, 256, (2, 40))
    masked, mask = span_masking(x, span_mask_prob=0.15, mean_span_len=3,
                                 mask_token_id=1)
    assert masked.shape == x.shape
    assert mask.shape == x.shape


# 13. span_masking masked positions contain mask_token_id
def test_span_masking_mask_token_id():
    x = torch.randint(2, 256, (1, 60))
    masked, mask = span_masking(x, span_mask_prob=0.3, mean_span_len=3,
                                 mask_token_id=99)
    assert (masked[mask] == 99).all()


# 14. TokenAugmenter.augment returns Tensor
def test_token_augmenter_augment_returns_tensor():
    cfg = AugmentationConfig(mask_prob=0.15, replace_prob=0.1,
                              delete_prob=0.0, insert_prob=0.0,
                              swap_prob=0.1, vocab_size=256, seed=42)
    aug = TokenAugmenter(cfg)
    x = torch.randint(2, 256, (1, 32))
    out = aug.augment(x)
    assert isinstance(out, torch.Tensor)


# 15. TokenAugmenter.augment_with_labels returns (Tensor, Tensor)
def test_token_augmenter_augment_with_labels():
    cfg = AugmentationConfig(mask_prob=0.15, replace_prob=0.0,
                              delete_prob=0.0, insert_prob=0.0,
                              swap_prob=0.0, vocab_size=256, seed=7)
    aug = TokenAugmenter(cfg)
    x = torch.randint(2, 256, (1, 20))
    augmented, original = aug.augment_with_labels(x)
    assert isinstance(augmented, torch.Tensor)
    assert isinstance(original, torch.Tensor)
    assert torch.equal(original, x)


# 16. AugmentedDataset len matches input list
def test_augmented_dataset_len():
    seqs = [torch.randint(2, 256, (20,)) for _ in range(10)]
    cfg = AugmentationConfig(mask_prob=0.15, replace_prob=0.0,
                              delete_prob=0.0, insert_prob=0.0, swap_prob=0.0)
    aug = TokenAugmenter(cfg)
    ds = AugmentedDataset(seqs, aug)
    assert len(ds) == 10


# 17. AugmentedDataset.get_batch returns correctly padded batch
def test_augmented_dataset_get_batch_padding():
    seqs = [
        torch.randint(2, 256, (10,)),
        torch.randint(2, 256, (20,)),
        torch.randint(2, 256, (15,)),
    ]
    cfg = AugmentationConfig(mask_prob=0.0, replace_prob=0.0,
                              delete_prob=0.0, insert_prob=0.0, swap_prob=0.0)
    aug = TokenAugmenter(cfg)
    ds = AugmentedDataset(seqs, aug)
    aug_batch, ori_batch = ds.get_batch([0, 1, 2])

    # originals: padded to length 20 (longest)
    assert ori_batch.shape == (3, 20)
    # augmented: no augmentation applied so also length 20
    assert aug_batch.shape[0] == 3
    # shorter sequences should have trailing zeros (padding)
    assert int(ori_batch[0, 10:].sum().item()) == 0
