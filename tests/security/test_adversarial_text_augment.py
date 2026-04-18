"""Tests for adversarial_text_augment.py."""

from __future__ import annotations

import torch
import pytest

from src.security.adversarial_text_augment import AdversarialAugmenter, AugConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
SEQ_LEN = 20


@pytest.fixture()
def config() -> AugConfig:
    return AugConfig(
        p_swap=0.5,
        p_delete=0.3,
        p_insert=0.3,
        p_replace=0.5,
        vocab_size=VOCAB_SIZE,
        seed=42,
    )


@pytest.fixture()
def augmenter(config: AugConfig) -> AdversarialAugmenter:
    return AdversarialAugmenter(config)


@pytest.fixture()
def ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, VOCAB_SIZE, (SEQ_LEN,), dtype=torch.long)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


# 1. AdversarialAugmenter instantiates
def test_instantiation(augmenter: AdversarialAugmenter) -> None:
    assert isinstance(augmenter, AdversarialAugmenter)


# 2. random_swap returns 1-D LongTensor of same length
def test_random_swap_shape_and_dtype(augmenter: AdversarialAugmenter, ids: torch.Tensor) -> None:
    result = augmenter.random_swap(ids)
    assert result.ndim == 1
    assert result.dtype == torch.long
    assert result.shape[0] == ids.shape[0]


# 3. random_swap values are all in [0, vocab_size)
def test_random_swap_value_range(augmenter: AdversarialAugmenter, ids: torch.Tensor) -> None:
    result = augmenter.random_swap(ids)
    assert result.min().item() >= 0
    assert result.max().item() < VOCAB_SIZE


# 4. random_delete returns 1-D LongTensor with length <= original
def test_random_delete_length(augmenter: AdversarialAugmenter, ids: torch.Tensor) -> None:
    result = augmenter.random_delete(ids)
    assert result.ndim == 1
    assert result.dtype == torch.long
    assert result.shape[0] <= ids.shape[0]


# 5. random_delete with p_delete=0 returns identical sequence
def test_random_delete_zero_prob(ids: torch.Tensor) -> None:
    cfg = AugConfig(p_delete=0.0, vocab_size=VOCAB_SIZE, seed=42)
    aug = AdversarialAugmenter(cfg)
    result = aug.random_delete(ids)
    assert torch.equal(result, ids)


# 6. random_insert returns 1-D LongTensor with length >= original
def test_random_insert_length(augmenter: AdversarialAugmenter, ids: torch.Tensor) -> None:
    result = augmenter.random_insert(ids)
    assert result.ndim == 1
    assert result.dtype == torch.long
    assert result.shape[0] >= ids.shape[0]


# 7. random_insert with p_insert=0 returns identical sequence
def test_random_insert_zero_prob(ids: torch.Tensor) -> None:
    cfg = AugConfig(p_insert=0.0, vocab_size=VOCAB_SIZE, seed=42)
    aug = AdversarialAugmenter(cfg)
    result = aug.random_insert(ids)
    assert torch.equal(result, ids)


# 8. random_replace returns same length as input
def test_random_replace_shape(augmenter: AdversarialAugmenter, ids: torch.Tensor) -> None:
    result = augmenter.random_replace(ids)
    assert result.ndim == 1
    assert result.dtype == torch.long
    assert result.shape[0] == ids.shape[0]


# 9. random_replace with p_replace=0 returns identical sequence
def test_random_replace_zero_prob(ids: torch.Tensor) -> None:
    cfg = AugConfig(p_replace=0.0, vocab_size=VOCAB_SIZE, seed=42)
    aug = AdversarialAugmenter(cfg)
    result = aug.random_replace(ids)
    assert torch.equal(result, ids)


# 10. augment returns 1-D LongTensor
def test_augment_returns_1d_long(augmenter: AdversarialAugmenter, ids: torch.Tensor) -> None:
    result = augmenter.augment(ids)
    assert result.ndim == 1
    assert result.dtype == torch.long


# 11. augment_batch returns 2-D Tensor of shape (B*n_augments, T_max)
def test_augment_batch_shape(augmenter: AdversarialAugmenter) -> None:
    torch.manual_seed(1)
    batch = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN), dtype=torch.long)
    n_augments = 3
    result = augmenter.augment_batch(batch, n_augments=n_augments)
    assert result.ndim == 2
    assert result.shape[0] == 4 * n_augments


# 12. All tokens in batch output are in [0, vocab_size)
def test_augment_batch_value_range(augmenter: AdversarialAugmenter) -> None:
    torch.manual_seed(2)
    batch = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN), dtype=torch.long)
    result = augmenter.augment_batch(batch, n_augments=2)
    assert result.min().item() >= 0
    assert result.max().item() < VOCAB_SIZE


# 13. augment with empty operations tuple returns original
def test_augment_empty_operations(augmenter: AdversarialAugmenter, ids: torch.Tensor) -> None:
    result = augmenter.augment(ids, operations=())
    assert torch.equal(result, ids)
