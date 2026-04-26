"""Tests for src/data/dynamic_batching.py.

All tests use small tensors so they run quickly without a GPU.
"""

from __future__ import annotations

import pytest
import torch

from src.data.dynamic_batching import (
    BatchConfig,
    DynamicBatcher,
    bucket_by_length,
    compute_padding_ratio,
    greedy_pack,
    pad_sequence_batch,
)

# ---------------------------------------------------------------------------
# BatchConfig defaults
# ---------------------------------------------------------------------------


def test_batch_config_defaults():
    cfg = BatchConfig()
    assert cfg.max_tokens_per_batch == 2048
    assert cfg.max_batch_size == 32
    assert cfg.pad_token_id == 0
    assert cfg.sort_by_length is True
    assert cfg.drop_last is False


# ---------------------------------------------------------------------------
# pad_sequence_batch
# ---------------------------------------------------------------------------


def test_pad_sequence_batch_shape():
    seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
    padded, mask = pad_sequence_batch(seqs, pad_value=0)
    assert padded.shape == (3, 3), f"Expected (3,3) got {padded.shape}"
    assert mask.shape == (3, 3)


def test_pad_sequence_batch_attention_mask_correct():
    """Mask should be 1 for real tokens and 0 for padding positions."""
    seqs = [torch.tensor([10, 20, 30]), torch.tensor([5])]
    padded, mask = pad_sequence_batch(seqs, pad_value=0)
    # Row 0: all three positions are real
    assert mask[0].tolist() == [1, 1, 1]
    # Row 1: first position real, rest padding
    assert mask[1].tolist() == [1, 0, 0]


def test_pad_sequence_batch_values():
    """Padded positions should contain pad_value."""
    seqs = [torch.tensor([7, 8, 9]), torch.tensor([42])]
    padded, _ = pad_sequence_batch(seqs, pad_value=99)
    assert padded[1, 1].item() == 99
    assert padded[1, 2].item() == 99


def test_pad_sequence_batch_identical_length_no_padding():
    """When all sequences have the same length, the mask should be all-ones."""
    seqs = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6])]
    padded, mask = pad_sequence_batch(seqs, pad_value=0)
    assert mask.shape == (3, 2)
    assert mask.sum().item() == 6, "No padding expected — all mask values should be 1"


# ---------------------------------------------------------------------------
# compute_padding_ratio
# ---------------------------------------------------------------------------


def test_compute_padding_ratio_all_same_length_is_zero():
    seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    ratio = compute_padding_ratio(seqs)
    assert ratio == 0.0, f"Expected 0.0, got {ratio}"


def test_compute_padding_ratio_one_long_one_short_greater_than_zero():
    seqs = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([9])]
    ratio = compute_padding_ratio(seqs)
    # Padding: (5 - 1) = 4 positions out of 10 total → 0.4
    assert ratio > 0.0
    assert ratio < 1.0


def test_compute_padding_ratio_empty_list():
    assert compute_padding_ratio([]) == 0.0


# ---------------------------------------------------------------------------
# bucket_by_length
# ---------------------------------------------------------------------------


def test_bucket_by_length_returns_n_buckets():
    seqs = [torch.arange(i + 1) for i in range(9)]  # lengths 1..9
    n_buckets = 3
    buckets = bucket_by_length(seqs, n_buckets)
    assert len(buckets) == n_buckets


def test_bucket_by_length_all_sequences_present():
    seqs = [torch.arange(i + 1) for i in range(7)]
    buckets = bucket_by_length(seqs, 3)
    total = sum(len(b) for b in buckets)
    assert total == len(seqs)


def test_bucket_by_length_sorted_within_bucket():
    """Sequences within each bucket should be in ascending length order."""
    seqs = [torch.arange(i + 1) for i in range(6)]
    buckets = bucket_by_length(seqs, 2)
    for bucket in buckets:
        lengths = [s.shape[0] for s in bucket]
        assert lengths == sorted(lengths)


# ---------------------------------------------------------------------------
# greedy_pack
# ---------------------------------------------------------------------------


def test_greedy_pack_each_bin_le_max_tokens():
    seqs = [torch.arange(i + 1) for i in range(10)]  # lengths 1..10
    max_tokens = 15
    bins = greedy_pack(seqs, max_tokens=max_tokens)
    for b in bins:
        total = sum(s.shape[0] for s in b)
        # A single oversized sequence may exceed the budget (no data is dropped),
        # but for lengths 1..10 all fit within 15.
        assert total <= max_tokens, f"Bin has {total} tokens, limit {max_tokens}"


def test_greedy_pack_fewer_bins_than_sequences():
    """Short sequences (length 1) should be packed together into fewer bins."""
    seqs = [torch.tensor([i]) for i in range(20)]  # 20 length-1 sequences
    bins = greedy_pack(seqs, max_tokens=10)
    assert len(bins) < len(seqs), f"Expected fewer bins ({len(bins)}) than sequences ({len(seqs)})"


def test_greedy_pack_no_sequences():
    bins = greedy_pack([], max_tokens=100)
    assert bins == []


def test_greedy_pack_all_sequences_present():
    seqs = [torch.arange(i + 1) for i in range(8)]
    bins = greedy_pack(seqs, max_tokens=20)
    total_in_bins = sum(len(b) for b in bins)
    assert total_in_bins == len(seqs)


# ---------------------------------------------------------------------------
# DynamicBatcher.collate
# ---------------------------------------------------------------------------


def test_collate_returns_correct_keys():
    cfg = BatchConfig(max_tokens_per_batch=128, max_batch_size=8)
    batcher = DynamicBatcher(cfg)
    seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    batch = batcher.collate(seqs)
    assert "input_ids" in batch
    assert "attention_mask" in batch


def test_collate_output_shapes_correct():
    cfg = BatchConfig(max_tokens_per_batch=128, max_batch_size=8)
    batcher = DynamicBatcher(cfg)
    seqs = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6]), torch.tensor([7])]
    batch = batcher.collate(seqs)
    B, T = batch["input_ids"].shape
    assert B == 3
    assert T == 4  # T_max


def test_collate_attention_mask_sums_equal_seq_lengths():
    cfg = BatchConfig(sort_by_length=False)
    batcher = DynamicBatcher(cfg)
    seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
    batch = batcher.collate(seqs)
    mask_sums = batch["attention_mask"].sum(dim=1).tolist()
    expected = [3, 2, 1]
    # sort_by_length=False so order is preserved
    assert mask_sums == expected, f"Expected {expected}, got {mask_sums}"


# ---------------------------------------------------------------------------
# DynamicBatcher.create_batches
# ---------------------------------------------------------------------------


def test_create_batches_token_budget_respected():
    cfg = BatchConfig(max_tokens_per_batch=20, max_batch_size=32)
    batcher = DynamicBatcher(cfg)
    # Lengths: 1,2,3,4,5,6,7 — mixed so many batches
    seqs = [torch.arange(i + 1) for i in range(7)]
    batches = batcher.create_batches(seqs)
    for batch in batches:
        input_ids = batch["input_ids"]
        B, T = input_ids.shape
        total_tokens = B * T
        assert total_tokens <= cfg.max_tokens_per_batch, (
            f"Batch has {total_tokens} tokens, budget {cfg.max_tokens_per_batch}"
        )


def test_create_batches_all_sequences_included():
    cfg = BatchConfig(max_tokens_per_batch=50, max_batch_size=16)
    batcher = DynamicBatcher(cfg)
    seqs = [torch.arange(i + 1) for i in range(10)]
    batches = batcher.create_batches(seqs)
    recovered = sum(int(batch["attention_mask"].sum().item()) for batch in batches)
    expected_real = sum(i + 1 for i in range(10))  # 1+2+...+10 = 55
    assert recovered == expected_real, f"Lost tokens: got {recovered}, expected {expected_real}"


def test_create_batches_empty_input():
    cfg = BatchConfig()
    batcher = DynamicBatcher(cfg)
    assert batcher.create_batches([]) == []


# ---------------------------------------------------------------------------
# DynamicBatcher.compute_efficiency
# ---------------------------------------------------------------------------


def test_compute_efficiency_in_range():
    cfg = BatchConfig(max_tokens_per_batch=30, max_batch_size=8)
    batcher = DynamicBatcher(cfg)
    seqs = [torch.arange(i + 1) for i in range(6)]
    batches = batcher.create_batches(seqs)
    eff = batcher.compute_efficiency(batches)
    assert 0.0 < eff <= 1.0, f"Efficiency {eff} out of range (0, 1]"


def test_compute_efficiency_perfect_when_no_padding():
    """Equal-length sequences produce zero padding → efficiency == 1.0."""
    cfg = BatchConfig(max_tokens_per_batch=100, max_batch_size=4)
    batcher = DynamicBatcher(cfg)
    # 4 sequences all length 5
    seqs = [torch.ones(5, dtype=torch.long) * i for i in range(4)]
    batches = batcher.create_batches(seqs)
    eff = batcher.compute_efficiency(batches)
    assert eff == pytest.approx(1.0)


def test_compute_efficiency_empty_batches():
    cfg = BatchConfig()
    batcher = DynamicBatcher(cfg)
    eff = batcher.compute_efficiency([])
    assert eff == pytest.approx(1.0)
