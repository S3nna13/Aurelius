"""Tests for batch inference optimizations — DynamicPadder, BucketBatcher,
ContinuousBatcher, ThroughputBenchmark, PrefixCacheManager."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.inference.batch_inference import (
    BucketBatcher,
    ContinuousBatcher,
    DynamicPadder,
    PrefixCacheManager,
    ThroughputBenchmark,
)

# ---------------------------------------------------------------------------
# Tiny model fixture: nn.Embedding(16, 16) -> nn.Linear(16, 16)
# Accepts (B, T) long tensor, returns (B, T, 16) logits.
# ---------------------------------------------------------------------------

VOCAB = 16
D_MODEL = 16


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.proj = nn.Linear(D_MODEL, VOCAB)

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: (B, T)  ->  (B, T, VOCAB)
        x = self.emb(input_ids)
        return self.proj(x)


@pytest.fixture()
def model() -> TinyModel:
    torch.manual_seed(0)
    m = TinyModel()
    m.train(False)
    return m


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_seq(length: int, vocab: int = VOCAB) -> Tensor:
    return torch.randint(1, vocab, (length,))


# ===========================================================================
# DynamicPadder tests
# ===========================================================================


def test_dynamic_padder_pad_batch_shape():
    """pad_batch output has correct (B, max_len) shape and mask shape."""
    padder = DynamicPadder(pad_token_id=0)
    seqs = [make_seq(3), make_seq(5), make_seq(2)]
    padded, mask = padder.pad_batch(seqs)
    assert padded.shape == (3, 5)
    assert mask.shape == (3, 5)


def test_dynamic_padder_attention_mask_correct():
    """Attention mask is True for real tokens and False for padding (right-pad)."""
    padder = DynamicPadder(pad_token_id=0, padding_side="right")
    seqs = [make_seq(3), make_seq(5)]
    padded, mask = padder.pad_batch(seqs)
    # Row 0 has 3 real tokens
    assert mask[0, :3].all()
    assert not mask[0, 3:].any()
    # Row 1 is fully real
    assert mask[1].all()


def test_dynamic_padder_right_vs_left_positions():
    """Right-padding places real tokens at the start; left-padding at the end."""
    seq_short = torch.tensor([1, 2, 3])
    seq_long = torch.tensor([4, 5, 6, 7, 8])

    right_padder = DynamicPadder(padding_side="right")
    left_padder = DynamicPadder(padding_side="left")

    right_pad, right_mask = right_padder.pad_batch([seq_short, seq_long])
    left_pad, left_mask = left_padder.pad_batch([seq_short, seq_long])

    # Right-padded: real tokens at positions 0,1,2; zeros at 3,4
    assert right_pad[0, 0].item() == 1
    assert right_pad[0, 4].item() == 0

    # Left-padded: real tokens at positions 2,3,4; zeros at 0,1
    assert left_pad[0, 2].item() == 1
    assert left_pad[0, 4].item() == 3
    assert left_pad[0, 0].item() == 0


def test_dynamic_padder_unpad_lengths():
    """unpad recovers tensors whose lengths match the original sequences."""
    padder = DynamicPadder(pad_token_id=0)
    seqs = [make_seq(2), make_seq(6), make_seq(4)]
    padded, mask = padder.pad_batch(seqs)
    recovered = padder.unpad(padded, mask)
    assert len(recovered) == len(seqs)
    for orig, rec in zip(seqs, recovered):
        assert rec.shape == orig.shape
        assert torch.equal(rec, orig)


def test_dynamic_padder_overhead_range():
    """padding_overhead is in [0, 1) and is 0 for equal-length sequences."""
    padder = DynamicPadder()
    equal_seqs = [make_seq(4), make_seq(4), make_seq(4)]
    overhead_equal = padder.padding_overhead(equal_seqs)
    assert overhead_equal == 0.0

    unequal_seqs = [make_seq(2), make_seq(8)]
    overhead_unequal = padder.padding_overhead(unequal_seqs)
    assert 0.0 <= overhead_unequal < 1.0


# ===========================================================================
# BucketBatcher tests
# ===========================================================================


def test_bucket_batcher_correct_bucket_assignment():
    """Sequences go to the smallest bucket whose boundary >= their length."""
    batcher = BucketBatcher(bucket_boundaries=[4, 8], batch_size=2)
    short = make_seq(3)  # -> bucket 0 (boundary 4)
    medium = make_seq(5)  # -> bucket 1 (boundary 8)
    long_ = make_seq(9)  # -> overflow bucket

    batcher.add(short)
    batcher.add(medium)
    batcher.add(long_)

    # Bucket 0 has 1, bucket 1 has 1, overflow has 1
    assert batcher.n_waiting() == 3

    # Add another short to fill bucket 0
    batcher.add(make_seq(4))  # boundary exactly 4 -> bucket 0
    # bucket 0 now has 2 items (batch_size=2), should return a batch
    result = batcher.get_batch()
    assert result is not None
    padded, mask, metas = result
    assert padded.shape[0] == 2  # batch_size


def test_bucket_batcher_get_batch_none_when_empty():
    """get_batch returns None when no bucket has enough sequences."""
    batcher = BucketBatcher(bucket_boundaries=[4, 8], batch_size=4)
    batcher.add(make_seq(3))
    batcher.add(make_seq(5))
    result = batcher.get_batch()
    assert result is None


def test_bucket_batcher_n_waiting_decreases_after_get_batch():
    """n_waiting decreases by batch_size after a successful get_batch."""
    batch_size = 2
    batcher = BucketBatcher(bucket_boundaries=[8], batch_size=batch_size)
    for _ in range(4):
        batcher.add(make_seq(5))

    before = batcher.n_waiting()
    assert before == 4

    result = batcher.get_batch()
    assert result is not None
    after = batcher.n_waiting()
    assert after == before - batch_size


# ===========================================================================
# ContinuousBatcher tests
# ===========================================================================


def test_continuous_batcher_add_request_increments_ids(model: TinyModel):
    """add_request returns incrementing integer request ids starting from 0."""
    batcher = ContinuousBatcher(max_batch_size=4, max_seq_len=16)
    ids = [batcher.add_request(make_seq(4)) for _ in range(4)]
    assert ids == [0, 1, 2, 3]


def test_continuous_batcher_step_returns_token_map(model: TinyModel):
    """step returns a dict mapping every active request_id to a next token."""
    batcher = ContinuousBatcher(max_batch_size=4, max_seq_len=16)
    rid0 = batcher.add_request(torch.tensor([1, 2, 3]), max_new_tokens=5)
    rid1 = batcher.add_request(torch.tensor([4, 5]), max_new_tokens=5)

    result = batcher.step(model)
    assert isinstance(result, dict)
    assert rid0 in result or rid1 in result  # at least one active
    for tok in result.values():
        assert isinstance(tok, int)


def test_continuous_batcher_active_count_decreases(model: TinyModel):
    """active_count decreases as requests exhaust their max_new_tokens."""
    batcher = ContinuousBatcher(max_batch_size=4, max_seq_len=16)
    batcher.add_request(torch.tensor([1, 2]), max_new_tokens=1)
    batcher.add_request(torch.tensor([3, 4]), max_new_tokens=1)

    assert batcher.active_count() == 2
    batcher.step(model)
    # After one step with max_new_tokens=1, both requests should be done
    assert batcher.active_count() == 0


# ===========================================================================
# ThroughputBenchmark tests
# ===========================================================================


def test_throughput_batch_efficiency_in_range():
    """batch_efficiency is in (0, 1] for valid inputs."""
    bench = ThroughputBenchmark()
    eff = bench.batch_efficiency(padded_tokens=10, real_tokens=7)
    assert 0.0 < eff <= 1.0


def test_throughput_tokens_per_second_positive_finite(model: TinyModel):
    """tokens_per_second returns a positive finite float."""
    bench = ThroughputBenchmark()
    input_ids = torch.randint(1, VOCAB, (1, 4))
    tps = bench.tokens_per_second(model, input_ids, n_tokens=5)
    assert isinstance(tps, float)
    assert tps > 0.0
    assert math.isfinite(tps)


# ===========================================================================
# PrefixCacheManager tests
# ===========================================================================


def test_prefix_cache_compute_prefix_kv_not_none(model: TinyModel):
    """compute_prefix_kv returns non-None (keys, values) tensors."""
    manager = PrefixCacheManager(model, max_cache_entries=4)
    prefix = torch.tensor([1, 2, 3, 4])
    keys, values = manager.compute_prefix_kv(prefix)
    assert keys is not None
    assert values is not None
    assert isinstance(keys, Tensor)
    assert isinstance(values, Tensor)


def test_prefix_cache_lookup_hit_on_repeated_prefix(model: TinyModel):
    """lookup returns a result after compute_prefix_kv stores it."""
    manager = PrefixCacheManager(model, max_cache_entries=4)
    prefix = torch.tensor([5, 6, 7])
    manager.compute_prefix_kv(prefix)

    result = manager.lookup(prefix)
    assert result is not None
    keys, values = result
    assert isinstance(keys, Tensor)
    assert isinstance(values, Tensor)


def test_prefix_cache_lookup_miss_on_new_prefix(model: TinyModel):
    """lookup returns None for a prefix that was never stored."""
    manager = PrefixCacheManager(model, max_cache_entries=4)
    seen = torch.tensor([1, 2, 3])
    unseen = torch.tensor([9, 8, 7])
    manager.compute_prefix_kv(seen)

    result = manager.lookup(unseen)
    assert result is None


def test_prefix_cache_hit_rate_increases_with_repeats(model: TinyModel):
    """cache_hit_rate increases when the same prefix is looked up repeatedly."""
    manager = PrefixCacheManager(model, max_cache_entries=4)
    prefix = torch.tensor([3, 1, 4, 1, 5])
    manager.compute_prefix_kv(prefix)

    # Miss on unknown prefix
    manager.lookup(torch.tensor([9, 9, 9]))
    rate_after_miss = manager.cache_hit_rate()

    # Hit on stored prefix
    manager.lookup(prefix)
    rate_after_hit = manager.cache_hit_rate()

    assert rate_after_hit > rate_after_miss
