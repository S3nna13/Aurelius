"""Tests for TokenizedShardDataset."""

import numpy as np
import pytest
import torch

from src.data.tokenized_loader import TokenizedShardDataset


@pytest.fixture
def tmp_shards(tmp_path):
    """Create two small .npy shard files for testing."""
    shard1 = np.arange(100, dtype=np.uint16)
    shard2 = np.arange(100, 200, dtype=np.uint16)
    p1 = tmp_path / "shard_00.npy"
    p2 = tmp_path / "shard_01.npy"
    np.save(p1, shard1)
    np.save(p2, shard2)
    return [p1, p2]


def test_dataset_length(tmp_shards):
    ds = TokenizedShardDataset(tmp_shards, seq_len=10)
    # Each shard has 100 tokens. With seq_len=10, stride=10:
    # windows per shard = (100 - 10 - 1) // 10 + 1 = 9 (indices 0..89 start positions)
    # Actually: need seq_len+1 = 11 tokens per window
    # valid starts: 0, 10, 20, ..., 89 -> need start+11 <= 100 -> start <= 89 -> 9 windows
    # But stride=10, so: floor((100 - 11) / 10) + 1 = floor(89/10)+1 = 8+1 = 9 windows per shard
    assert len(ds) == 18  # 9 per shard × 2


def test_item_shapes(tmp_shards):
    ds = TokenizedShardDataset(tmp_shards, seq_len=10)
    input_ids, labels = ds[0]
    assert input_ids.shape == (10,)
    assert labels.shape == (10,)
    assert input_ids.dtype == torch.int64
    assert labels.dtype == torch.int64


def test_labels_are_shifted(tmp_shards):
    ds = TokenizedShardDataset(tmp_shards, seq_len=10)
    input_ids, labels = ds[0]
    # labels[i] should be input_ids[i+1] for i < seq_len-1
    assert torch.all(labels[:-1] == input_ids[1:])


def test_custom_stride(tmp_shards):
    ds = TokenizedShardDataset(tmp_shards, seq_len=10, stride=5)
    # stride=5: floor((100 - 11) / 5) + 1 = floor(89/5)+1 = 17+1 = 18 windows per shard
    assert len(ds) == 36  # 18 per shard × 2


def test_cross_shard_access(tmp_shards):
    ds = TokenizedShardDataset(tmp_shards, seq_len=10)
    # Windows from shard 0 come first, then shard 1
    _, _ = ds[0]  # shard 0
    _, _ = ds[9]  # should be shard 1 (first window of shard 1)


def test_returns_tensors(tmp_shards):
    ds = TokenizedShardDataset(tmp_shards, seq_len=10)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
