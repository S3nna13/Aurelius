"""Tests for sink-token cache utilities."""

import pytest
import torch

from src.inference.sink_cache import SinkCache, compress_kv_cache, sink_window_indices


def test_sink_window_indices_keep_prefix_and_tail():
    indices = sink_window_indices(total_tokens=10, sink_tokens=2, window_size=3)
    assert indices == [0, 1, 7, 8, 9]


def test_sink_window_indices_handles_short_sequence():
    indices = sink_window_indices(total_tokens=3, sink_tokens=4, window_size=2)
    assert indices == [0, 1, 2]


def test_compress_kv_cache_reduces_sequence_length():
    keys = torch.randn(1, 10, 2, 4)
    values = torch.randn(1, 10, 2, 4)
    compressed_k, compressed_v, indices = compress_kv_cache(keys, values, sink_tokens=2, window_size=3)
    assert compressed_k.shape[1] == len(indices)
    assert compressed_v.shape[1] == len(indices)


def test_sink_cache_append_tracks_current_length():
    cache = SinkCache(sink_tokens=2, window_size=3)
    for _ in range(6):
        cache.append(torch.randn(1, 2, 4), torch.randn(1, 2, 4))
    assert cache.current_length() == 5


def test_sink_cache_preserves_sink_positions():
    cache = SinkCache(sink_tokens=2, window_size=2)
    keys = [torch.full((1, 1, 1), float(index)) for index in range(5)]
    for key in keys:
        cache.append(key, key)
    assert torch.allclose(cache.key_cache[:, 0], torch.tensor([[[0.0]]]))
    assert torch.allclose(cache.key_cache[:, 1], torch.tensor([[[1.0]]]))


def test_compress_kv_cache_rejects_bad_shapes():
    with pytest.raises(ValueError):
        compress_kv_cache(torch.randn(1, 2, 3), torch.randn(1, 2, 3), sink_tokens=1, window_size=1)


def test_sink_window_indices_reject_negative_args():
    with pytest.raises(ValueError):
        sink_window_indices(-1, 1, 1)
