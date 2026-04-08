"""Tests for flash decoding helpers."""

import pytest
import torch

from src.inference.flash_decoding import append_to_kv_cache, flash_decode_step


def naive_last_token_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    scale = query.size(-1) ** -0.5
    scores = torch.einsum("bhd,bshd->bhs", query * scale, keys)
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bshd->bhd", weights, values)


def test_flash_decode_matches_naive_attention():
    query = torch.randn(2, 4, 16)
    keys = torch.randn(2, 5, 4, 16)
    values = torch.randn(2, 5, 4, 16)
    expected = naive_last_token_attention(query, keys, values)
    actual = flash_decode_step(query, keys, values)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_flash_decode_respects_lengths_mask():
    query = torch.randn(2, 2, 8)
    keys = torch.randn(2, 4, 2, 8)
    values = torch.randn(2, 4, 2, 8)
    lengths = torch.tensor([4, 2])
    actual = flash_decode_step(query, keys, values, lengths=lengths)
    expected_second = naive_last_token_attention(query[1:2], keys[1:2, :2], values[1:2, :2])
    assert torch.allclose(actual[1:2], expected_second, atol=1e-6)


def test_append_to_kv_cache_grows_sequence_length():
    keys = torch.randn(1, 3, 2, 4)
    values = torch.randn(1, 3, 2, 4)
    new_key = torch.randn(1, 2, 4)
    new_value = torch.randn(1, 2, 4)
    next_keys, next_values = append_to_kv_cache(keys, values, new_key, new_value)
    assert next_keys.shape[1] == 4
    assert next_values.shape[1] == 4


def test_flash_decode_uses_custom_scale():
    query = torch.ones(1, 1, 2)
    keys = torch.ones(1, 2, 1, 2)
    values = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    low_scale = flash_decode_step(query, keys, values, scale=0.1)
    high_scale = flash_decode_step(query, keys, values, scale=10.0)
    assert torch.allclose(low_scale, high_scale, atol=1e-6)


def test_flash_decode_rejects_bad_shapes():
    with pytest.raises(ValueError):
        flash_decode_step(torch.randn(2, 3), torch.randn(2, 4, 3, 5), torch.randn(2, 4, 3, 5))


def test_flash_decode_rejects_mismatched_query_shape():
    with pytest.raises(ValueError):
        flash_decode_step(torch.randn(2, 3, 4), torch.randn(2, 5, 2, 4), torch.randn(2, 5, 2, 4))


def test_append_to_kv_cache_rejects_bad_new_tensor_rank():
    with pytest.raises(ValueError):
        append_to_kv_cache(torch.randn(1, 3, 2, 4), torch.randn(1, 3, 2, 4), torch.randn(1, 1, 2, 4), torch.randn(1, 2, 4))
