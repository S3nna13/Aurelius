"""Tests for flash-style prefill helpers."""

import pytest
import torch

from src.inference.flash_prefill import causal_attention_scores, flash_prefill, prefill_chunked


def naive_prefill(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    batch, seq_len, _, head_dim = query.shape
    outputs = []
    scale = head_dim**-0.5
    for index in range(seq_len):
        q = query[:, index : index + 1]
        k = key[:, : index + 1]
        v = value[:, : index + 1]
        scores = torch.einsum("bthd,bshd->bhts", q * scale, k)
        weights = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("bhts,bshd->bthd", weights, v))
    return torch.cat(outputs, dim=1)


def test_causal_attention_scores_shape():
    query = torch.randn(2, 4, 3, 5)
    scores = causal_attention_scores(query, query)
    assert scores.shape == (2, 3, 4, 4)


def test_flash_prefill_matches_naive_attention():
    query = torch.randn(1, 5, 2, 4)
    key = torch.randn(1, 5, 2, 4)
    value = torch.randn(1, 5, 2, 4)
    actual = flash_prefill(query, key, value)
    expected = naive_prefill(query, key, value)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_prefill_chunked_matches_full_prefill():
    query = torch.randn(1, 6, 2, 4)
    key = torch.randn(1, 6, 2, 4)
    value = torch.randn(1, 6, 2, 4)
    full = flash_prefill(query, key, value)
    chunked = prefill_chunked(query, key, value, chunk_size=2)
    assert torch.allclose(full, chunked, atol=1e-6)


def test_causal_attention_scores_mask_future_positions():
    query = torch.randn(1, 3, 1, 2)
    scores = causal_attention_scores(query, query)
    assert torch.isinf(scores[0, 0, 0, 1])


def test_flash_prefill_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        flash_prefill(torch.randn(1, 2, 3, 4), torch.randn(1, 3, 3, 4), torch.randn(1, 3, 3, 4))


def test_prefill_chunked_rejects_bad_chunk_size():
    with pytest.raises(ValueError):
        prefill_chunked(
            torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4), 0
        )


def test_causal_attention_scores_rejects_bad_rank():
    with pytest.raises(ValueError):
        causal_attention_scores(torch.randn(1, 2, 3), torch.randn(1, 2, 3))
