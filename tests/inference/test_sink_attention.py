"""Tests for sink attention utilities."""

import pytest
import torch

from src.inference.sink_attention import apply_sink_attention, sink_attention_mask, sink_attention_weights


def test_sink_attention_mask_shape():
    mask = sink_attention_mask(seq_len=6, sink_tokens=2, window_size=3)
    assert mask.shape == (6, 6)


def test_sink_attention_mask_keeps_sink_tokens_globally():
    mask = sink_attention_mask(seq_len=6, sink_tokens=2, window_size=1)
    assert mask[5, 0].item()
    assert mask[5, 1].item()


def test_sink_attention_mask_keeps_recent_window():
    mask = sink_attention_mask(seq_len=6, sink_tokens=1, window_size=2)
    assert mask[5, 4].item()
    assert mask[5, 5].item()
    assert not mask[5, 2].item()


def test_apply_sink_attention_masks_disallowed_positions():
    scores = torch.zeros(1, 1, 4, 4)
    masked = apply_sink_attention(scores, sink_tokens=1, window_size=1)
    assert torch.isinf(masked[0, 0, 3, 1])


def test_sink_attention_weights_sum_to_one():
    scores = torch.randn(1, 2, 4, 4)
    weights = sink_attention_weights(scores, sink_tokens=1, window_size=2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-6)


def test_sink_attention_mask_rejects_negative_args():
    with pytest.raises(ValueError):
        sink_attention_mask(-1, 1, 1)


def test_apply_sink_attention_rejects_bad_rank():
    with pytest.raises(ValueError):
        apply_sink_attention(torch.randn(2, 3, 4), sink_tokens=1, window_size=2)

