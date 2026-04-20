"""Tests for sliding_window_causal_mask."""

from __future__ import annotations

import pytest
import torch

from src.longcontext.sliding_window_causal_mask import SlidingWindowCausalMaskBuilder


def test_shape_and_causal():
    b = SlidingWindowCausalMaskBuilder(window_size=4)
    m = b.build(10, dtype=torch.float32)
    assert m.shape == (1, 1, 10, 10)
    assert m[0, 0, 0, 0] == 0.0
    assert m[0, 0, 0, 1] < 0  # future masked
    assert not torch.isnan(m).any()


def test_future_masked():
    b = SlidingWindowCausalMaskBuilder(window_size=8, neg_value=-1e4)
    m = b.build(6, dtype=torch.float32)[0, 0]
    assert m[0, 1] < 0  # cannot attend to future


def test_window_excludes_far_past():
    b = SlidingWindowCausalMaskBuilder(window_size=2, neg_value=-1e9)
    m = b.build(6, dtype=torch.float32)[0, 0]
    assert m[3, 0] < 0  # distance 3 >= window 2


def test_full_band_when_window_ge_seq():
    b = SlidingWindowCausalMaskBuilder(window_size=100)
    m = b.build(8, dtype=torch.float32)[0, 0]
    for i in range(8):
        for j in range(i + 1):
            assert m[i, j] == 0.0
        for j in range(i + 1, 8):
            assert m[i, j] < 0


def test_invalid_window():
    with pytest.raises(ValueError):
        SlidingWindowCausalMaskBuilder(window_size=0)


def test_invalid_seq_len():
    with pytest.raises(ValueError):
        SlidingWindowCausalMaskBuilder(2).build(0)


def test_determinism():
    torch.manual_seed(0)
    a = SlidingWindowCausalMaskBuilder(5).build(32)
    torch.manual_seed(0)
    b = SlidingWindowCausalMaskBuilder(5).build(32)
    assert torch.equal(a, b)


def test_seq_len_one():
    m = SlidingWindowCausalMaskBuilder(1).build(1)
    assert m.shape == (1, 1, 1, 1)
    assert m[0, 0, 0, 0] == 0.0


@pytest.mark.parametrize("seq_len", [64, 512])
def test_longer_sequences(seq_len: int):
    b = SlidingWindowCausalMaskBuilder(window_size=32, neg_value=-1e4)
    m = b.build(seq_len, dtype=torch.float32)
    assert m.shape == (1, 1, seq_len, seq_len)
    diag = m[0, 0, torch.arange(seq_len), torch.arange(seq_len)]
    assert torch.allclose(diag, torch.zeros(seq_len))


def test_softmax_stable_with_mask():
    b = SlidingWindowCausalMaskBuilder(window_size=4)
    m = b.build(12, dtype=torch.float32)
    scores = torch.randn(1, 1, 12, 12)
    w = torch.softmax(scores + m, dim=-1)
    assert torch.isfinite(w).all()
    assert (w.sum(-1) - 1).abs().max() < 1e-3
