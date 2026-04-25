"""Tests for src/longcontext/sliding_window_attention.py (8+ tests)."""

from __future__ import annotations

import math

import pytest
import torch

from src.longcontext.sliding_window_attention import (
    SlidingWindowAttention,
    SWAConfig,
)

# Tiny tensor dimensions for CPU-only tests
B, H, S, D = 1, 2, 16, 8


def _make_qkv(seq: int = S) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    q = torch.randn(B, H, seq, D)
    k = torch.randn(B, H, seq, D)
    v = torch.randn(B, H, seq, D)
    return q, k, v


# ---------------------------------------------------------------------------
# 1. Output shape matches input
# ---------------------------------------------------------------------------
def test_output_shape():
    cfg = SWAConfig(window_size=4, global_tokens=2, causal=True)
    swa = SlidingWindowAttention(cfg)
    q, k, v = _make_qkv()
    out = swa(q, k, v)
    assert out.shape == (B, H, S, D)


# ---------------------------------------------------------------------------
# 2. Default config runs without error
# ---------------------------------------------------------------------------
def test_default_config():
    swa = SlidingWindowAttention()
    q, k, v = _make_qkv()
    out = swa(q, k, v)
    assert out.shape == (B, H, S, D)


# ---------------------------------------------------------------------------
# 3. Non-causal mode runs
# ---------------------------------------------------------------------------
def test_non_causal():
    cfg = SWAConfig(window_size=4, global_tokens=2, causal=False)
    swa = SlidingWindowAttention(cfg)
    q, k, v = _make_qkv()
    out = swa(q, k, v)
    assert out.shape == (B, H, S, D)


# ---------------------------------------------------------------------------
# 4. get_effective_context: exact multiple
# ---------------------------------------------------------------------------
def test_get_effective_context_exact():
    cfg = SWAConfig(window_size=512)
    swa = SlidingWindowAttention(cfg)
    assert swa.get_effective_context(512) == 512
    assert swa.get_effective_context(1024) == 1024


# ---------------------------------------------------------------------------
# 5. get_effective_context: rounds up
# ---------------------------------------------------------------------------
def test_get_effective_context_rounds_up():
    cfg = SWAConfig(window_size=512)
    swa = SlidingWindowAttention(cfg)
    assert swa.get_effective_context(513) == 1024
    assert swa.get_effective_context(1) == 512


# ---------------------------------------------------------------------------
# 6. All-global sequence (seq <= global_tokens) produces output
# ---------------------------------------------------------------------------
def test_all_global_tokens():
    cfg = SWAConfig(window_size=4, global_tokens=16, causal=True)
    swa = SlidingWindowAttention(cfg)
    q, k, v = _make_qkv()
    out = swa(q, k, v)
    assert out.shape == (B, H, S, D)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# 7. Output is finite (no NaN or Inf)
# ---------------------------------------------------------------------------
def test_output_finite():
    cfg = SWAConfig(window_size=4, global_tokens=2, causal=True)
    swa = SlidingWindowAttention(cfg)
    q, k, v = _make_qkv()
    out = swa(q, k, v)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# 8. window_size=1 (attend only to self in causal mode)
# ---------------------------------------------------------------------------
def test_window_size_one():
    cfg = SWAConfig(window_size=1, global_tokens=0, causal=True)
    swa = SlidingWindowAttention(cfg)
    q, k, v = _make_qkv()
    out = swa(q, k, v)
    assert out.shape == (B, H, S, D)


# ---------------------------------------------------------------------------
# 9. Longer sequence works
# ---------------------------------------------------------------------------
def test_longer_sequence():
    cfg = SWAConfig(window_size=8, global_tokens=4, causal=True)
    swa = SlidingWindowAttention(cfg)
    q, k, v = _make_qkv(seq=32)
    out = swa(q, k, v)
    assert out.shape == (B, H, 32, D)
    assert not torch.isnan(out).any()
