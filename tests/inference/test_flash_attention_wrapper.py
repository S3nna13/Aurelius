"""Tests for FlashAttentionWrapper and FlashAttnConfig."""

from __future__ import annotations

import math

import torch

from src.inference.flash_attention_wrapper import FlashAttentionWrapper, FlashAttnConfig


def make_qkv(batch=2, n_heads=4, seq_len=8, head_dim=16, device="cpu"):
    q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    return q, k, v


def naive_attention(q, k, v, causal=True, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        seq = q.size(2)
        mask = torch.ones(seq, seq, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


class TestFlashAttnConfig:
    def test_defaults(self):
        cfg = FlashAttnConfig()
        assert cfg.causal is True
        assert cfg.softmax_scale is None
        assert cfg.dropout_p == 0.0
        assert cfg.window_size == (-1, -1)
        assert cfg.alibi_slopes is None

    def test_custom_values(self):
        cfg = FlashAttnConfig(causal=False, softmax_scale=0.1, dropout_p=0.1)
        assert cfg.causal is False
        assert cfg.softmax_scale == 0.1
        assert cfg.dropout_p == 0.1

    def test_window_size_set(self):
        cfg = FlashAttnConfig(window_size=(64, 0))
        assert cfg.window_size == (64, 0)


class TestFlashAttentionWrapperInit:
    def test_default_config_assigned(self):
        wrapper = FlashAttentionWrapper()
        assert isinstance(wrapper.config, FlashAttnConfig)

    def test_custom_config_preserved(self):
        cfg = FlashAttnConfig(causal=False)
        wrapper = FlashAttentionWrapper(cfg)
        assert wrapper.config.causal is False

    def test_is_flash_available_is_bool(self):
        wrapper = FlashAttentionWrapper()
        assert isinstance(wrapper.is_flash_available, bool)


class TestBuildCausalMask:
    def test_shape(self):
        wrapper = FlashAttentionWrapper()
        mask = wrapper._build_causal_mask(6, torch.device("cpu"))
        assert mask.shape == (6, 6)

    def test_lower_triangular(self):
        wrapper = FlashAttentionWrapper()
        mask = wrapper._build_causal_mask(5, torch.device("cpu"))
        assert mask.dtype == torch.bool
        for i in range(5):
            for j in range(5):
                assert mask[i, j].item() == (j <= i)

    def test_size_one(self):
        wrapper = FlashAttentionWrapper()
        mask = wrapper._build_causal_mask(1, torch.device("cpu"))
        assert mask.shape == (1, 1)
        assert mask[0, 0].item() is True


class TestManualAttention:
    def test_matches_naive_causal(self):
        wrapper = FlashAttentionWrapper(FlashAttnConfig(causal=True))
        q, k, v = make_qkv()
        mask = wrapper._build_combined_mask(q, k, None)
        out = wrapper._manual_attention(q, k, v, mask)
        expected = naive_attention(q, k, v, causal=True)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_output_shape(self):
        wrapper = FlashAttentionWrapper()
        q, k, v = make_qkv(batch=3, n_heads=2, seq_len=5, head_dim=8)
        mask = wrapper._build_combined_mask(q, k, None)
        out = wrapper._manual_attention(q, k, v, mask)
        assert out.shape == q.shape

    def test_no_mask_noncausal(self):
        wrapper = FlashAttentionWrapper(FlashAttnConfig(causal=False))
        q, k, v = make_qkv(seq_len=4, head_dim=8)
        out = wrapper._manual_attention(q, k, v, mask=None)
        expected = naive_attention(q, k, v, causal=False)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_custom_scale(self):
        scale = 0.25
        wrapper = FlashAttentionWrapper(FlashAttnConfig(causal=False, softmax_scale=scale))
        q, k, v = make_qkv(seq_len=4, head_dim=8)
        out = wrapper._manual_attention(q, k, v, mask=None)
        expected = naive_attention(q, k, v, causal=False, scale=scale)
        assert torch.allclose(out, expected, atol=1e-5)


class TestForwardCPU:
    def test_forward_output_shape(self):
        wrapper = FlashAttentionWrapper()
        q, k, v = make_qkv()
        out = wrapper.forward(q, k, v)
        assert out.shape == q.shape

    def test_forward_causal_matches_naive(self):
        wrapper = FlashAttentionWrapper(FlashAttnConfig(causal=True))
        q, k, v = make_qkv(batch=1, n_heads=2, seq_len=6, head_dim=8)
        out = wrapper.forward(q, k, v)
        expected = naive_attention(q, k, v, causal=True)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_forward_noncausal(self):
        wrapper = FlashAttentionWrapper(FlashAttnConfig(causal=False))
        q, k, v = make_qkv(batch=1, n_heads=2, seq_len=4, head_dim=8)
        out = wrapper.forward(q, k, v)
        expected = naive_attention(q, k, v, causal=False)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_forward_with_attention_mask(self):
        wrapper = FlashAttentionWrapper(FlashAttnConfig(causal=False))
        q, k, v = make_qkv(batch=1, n_heads=1, seq_len=4, head_dim=8)
        additive_mask = torch.zeros(1, 1, 4, 4)
        out = wrapper.forward(q, k, v, attention_mask=additive_mask)
        assert out.shape == q.shape

    def test_window_size_local_attention(self):
        cfg = FlashAttnConfig(causal=True, window_size=(2, 0))
        wrapper = FlashAttentionWrapper(cfg)
        q, k, v = make_qkv(batch=1, n_heads=2, seq_len=6, head_dim=8)
        out = wrapper.forward(q, k, v)
        assert out.shape == q.shape

    def test_deterministic_output(self):
        wrapper = FlashAttentionWrapper(FlashAttnConfig(causal=True, dropout_p=0.0))
        q, k, v = make_qkv(batch=2, n_heads=4, seq_len=8, head_dim=16)
        out1 = wrapper.forward(q, k, v)
        out2 = wrapper.forward(q, k, v)
        assert torch.allclose(out1, out2)

    def test_batch_independence(self):
        wrapper = FlashAttentionWrapper(FlashAttnConfig(causal=True))
        q, k, v = make_qkv(batch=2, n_heads=2, seq_len=4, head_dim=8)
        out_full = wrapper.forward(q, k, v)
        out_b0 = wrapper.forward(q[:1], k[:1], v[:1])
        assert torch.allclose(out_full[:1], out_b0, atol=1e-5)

    def test_single_token(self):
        wrapper = FlashAttentionWrapper(FlashAttnConfig(causal=True))
        q, k, v = make_qkv(batch=1, n_heads=2, seq_len=1, head_dim=8)
        out = wrapper.forward(q, k, v)
        assert out.shape == (1, 2, 1, 8)
