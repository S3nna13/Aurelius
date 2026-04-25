"""Tests for flops_profiler — FLOPs estimation for transformer layers."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.profiling.flops_profiler import FlopsProfiler, flops_for_linear, flops_for_attention


class TestFlopsForLinear:
    def test_linear_flops_matches_formula(self):
        layer = nn.Linear(512, 1024, bias=False)
        flops = flops_for_linear(layer)
        expected = 2 * 512 * 1024
        assert flops == expected

    def test_linear_flops_with_bias(self):
        layer = nn.Linear(256, 512, bias=True)
        flops = flops_for_linear(layer)
        expected = 2 * 256 * 512 + 512  # forward pass includes bias add
        assert flops >= expected

    def test_zero_input_dim_returns_zero(self):
        layer = nn.Linear(0, 64)
        assert flops_for_linear(layer) == 0


class TestFlopsForAttention:
    def test_attention_flops_scales_with_seq_len(self):
        flops_16 = flops_for_attention(seq_len=16, d_model=512, n_heads=8)
        flops_32 = flops_for_attention(seq_len=32, d_model=512, n_heads=8)
        assert flops_32 > flops_16

    def test_attention_flops_formula(self):
        d = 512
        flops = flops_for_attention(seq_len=128, d_model=d, n_heads=8)
        qkv = 3 * 2 * d * d
        attn = 2 * 128 * 128 * d  # QK^T + softmax scaling
        out_proj = 2 * d * d
        expected = qkv + attn + out_proj
        assert flops == expected

    def test_single_token_no_attention(self):
        flops = flops_for_attention(seq_len=1, d_model=64, n_heads=2)
        qkv = 3 * 2 * 64 * 64
        assert flops == qkv


class TestFlopsProfiler:
    def test_profile_linear_returns_flops(self):
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        profiler = FlopsProfiler()
        result = profiler.profile(model, input_shape=(1, 10))
        assert result > 0

    def test_profile_transformer_block(self):
        d = 64
        block = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=256, batch_first=True)
        profiler = FlopsProfiler()
        result = profiler.profile(block, input_shape=(1, 16, d))
        assert result > 0

    def test_profile_returns_dict_by_module(self):
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
        profiler = FlopsProfiler()
        by_module = profiler.profile_by_module(model, input_shape=(1, 10))
        assert isinstance(by_module, dict)
        assert len(by_module) >= 2

    def test_empty_model_returns_zero(self):
        profiler = FlopsProfiler()
        assert profiler.profile(nn.Sequential(), input_shape=(1, 1)) == 0
