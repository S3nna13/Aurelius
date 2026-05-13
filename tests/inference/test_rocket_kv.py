"""Tests for RocketKV two-stage KV cache compression."""

from __future__ import annotations

import torch

from src.inference.rocket_kv import RocketKVCache


def test_rocket_kv_stage1_evict():
    cache = RocketKVCache(n_layers=2, n_heads=4, head_dim=64, top_k_positions=64)
    k = torch.randn(2, 4, 128, 64)
    v = torch.randn(2, 4, 128, 64)
    k_ret, v_ret = cache.stage1_evict(layer=0, k=k, v=v)
    assert k_ret.shape == (2, 4, 64, 64)
    assert v_ret.shape == (2, 4, 64, 64)


def test_rocket_kv_small_input_passthrough():
    cache = RocketKVCache(n_layers=2, n_heads=4, head_dim=64, top_k_positions=128)
    k = torch.randn(2, 4, 64, 64)
    v = torch.randn(2, 4, 64, 64)
    k_ret, v_ret = cache.stage1_evict(layer=0, k=k, v=v)
    assert k_ret.shape == k.shape


def test_rocket_kv_stage2_get():
    cache = RocketKVCache(n_layers=2, n_heads=4, head_dim=64, top_k_positions=64)
    k = torch.randn(2, 4, 128, 64)
    v = torch.randn(2, 4, 128, 64)
    cache.stage1_evict(layer=0, k=k, v=v)
    k_cached, v_cached = cache.stage2_get(layer=0)
    assert k_cached is not None
    assert v_cached is not None


def test_rocket_kv_multiturn_update():
    cache = RocketKVCache(n_layers=2, n_heads=4, head_dim=64, top_k_positions=64)
    k1 = torch.randn(2, 4, 64, 64)
    v1 = torch.randn(2, 4, 64, 64)
    cache.stage1_evict(layer=0, k=k1, v=v1)
    k2 = torch.randn(2, 4, 32, 64)
    v2 = torch.randn(2, 4, 32, 64)
    cache.update_multiturn(layer=0, new_k=k2, new_v=v2)
    k_cached, v_cached = cache.stage2_get(layer=0)
    assert k_cached.shape[2] == 64


def test_rocket_kv_clear():
    cache = RocketKVCache(n_layers=2, n_heads=4, head_dim=64, top_k_positions=64)
    k = torch.randn(2, 4, 128, 64)
    v = torch.randn(2, 4, 128, 64)
    cache.stage1_evict(layer=0, k=k, v=v)
    cache.clear()
    k_cached, v_cached = cache.stage2_get(layer=0)
    assert k_cached is None
    assert v_cached is None
