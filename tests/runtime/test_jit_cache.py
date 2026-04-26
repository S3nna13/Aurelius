"""Tests for src/runtime/jit_cache.py — 8+ tests, CPU-only."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.runtime.jit_cache import (
    RUNTIME_REGISTRY,
    JITCache,
    JITCacheConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_config() -> JITCacheConfig:
    return JITCacheConfig(cache_dir=".test_jit_cache", max_entries=4)


@pytest.fixture()
def simple_linear() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(8, 4)


@pytest.fixture()
def tiny_inputs() -> list:
    return [torch.randn(2, 8)]


# ---------------------------------------------------------------------------
# JITCacheConfig tests
# ---------------------------------------------------------------------------


class TestJITCacheConfig:
    def test_defaults(self):
        cfg = JITCacheConfig()
        assert cfg.cache_dir == ".aurelius_jit_cache"
        assert cfg.max_entries == 64

    def test_custom(self):
        cfg = JITCacheConfig(cache_dir="/tmp/jit", max_entries=8)
        assert cfg.cache_dir == "/tmp/jit"
        assert cfg.max_entries == 8


# ---------------------------------------------------------------------------
# JITCache tests
# ---------------------------------------------------------------------------


class TestJITCache:
    def test_get_or_compile_returns_module(self, default_config, simple_linear, tiny_inputs):
        cache = JITCache(default_config)
        result = cache.get_or_compile(simple_linear, tiny_inputs)
        assert result is not None
        assert callable(result)

    def test_cache_hit_returns_same_object(self, default_config, simple_linear, tiny_inputs):
        cache = JITCache(default_config)
        first = cache.get_or_compile(simple_linear, tiny_inputs)
        second = cache.get_or_compile(simple_linear, tiny_inputs)
        assert first is second

    def test_cache_size_increments(self, default_config, tiny_inputs):
        """Two modules with different input shapes should produce distinct keys."""
        cache = JITCache(default_config)
        assert cache.cache_size() == 0
        m = nn.Linear(8, 4)
        inputs_a = [torch.randn(2, 8)]
        inputs_b = [torch.randn(3, 8)]  # different batch size → different shape key
        cache.get_or_compile(m, inputs_a)
        assert cache.cache_size() == 1
        cache.get_or_compile(m, inputs_b)
        assert cache.cache_size() == 2

    def test_lru_eviction(self):
        """Three distinct input shapes should trigger LRU eviction at max_entries=2."""
        cfg = JITCacheConfig(max_entries=2)
        cache = JITCache(cfg)
        m = nn.Linear(8, 4)
        inputs_a = [torch.randn(1, 8)]
        inputs_b = [torch.randn(2, 8)]
        inputs_c = [torch.randn(3, 8)]

        cache.get_or_compile(m, inputs_a)
        cache.get_or_compile(m, inputs_b)
        assert cache.cache_size() == 2
        # Adding a third entry should evict the LRU (inputs_a)
        cache.get_or_compile(m, inputs_c)
        assert cache.cache_size() == 2  # still at max_entries

    def test_clear_empties_cache(self, default_config, simple_linear, tiny_inputs):
        cache = JITCache(default_config)
        cache.get_or_compile(simple_linear, tiny_inputs)
        assert cache.cache_size() == 1
        cache.clear()
        assert cache.cache_size() == 0

    def test_invalidate_removes_entry(self, default_config, simple_linear, tiny_inputs):
        cache = JITCache(default_config)
        cache.get_or_compile(simple_linear, tiny_inputs)
        assert cache.cache_size() == 1
        found = cache.invalidate(simple_linear, tiny_inputs)
        assert found is True
        assert cache.cache_size() == 0

    def test_invalidate_missing_returns_false(self, default_config, simple_linear, tiny_inputs):
        cache = JITCache(default_config)
        found = cache.invalidate(simple_linear, tiny_inputs)
        assert found is False

    def test_compiled_module_is_callable(self, default_config, simple_linear, tiny_inputs):
        cache = JITCache(default_config)
        compiled = cache.get_or_compile(simple_linear, tiny_inputs)
        out = compiled(tiny_inputs[0])
        assert out.shape == (2, 4)

    def test_registry_entry(self):
        assert "jit_cache" in RUNTIME_REGISTRY
        assert RUNTIME_REGISTRY["jit_cache"] is JITCache
