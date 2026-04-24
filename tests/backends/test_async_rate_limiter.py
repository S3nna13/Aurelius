"""Tests for src/backends/async_rate_limiter.py"""
from __future__ import annotations

import asyncio
import time

import pytest

from src.backends.async_rate_limiter import (
    ASYNC_RATE_LIMITER,
    AsyncFixedWindowRateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
)


# ---------------------------------------------------------------------------
# RateLimitConfig
# ---------------------------------------------------------------------------

def test_config_burst_limit():
    cfg = RateLimitConfig(requests_per_window=100, window_seconds=60.0, burst_multiplier=2.0)
    assert cfg.burst_limit == 200


def test_config_default_burst_multiplier():
    cfg = RateLimitConfig(requests_per_window=10, window_seconds=1.0)
    assert cfg.burst_multiplier == 1.5
    assert cfg.burst_limit == 15


# ---------------------------------------------------------------------------
# acquire — happy path
# ---------------------------------------------------------------------------

def test_acquire_returns_true_within_limit():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(5, 60.0))
    result = asyncio.run(limiter.acquire("k"))
    assert result is True


def test_acquire_exhausts_exactly_at_limit():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(3, 60.0))

    async def run():
        results = [await limiter.acquire("x") for _ in range(3)]
        return results

    results = asyncio.run(run())
    assert all(results)


def test_acquire_raises_after_limit():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(2, 60.0))

    async def run():
        await limiter.acquire("k")
        await limiter.acquire("k")
        await limiter.acquire("k")

    with pytest.raises(RateLimitExceeded):
        asyncio.run(run())


def test_acquire_per_key_isolation():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(1, 60.0))

    async def run():
        r1 = await limiter.acquire("key1")
        r2 = await limiter.acquire("key2")
        return r1, r2

    r1, r2 = asyncio.run(run())
    assert r1 is True
    assert r2 is True


# ---------------------------------------------------------------------------
# try_acquire
# ---------------------------------------------------------------------------

def test_try_acquire_returns_false_instead_of_raising():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(1, 60.0))

    async def run():
        await limiter.acquire("k")
        return await limiter.try_acquire("k")

    result = asyncio.run(run())
    assert result is False


def test_try_acquire_returns_true_within_limit():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(5, 60.0))
    result = asyncio.run(limiter.try_acquire("k"))
    assert result is True


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_clears_count():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(2, 60.0))

    async def run():
        await limiter.acquire("k")
        await limiter.acquire("k")
        limiter.reset("k")
        return await limiter.acquire("k")

    result = asyncio.run(run())
    assert result is True


def test_reset_unknown_key_no_error():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(5, 60.0))
    limiter.reset("does-not-exist")


# ---------------------------------------------------------------------------
# get_remaining
# ---------------------------------------------------------------------------

def test_get_remaining_full_on_new_key():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(10, 60.0))
    assert limiter.get_remaining("new") == 10


def test_get_remaining_decrements():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(5, 60.0))

    async def run():
        await limiter.acquire("r")
        await limiter.acquire("r")
        return limiter.get_remaining("r")

    remaining = asyncio.run(run())
    assert remaining == 3


def test_get_remaining_zero_at_limit():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(2, 60.0))

    async def run():
        await limiter.acquire("r")
        await limiter.acquire("r")
        return limiter.get_remaining("r")

    remaining = asyncio.run(run())
    assert remaining == 0


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

def test_get_stats_returns_dict():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(10, 60.0))

    async def run():
        await limiter.acquire("s")
        return limiter.get_stats()

    stats = asyncio.run(run())
    assert "s" in stats
    assert stats["s"]["count"] == 1
    assert stats["s"]["remaining"] == 9
    assert "window_start" in stats["s"]


def test_get_stats_multiple_keys():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(10, 60.0))

    async def run():
        await limiter.acquire("a")
        await limiter.acquire("b")
        await limiter.acquire("b")
        return limiter.get_stats()

    stats = asyncio.run(run())
    assert stats["a"]["count"] == 1
    assert stats["b"]["count"] == 2


# ---------------------------------------------------------------------------
# Window expiry
# ---------------------------------------------------------------------------

def test_window_resets_after_expiry():
    limiter = AsyncFixedWindowRateLimiter(RateLimitConfig(1, 0.05))

    async def run():
        await limiter.acquire("w")
        await asyncio.sleep(0.06)
        return await limiter.acquire("w")

    result = asyncio.run(run())
    assert result is True


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def test_singleton_config():
    assert ASYNC_RATE_LIMITER._config.requests_per_window == 100
    assert ASYNC_RATE_LIMITER._config.window_seconds == 60.0
