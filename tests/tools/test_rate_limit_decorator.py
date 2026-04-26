"""Tests for rate limit decorator."""

from __future__ import annotations

import time

from src.tools.rate_limit_decorator import RateLimitConfig, ratelimit


class TestRateLimitDecorator:
    def test_allows_within_limit(self):
        call_times = []

        @ratelimit(RateLimitConfig(calls=5, per_seconds=1.0))
        def fn():
            call_times.append(time.monotonic())
            return len(call_times)

        for _ in range(3):
            fn()
        assert len(call_times) == 3
