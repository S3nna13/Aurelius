"""Tests for request throttler."""
from __future__ import annotations

import pytest

from src.security.request_throttler import RequestThrottler


class TestRequestThrottler:
    def test_allows_initial_requests(self):
        rt = RequestThrottler(default_capacity=5, default_refill=10.0)
        for _ in range(5):
            assert rt.allow("alice") is True

    def test_blocks_excess(self):
        rt = RequestThrottler(default_capacity=2, default_refill=0.0)
        assert rt.allow("bob") is True
        assert rt.allow("bob") is True
        assert rt.allow("bob") is False

    def test_different_callers_independent(self):
        rt = RequestThrottler(default_capacity=1, default_refill=0.0)
        assert rt.allow("carol") is True
        assert rt.allow("dave") is True
        assert rt.allow("carol") is False

    def test_reset(self):
        rt = RequestThrottler(default_capacity=1, default_refill=0.0)
        assert rt.allow("eve") is True
        assert rt.allow("eve") is False
        rt.reset_caller("eve")
        assert rt.allow("eve") is True

    def test_active_callers(self):
        rt = RequestThrottler()
        rt.allow("a")
        rt.allow("b")
        rt.allow("c")
        assert rt.active_callers() == 3