"""Tests for retry decorator."""
from __future__ import annotations

import pytest

from src.tools.retry_decorator import retry, RetryConfig


class TestRetryDecorator:
    def test_success_no_retry(self):
        call_count = [0]

        @retry()
        def fn():
            call_count[0] += 1
            return 42

        assert fn() == 42
        assert call_count[0] == 1

    def test_retry_on_failure(self):
        call_count = [0]

        @retry(RetryConfig(max_attempts=3, base_delay=0.01, backoff="constant"))
        def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("try again")
            return "ok"

        assert fn() == "ok"
        assert call_count[0] == 3