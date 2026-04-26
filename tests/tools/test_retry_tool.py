"""Tests for retry tool."""

from __future__ import annotations

import pytest

from src.tools.retry_tool import RetryTool


class TestRetryTool:
    def test_success_on_first_attempt(self):
        tool = RetryTool(max_retries=3)
        result = tool.execute(lambda: 42)
        assert result == 42
        assert tool.attempts == 1

    def test_retries_and_succeeds(self):
        tool = RetryTool(max_retries=3, base_delay=0.01, jitter=False)
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("not yet")
            return "ok"

        result = tool.execute(flaky)
        assert result == "ok"
        assert call_count[0] == 3

    def test_exhausts_retries(self):
        tool = RetryTool(max_retries=2, base_delay=0.01, jitter=False)

        def always_fails():
            raise ValueError("always")

        with pytest.raises(ValueError):
            tool.execute(always_fails)
        assert tool.attempts == 3  # initial + 2 retries

    def test_reset(self):
        tool = RetryTool(max_retries=1, base_delay=0.01)
        with pytest.raises(ValueError):
            tool.execute(lambda: (_ for _ in ()).throw(ValueError("x")))
        assert tool.attempts == 2
        tool.reset()
        assert tool.attempts == 0
