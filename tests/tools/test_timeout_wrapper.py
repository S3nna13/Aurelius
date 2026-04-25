"""Tests for timeout wrapper."""
from __future__ import annotations

import time

import pytest

from src.tools.timeout_wrapper import TimeoutWrapper, TimeoutError


class TestTimeoutWrapper:
    def test_executes_before_timeout(self):
        tw = TimeoutWrapper(timeout_seconds=5.0)
        result = tw.execute(lambda: 42)
        assert result == 42

    def test_raises_on_timeout(self):
        tw = TimeoutWrapper(timeout_seconds=0.1)

        def slow():
            time.sleep(10)

        with pytest.raises(TimeoutError):
            tw.execute(slow)