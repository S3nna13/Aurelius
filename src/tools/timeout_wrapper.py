"""Tool execution timeout wrapper with cancellation support."""
from __future__ import annotations

import signal
from dataclasses import dataclass
from typing import Any, Callable


class TimeoutError(RuntimeError):
    """Raised when a tool execution exceeds its timeout."""


@dataclass
class TimeoutWrapper:
    """Wrap a tool call with a timeout using SIGALRM."""

    timeout_seconds: float = 30.0

    def execute(self, fn: Callable[[], Any]) -> Any:
        import time

        deadline = time.monotonic() + self.timeout_seconds

        def handler(signum, frame):
            raise TimeoutError(f"timed out after {self.timeout_seconds}s")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(self.timeout_seconds) + 1)
        try:
            result = fn()
            elapsed = time.monotonic() - (deadline - self.timeout_seconds)
            if elapsed > self.timeout_seconds:
                raise TimeoutError(f"timed out after {self.timeout_seconds}s")
            return result
        except TimeoutError:
            raise
        finally:
            signal.alarm(0)


TIMEOUT_TOOL = TimeoutWrapper()