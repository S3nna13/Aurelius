"""Retry-with-backoff tool wrapper for resilient tool execution."""
from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RetryTool:
    """Wrap any callable with retry + exponential backoff."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    _attempts: list[float] = field(default_factory=list, repr=False)

    def execute(self, fn: Callable[[], Any]) -> Any:
        self._attempts = []
        for attempt in range(self.max_retries + 1):
            try:
                result = fn()
                self._attempts.append(time.monotonic())
                return result
            except Exception as e:
                self._attempts.append(time.monotonic())
                if attempt == self.max_retries:
                    raise
                delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
                if self.jitter:
                    delay *= 0.5 + random.random() * 0.5
                time.sleep(delay)
        raise RuntimeError("unreachable")

    @property
    def attempts(self) -> int:
        return len(self._attempts)

    def reset(self) -> None:
        self._attempts.clear()


RETRY_TOOL = RetryTool()