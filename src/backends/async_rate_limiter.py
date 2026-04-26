"""asyncio-native fixed-window rate limiter (Redis Lua fallback without Redis)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

__all__ = [
    "RateLimitConfig",
    "RateLimitExceeded",
    "AsyncFixedWindowRateLimiter",
    "ASYNC_RATE_LIMITER",
]


@dataclass
class RateLimitConfig:
    requests_per_window: int
    window_seconds: float
    burst_multiplier: float = 1.5

    @property
    def burst_limit(self) -> int:
        return int(self.requests_per_window * self.burst_multiplier)


class RateLimitExceeded(Exception):
    pass


@dataclass
class _WindowState:
    window_start: float
    count: int


class AsyncFixedWindowRateLimiter:
    """asyncio-native fixed-window rate limiter (Redis Lua fallback without Redis)."""

    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        self._windows: dict[str, _WindowState] = {}

    def _get_or_create(self, key: str) -> _WindowState:
        if key not in self._windows:
            self._windows[key] = _WindowState(window_start=time.monotonic(), count=0)
        return self._windows[key]

    def _maybe_reset(self, state: _WindowState) -> _WindowState:
        now = time.monotonic()
        if now >= state.window_start + self._config.window_seconds:
            state.window_start = now
            state.count = 0
        return state

    async def acquire(self, key: str = "default") -> bool:
        state = self._maybe_reset(self._get_or_create(key))
        if state.count < self._config.requests_per_window:
            state.count += 1
            return True
        raise RateLimitExceeded(
            f"Rate limit exceeded for key '{key}': "
            f"{self._config.requests_per_window} req/{self._config.window_seconds}s"
        )

    async def try_acquire(self, key: str = "default") -> bool:
        try:
            return await self.acquire(key)
        except RateLimitExceeded:
            return False

    def reset(self, key: str = "default") -> None:
        if key in self._windows:
            state = self._windows[key]
            state.window_start = time.monotonic()
            state.count = 0

    def get_remaining(self, key: str = "default") -> int:
        if key not in self._windows:
            return self._config.requests_per_window
        state = self._maybe_reset(self._windows[key])
        return max(0, self._config.requests_per_window - state.count)

    def get_stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, state in self._windows.items():
            self._maybe_reset(state)
            result[key] = {
                "count": state.count,
                "window_start": state.window_start,
                "remaining": max(0, self._config.requests_per_window - state.count),
            }
        return result


ASYNC_RATE_LIMITER = AsyncFixedWindowRateLimiter(RateLimitConfig(100, 60.0))
