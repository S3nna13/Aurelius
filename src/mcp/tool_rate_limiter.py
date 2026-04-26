from __future__ import annotations
import time
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class RateLimitResult:
    allowed: bool
    retry_after: float = 0.0

class ToolRateLimiter:
    def __init__(self, max_calls: int = 60, window: float = 60.0) -> None:
        self.max_calls = max_calls
        self.window = window
        self._calls: dict[str, list[float]] = defaultdict(list)

    def check_call(self, tool_name: str) -> RateLimitResult:
        now = time.monotonic()
        calls = self._calls[tool_name]
        calls[:] = [t for t in calls if now - t < self.window]
        if len(calls) >= self.max_calls:
            return RateLimitResult(allowed=False, retry_after=self.window - (now - calls[0]))
        calls.append(now)
        return RateLimitResult(allowed=True)

    def get_stats(self, tool_name: str) -> dict:
        calls = self._calls.get(tool_name, [])
        now = time.monotonic()
        active = [t for t in calls if now - t < self.window]
        return {"calls": len(active), "remaining": max(0, self.max_calls - len(active))}

TOOL_RATE_LIMITER = ToolRateLimiter()
