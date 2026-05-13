"""Rate limiting backends for the Aurelius API.

Two implementations:
  • MemoryRateLimiter — in‑process token bucket (single‑instance)
  • RedisRateLimiter — distributed Lua‑script token bucket (multi‑instance)

Selection via environment variable:
  AURELIUS_RATE_LIMIT_REDIS_URL — if set, uses Redis; otherwise memory.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

# Optional Redis support
_HAVE_REDIS = False
try:
    import redis  # type: ignore
    _HAVE_REDIS = True
except Exception:  # pragma: no cover
    redis = None  # type: ignore


class MemoryRateLimiter:
    """Simple in‑memory token bucket."""

    def __init__(self, limit: int, window: int) -> None:
        self._limit = limit
        self._window = window
        self._store: dict[str, tuple[int, float]] = {}

    def _cleanup(self) -> None:
        if len(self._store) > 1000:
            cutoff = time.time() - self._window * 2
            for ip, (_, ts) in list(self._store.items()):
                if ts < cutoff:
                    self._store.pop(ip, None)

    def allow(self, ip: str) -> bool:
        now = time.time()
        self._cleanup()
        tokens, ts = self._store.get(ip, (self._limit, now))
        tokens = min(self._limit, tokens + int((now - ts) * (self._limit / self._window)))
        if tokens < 1:
            return False
        self._store[ip] = (tokens - 1, now)
        return True


class RedisRateLimiter:
    """Redis‑backed distributed token bucket using an atomic Lua script."""

    def __init__(
        self,
        limit: int,
        window: int,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "rl:",
    ) -> None:
        if not _HAVE_REDIS or redis is None:
            raise RuntimeError(
                "redis-py is not installed. Install with: pip install redis"
            )
        self._limit = limit
        self._window = window
        self._prefix = prefix
        self._client = redis.from_url(redis_url, decode_responses=True)

    def _key(self, ip: str) -> str:
        return f"{self._prefix}{ip}"

    def allow(self, ip: str) -> bool:
        key = self._key(ip)
        script = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local tokens = tonumber(redis.call('HGET', key, 'tokens'))
local last = tonumber(redis.call('HGET', key, 'ts'))
if tokens == nil then
  tokens = limit
  last = now
else
  local elapsed = now - last
  local refill = elapsed * (limit / window)
  tokens = math.min(limit, tokens + refill)
  last = now
end
if tokens < 1 then
  return 0
end
tokens = tokens - 1
redis.call('HMSET', key, 'tokens', tokens, 'ts', last)
redis.call('EXPIRE', key, math.ceil(window * 2))
return 1
"""
        allowed = self._client.eval(script, 1, key, self._limit, self._window, time.time())
        return bool(allowed)


def get_rate_limiter() -> Callable[[str], bool]:
    """Factory: return configured rate limiter callable."""
    limit = int(os.environ.get("AURELIUS_RATE_LIMIT", "120"))
    window = int(os.environ.get("AURELIUS_RATE_WINDOW", "60"))
    redis_url = os.environ.get("AURELIUS_RATE_LIMIT_REDIS_URL")
    if redis_url:
        if not _HAVE_REDIS:
            raise RuntimeError(
                "AURELIUS_RATE_LIMIT_REDIS_URL set but redis-py is not available"
            )
        return RedisRateLimiter(
            limit=limit,
            window=window,
            redis_url=redis_url,
            prefix=os.environ.get("AURELIUS_RATE_LIMIT_PREFIX", "rl:"),
        ).allow
    else:
        return MemoryRateLimiter(limit=limit, window=window).allow


__all__ = ["get_rate_limiter", "MemoryRateLimiter", "RedisRateLimiter"]
