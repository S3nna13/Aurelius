"""Simple rate limiter decorator for function call throttling."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RateLimitConfig:
    calls: int = 10
    per_seconds: float = 1.0
    _window: list[float] | None = None

    def __post_init__(self) -> None:
        self._window = []


def ratelimit(config: RateLimitConfig | None = None) -> Callable[[F], F]:
    cfg = config or RateLimitConfig()

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            now = time.monotonic()
            cfg._window = [t for t in (cfg._window or []) if now - t < cfg.per_seconds]
            if len(cfg._window) >= cfg.calls:
                sleep = cfg.per_seconds - (now - cfg._window[0])
                if sleep > 0:
                    time.sleep(sleep)
            cfg._window.append(time.monotonic())
            return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


RATE_LIMIT_CONFIG = RateLimitConfig()