"""Simple retry decorator with configurable backoff strategy."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff: str = "exponential"  # constant, linear, exponential


def retry(config: RetryConfig | None = None) -> Callable[[F], F]:
    cfg = config or RetryConfig()

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(cfg.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt == cfg.max_attempts - 1:
                        raise
                    if cfg.backoff == "constant":
                        delay = cfg.base_delay
                    elif cfg.backoff == "linear":
                        delay = cfg.base_delay * (attempt + 1)
                    else:
                        delay = min(cfg.base_delay * (2**attempt), cfg.max_delay)
                    time.sleep(delay)
            raise RuntimeError("unreachable") from last_exc

        return wrapper  # type: ignore

    return decorator


RETRY_CONFIG = RetryConfig()
