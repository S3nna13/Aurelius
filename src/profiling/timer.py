"""Wall-clock timing decorator for profiling code sections."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class Timer:
    """Precision timer for profiling."""

    name: str = ""
    _start: float | None = None
    _elapsed: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is not None:
            self._elapsed += time.perf_counter() - self._start
            self._start = None
        return self._elapsed

    @property
    def elapsed(self) -> float:
        if self._start is not None:
            return self._elapsed + (time.perf_counter() - self._start)
        return self._elapsed

    def reset(self) -> None:
        self._elapsed = 0.0
        self._start = None


def timed(name: str = "") -> Callable[[F], F]:
    """Decorator that wraps a function with a Timer."""
    def decorator(func: F) -> F:
        timer = Timer(name=name or func.__name__)
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            timer.start()
            try:
                return func(*args, **kwargs)
            finally:
                timer.stop()
        wrapper._timer = timer  # type: ignore
        return wrapper  # type: ignore
    return decorator


_TIMERS: dict[str, Timer] = {}


def get_timer(name: str) -> Timer:
    if name not in _TIMERS:
        _TIMERS[name] = Timer(name=name)
    return _TIMERS[name]


def report_all() -> dict[str, float]:
    return {name: t.elapsed for name, t in _TIMERS.items()}