"""Circuit breaker for downstream protocol calls."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto


class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


@dataclass
class CircuitBreaker:
    """Stateful circuit breaker for calling downstream services.

    Closed → Open on failure_threshold failures.
    Open → Half-Open after cooldown.
    Half-Open → Closed on success, back to Open on failure.
    """

    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _last_failure_time: float = 0.0

    def call(self, fn):
        if self._state is CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
            else:
                raise RuntimeError("circuit breaker open")

        try:
            result = fn()
        except Exception as e:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
            raise e

        if self._state is CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
        self._failure_count = 0
        return result

    def reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count
