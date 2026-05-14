"""Circuit breaker pattern for fault-tolerant service calls.

Tracks failure rate and transitions between states:
  * CLOSED   — normal operation, failures are counted.
  * OPEN     — threshold exceeded; calls fail fast.
  * HALF_OPEN — after timeout, one probe call is allowed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class _State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Thread-safe circuit breaker.

    Parameters
    ----------
    failure_threshold:
        Number of consecutive failures required to OPEN the circuit.
    recovery_timeout:
        Seconds to wait before transitioning OPEN → HALF_OPEN.
    half_open_max_calls:
        Number of probe calls allowed in HALF_OPEN state before deciding.
    success_threshold_half_open:
        Consecutive successes required in HALF_OPEN to close the circuit.
    name:
        Optional identifier for logging / debugging.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold_half_open: int = 2
    name: str = "default"

    _state: _State = field(default=_State.CLOSED, repr=False)
    _failure_count: int = field(default=0, repr=False)
    _success_count: int = field(default=0, repr=False)
    _half_open_calls: int = field(default=0, repr=False)
    _last_failure_time: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> str:
        """Return current state as a lower-case string."""
        with self._lock:
            return self._state.value

    def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* respecting the circuit state.

        Raises
        ------
        CircuitBreakerOpenError
            If the circuit is OPEN and the recovery timeout has not elapsed.
        """
        self._before_call()
        try:
            result = fn(*args, **kwargs)
        except BaseException:
            self._on_failure()
            raise
        self._on_success()
        return result

    def record_success(self) -> None:
        """Manually record a success (useful when wrapping async / external work)."""
        self._on_success()

    def record_failure(self) -> None:
        """Manually record a failure."""
        self._on_failure()

    # ------------------------------------------------------------------ #
    # Internal state machine
    # ------------------------------------------------------------------ #

    def _before_call(self) -> None:
        with self._lock:
            if self._state is _State.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._state = _State.HALF_OPEN
                    self._half_open_calls = 0
                    self._success_count = 0
                else:
                    raise CircuitBreakerOpenError(f"Circuit {self.name!r} is OPEN")
            if self._state is _State.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit {self.name!r} is OPEN (half-open limit reached)"
                    )
                self._half_open_calls += 1

    def _on_success(self) -> None:
        with self._lock:
            if self._state is _State.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold_half_open:
                    self._reset()
            elif self._state is _State.CLOSED:
                self._failure_count = 0

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state is _State.HALF_OPEN:
                self._state = _State.OPEN
                self._half_open_calls = 0
                self._success_count = 0
            elif self._state is _State.CLOSED and self._failure_count >= self.failure_threshold:
                self._state = _State.OPEN

    def _reset(self) -> None:
        self._state = _State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = 0.0


class CircuitBreakerOpenError(Exception):
    """Raised when a call is attempted while the circuit is OPEN."""
