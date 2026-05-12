"""Circuit breaker for serving-layer resilience.

Classic three-state circuit breaker (closed / open / half-open) inspired by
``SHAdd0WTAka/Zen-Ai-Pentest``'s ``core/orchestrator.py::CircuitBreaker``.

- ``CLOSED``: requests flow. After ``failure_threshold`` consecutive failures,
  the breaker trips open.
- ``OPEN``: requests short-circuit with :class:`CircuitOpenError`. After
  ``recovery_timeout_s`` elapses, the next ``state`` read transitions to
  ``HALF_OPEN``.
- ``HALF_OPEN``: probe traffic allowed. After ``probe_successes_required``
  successes the breaker closes and the recovery timeout resets; any failure
  re-opens the breaker with the recovery timeout multiplied by
  ``backoff_multiplier`` (capped at ``max_recovery_timeout_s``).

Pure stdlib.  Thread-safe via an internal :class:`threading.Lock`.
Time is injectable (``time_fn``) so tests can be deterministic.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is open."""


@dataclass
class CircuitStateTransition:
    from_state: CircuitState
    to_state: CircuitState
    at_time: float
    reason: str


@dataclass
class CircuitBreaker:
    """A three-state circuit breaker.

    All state transitions are appended to ``state_transitions``. The
    ``state`` property is side-effecting: reading it while the breaker
    is OPEN and the recovery timeout has elapsed flips the breaker to
    HALF_OPEN (and records the transition) so probe calls may run.
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0
    probe_successes_required: int = 3
    backoff_multiplier: float = 2.0
    max_recovery_timeout_s: float = 600.0
    time_fn: Callable[[], float] = time.monotonic

    # Internal state (not part of constructor contract).
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _probe_successes: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _current_recovery_timeout_s: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    total_calls: int = field(default=0, init=False)
    successes: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)
    state_transitions: list[CircuitStateTransition] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.probe_successes_required < 1:
            raise ValueError("probe_successes_required must be >= 1")
        if self.recovery_timeout_s <= 0:
            raise ValueError("recovery_timeout_s must be > 0")
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1.0")
        if self.max_recovery_timeout_s < self.recovery_timeout_s:
            raise ValueError("max_recovery_timeout_s must be >= recovery_timeout_s")
        self._current_recovery_timeout_s = self.recovery_timeout_s

    # ------------------------------------------------------------------
    # Internal helpers (must be called with ``self._lock`` held).
    # ------------------------------------------------------------------
    def _transition(self, to_state: CircuitState, reason: str) -> None:
        if to_state == self._state:
            return
        self.state_transitions.append(
            CircuitStateTransition(
                from_state=self._state,
                to_state=to_state,
                at_time=self.time_fn(),
                reason=reason,
            )
        )
        self._state = to_state

    def _maybe_half_open_locked(self) -> None:
        if self._state != CircuitState.OPEN:
            return
        if self._opened_at is None:
            return
        if (self.time_fn() - self._opened_at) >= self._current_recovery_timeout_s:
            self._probe_successes = 0
            self._transition(
                CircuitState.HALF_OPEN,
                f"recovery_timeout {self._current_recovery_timeout_s}s elapsed",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open_locked()
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._maybe_half_open_locked()
            self.successes += 1
            if self._state == CircuitState.CLOSED:
                self._consecutive_failures = 0
            elif self._state == CircuitState.HALF_OPEN:
                self._probe_successes += 1
                if self._probe_successes >= self.probe_successes_required:
                    self._consecutive_failures = 0
                    self._probe_successes = 0
                    self._opened_at = None
                    self._current_recovery_timeout_s = self.recovery_timeout_s
                    self._transition(
                        CircuitState.CLOSED,
                        f"{self.probe_successes_required} consecutive probe successes",
                    )
            # OPEN: success records are unusual (manual) -- do not auto-heal.

    def record_failure(self) -> None:
        with self._lock:
            self._maybe_half_open_locked()
            self.failures += 1
            if self._state == CircuitState.CLOSED:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.failure_threshold:
                    self._open_locked(f"{self._consecutive_failures} consecutive failures")
            elif self._state == CircuitState.HALF_OPEN:
                # Any failure during probe re-opens with backoff.
                self._open_locked("probe failure during half-open")

    def _open_locked(self, reason: str) -> None:
        was_half_open = self._state == CircuitState.HALF_OPEN
        self._opened_at = self.time_fn()
        if was_half_open:
            self._current_recovery_timeout_s = min(
                self._current_recovery_timeout_s * self.backoff_multiplier,
                self.max_recovery_timeout_s,
            )
        self._probe_successes = 0
        self._transition(CircuitState.OPEN, reason)

    def force_open(self, reason: str = "force_open") -> None:
        with self._lock:
            self._open_locked(reason)

    def force_close(self, reason: str = "force_close") -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._probe_successes = 0
            self._opened_at = None
            self._current_recovery_timeout_s = self.recovery_timeout_s
            self._transition(CircuitState.CLOSED, reason)

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke ``fn(*args, **kwargs)`` guarded by the breaker.

        Raises :class:`CircuitOpenError` if the breaker is OPEN.  On
        exception, records a failure and re-raises.  On success, records
        a success and returns the value.
        """
        with self._lock:
            self._maybe_half_open_locked()
            self.total_calls += 1
            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(f"circuit '{self.name}' is OPEN; short-circuiting call")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    # Context manager: acts as a gate -- enter checks state, exit
    # records success or failure based on whether an exception escaped.
    def __enter__(self) -> CircuitBreaker:
        with self._lock:
            self._maybe_half_open_locked()
            self.total_calls += 1
            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(f"circuit '{self.name}' is OPEN; short-circuiting call")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()
        return False  # never swallow
