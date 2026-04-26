"""Circuit breaker pattern for backend fault tolerance.

Implements the classic CLOSED → OPEN → HALF_OPEN state machine so that
repeated backend failures are detected quickly and the system is given time
to recover before new requests are allowed through.

All logic is stdlib-only; no foreign dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CIRCUIT_BREAKER_REGISTRY",
]


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Tuning knobs for a :class:`CircuitBreaker`."""

    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0
    success_threshold: int = 2


class CircuitBreaker:
    """State machine that guards a downstream resource.

    Parameters
    ----------
    name:
        Human-readable label used in :meth:`status` output.
    config:
        Tuning configuration; a default :class:`CircuitBreakerConfig` is used
        when *config* is ``None``.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        self._name = name
        self._config: CircuitBreakerConfig = (
            config if config is not None else CircuitBreakerConfig()
        )
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._open_time: float = 0.0

    # ------------------------------------------------------------------
    # Public properties (read-only views)
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def success_count(self) -> int:
        return self._success_count

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def record_success(self) -> CircuitState:
        """Notify the breaker that the last call succeeded.

        In ``HALF_OPEN``: increments the success counter; when it reaches
        ``success_threshold`` the circuit closes and all counts reset.

        In ``CLOSED``: resets the failure counter (health restored).

        Returns the current :class:`CircuitState` after the transition.
        """
        if self._state is CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        elif self._state is CircuitState.CLOSED:
            self._failure_count = 0
        return self._state

    def record_failure(self) -> CircuitState:
        """Notify the breaker that the last call failed.

        In ``CLOSED``: increments failure counter; trips to ``OPEN`` when the
        threshold is reached.

        In ``HALF_OPEN``: immediately trips back to ``OPEN``.

        Returns the current :class:`CircuitState` after the transition.
        """
        if self._state is CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._open_time = time.monotonic()
            self._success_count = 0
        else:
            self._failure_count += 1
            if (
                self._state is CircuitState.CLOSED
                and self._failure_count >= self._config.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._open_time = time.monotonic()
        return self._state

    def allow_request(self) -> bool:
        """Decide whether a new request may proceed.

        * ``CLOSED`` → always ``True``.
        * ``OPEN`` → ``False`` until ``recovery_timeout_s`` elapses, then
          transitions to ``HALF_OPEN`` and returns ``True``.
        * ``HALF_OPEN`` → ``True`` (probe requests are permitted).
        """
        if self._state is CircuitState.CLOSED:
            return True
        if self._state is CircuitState.OPEN:
            elapsed = time.monotonic() - self._open_time
            if elapsed >= self._config.recovery_timeout_s:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                return True
            return False
        # HALF_OPEN
        return True

    def reset(self) -> None:
        """Force the breaker back to its initial ``CLOSED`` state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._open_time = 0.0

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return a JSON-safe snapshot of the breaker's current state."""
        return {
            "name": self._name,
            "state": self._state.value,
            "failures": self._failure_count,
            "successes": self._success_count,
        }


CIRCUIT_BREAKER_REGISTRY: dict[str, type[CircuitBreaker]] = {"default": CircuitBreaker}
