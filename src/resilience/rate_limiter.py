"""Token-bucket rate limiter with burst allowance.

Thread-safe, stdlib-only implementation suitable for limiting
request rates across threads or within a single process.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


class RateLimitExceededError(Exception):
    """Raised when a token cannot be acquired within the allowed wait time."""


@dataclass
class RateLimiter:
    """Token bucket rate limiter.

    Parameters
    ----------
    rate:
        Tokens added to the bucket per second (sustained throughput).
    burst:
        Maximum bucket capacity (allows short bursts).
    """

    rate: float = 10.0
    burst: int = 20

    _tokens: float = field(default=0.0, repr=False)
    _last_update: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        with self._lock:
            self._tokens = float(self.burst)
            self._last_update = time.monotonic()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def tokens(self) -> float:
        """Current token count (approximate, for observability)."""
        with self._lock:
            self._add_tokens()
            return self._tokens

    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: float | None = None) -> bool:
        """Attempt to acquire *tokens* from the bucket.

        Parameters
        ----------
        tokens:
            Number of tokens to consume.
        blocking:
            If ``False``, return immediately.
        timeout:
            Maximum seconds to wait when *blocking* is ``True``.

        Returns
        -------
        bool
            ``True`` if the tokens were acquired, ``False`` otherwise.

        Raises
        ------
        RateLimitExceededError
            If *blocking* is ``True`` and the wait times out.
        """
        deadline: float | None = None
        if blocking and timeout is not None:
            deadline = time.monotonic() + timeout

        with self._lock:
            while True:
                self._add_tokens()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                if not blocking:
                    return False
                wait_needed = (tokens - self._tokens) / self.rate
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RateLimitExceededError("Rate limit wait timed out")
                    wait_needed = min(wait_needed, remaining)
                self._lock.release()
                try:
                    time.sleep(wait_needed)
                finally:
                    self._lock.acquire()

    def execute(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* after acquiring a single token.

        Raises :exc:`RateLimitExceededError` if the token cannot be acquired.
        """
        self.acquire(tokens=1, blocking=True, timeout=None)
        return fn(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _add_tokens(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_update = now
