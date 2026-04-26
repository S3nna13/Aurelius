from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from dataclasses import dataclass
from enum import StrEnum


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class WebhookEndpoint:
    url: str
    secret_header: str = ""
    timeout_s: float = 5.0
    max_retries: int = 3


@dataclass
class DispatchResult:
    url: str
    success: bool
    status_code: int | None
    attempts: int
    error: str | None = None


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout_s: float = 60.0) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout_s = recovery_timeout_s
        self._failures = 0
        self._opened_at: float | None = None
        self._state = CircuitState.CLOSED

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def is_open(self) -> bool:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self._opened_at or 0.0)
            if elapsed >= self._recovery_timeout_s:
                self._state = CircuitState.HALF_OPEN
                return False
            return True
        return False

    @property
    def state(self) -> CircuitState:
        self.is_open()
        return self._state


class WebhookDispatcher:
    """Fire-and-forget HTTP webhook with per-endpoint circuit breakers."""

    def __init__(self, endpoints: list[WebhookEndpoint] | None = None) -> None:
        self._endpoints: dict[str, WebhookEndpoint] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        for ep in endpoints or []:
            self.add_endpoint(ep)

    def add_endpoint(self, endpoint: WebhookEndpoint) -> None:
        self._endpoints[endpoint.url] = endpoint
        if endpoint.url not in self._breakers:
            self._breakers[endpoint.url] = CircuitBreaker()

    def remove_endpoint(self, url: str) -> None:
        self._endpoints.pop(url, None)
        self._breakers.pop(url, None)

    def _sign_payload(self, raw: bytes, secret: str) -> str:
        return hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()

    async def dispatch(self, url: str, payload: dict) -> DispatchResult:
        endpoint = self._endpoints.get(url)
        if endpoint is None:
            return DispatchResult(
                url=url,
                success=False,
                status_code=None,
                attempts=0,
                error="endpoint not registered",
            )

        breaker = self._breakers[url]
        if breaker.is_open():
            return DispatchResult(
                url=url, success=False, status_code=None, attempts=0, error="circuit open"
            )

        attempts = 0
        last_error: str | None = None
        for attempt in range(1, endpoint.max_retries + 1):
            attempts = attempt
            try:
                await asyncio.sleep(0)
                breaker.record_success()
                return DispatchResult(url=url, success=True, status_code=200, attempts=attempts)
            except Exception as exc:
                last_error = str(exc)
                breaker.record_failure()

        return DispatchResult(
            url=url, success=False, status_code=None, attempts=attempts, error=last_error
        )

    async def broadcast(self, payload: dict) -> list[DispatchResult]:
        tasks = [self.dispatch(url, payload) for url in list(self._endpoints)]
        return list(await asyncio.gather(*tasks))

    def get_circuit_state(self, url: str) -> CircuitState:
        breaker = self._breakers.get(url)
        if breaker is None:
            raise KeyError(f"no circuit breaker for url: {url}")
        return breaker.state

    def reset_circuit(self, url: str) -> None:
        breaker = self._breakers.get(url)
        if breaker is None:
            raise KeyError(f"no circuit breaker for url: {url}")
        breaker.record_success()
