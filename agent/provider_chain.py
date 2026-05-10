"""Provider chain with ordered fallback and circuit-breaker semantics.

Wraps multiple ``generate_fn``-compatible callables so the agent loop
can automatically fail over to the next provider when the current one
fails. Each provider has an independent circuit-breaker: after
``failure_threshold`` consecutive failures it is skipped for a cooldown
period, then retried.

Compatible with the ``generate_fn: Callable[[list[dict]], str]``
signature used by :class:`ReActLoop`, :class:`BudgetBoundedLoop`, etc.

Stdlib-only.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderStats:
    """Running statistics for a single provider."""

    name: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_error: str = ""
    circuit_open: bool = False

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls > 0 else 0.0


@dataclass
class ChainConfig:
    """Configuration for :class:`ProviderChain`."""

    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    max_attempts: int = 0  # 0 = try all providers once


@dataclass
class ChainResult:
    """Result of a single chain invocation."""

    provider_name: str
    output: str
    attempts: int
    tried_providers: list[str]
    error: str | None = None


class ProviderChain:
    """Ordered fallback chain across multiple ``generate_fn`` providers.

    Parameters
    ----------
    providers:
        Ordered mapping of ``{name: generate_fn}``. The first key is the
        primary provider; subsequent keys are fallbacks tried in order.
    config:
        Circuit-breaker configuration.
    """

    def __init__(
        self,
        providers: dict[str, Callable[[list[dict]], str]],
        config: ChainConfig | None = None,
    ) -> None:
        if not providers:
            raise ValueError("providers must be non-empty")
        for name, fn in providers.items():
            if not callable(fn):
                raise TypeError(f"provider {name!r} must be callable")
        self._providers: dict[str, Callable[[list[dict]], str]] = dict(providers)
        self._config: ChainConfig = config if config is not None else ChainConfig()
        self._stats: dict[str, ProviderStats] = {
            name: ProviderStats(name=name) for name in providers
        }
        self._lock = threading.Lock()

    @property
    def provider_names(self) -> list[str]:
        return list(self._providers.keys())

    def stats(self, name: str | None = None) -> dict[str, Any]:
        """Return stats for a single provider or all."""
        with self._lock:
            if name is not None:
                s = self._stats.get(name)
                if s is None:
                    raise KeyError(f"unknown provider: {name!r}")
                return {
                    "name": s.name,
                    "calls": s.calls,
                    "successes": s.successes,
                    "failures": s.failures,
                    "consecutive_failures": s.consecutive_failures,
                    "success_rate": s.success_rate,
                    "circuit_open": s.circuit_open,
                    "last_error": s.last_error,
                }
            return {n: self.stats(n) for n in self._stats}

    def reset_circuit(self, name: str) -> None:
        """Manually reset a tripped circuit breaker."""
        with self._lock:
            s = self._stats.get(name)
            if s is None:
                raise KeyError(f"unknown provider: {name!r}")
            s.consecutive_failures = 0
            s.circuit_open = False

    def _is_available(self, s: ProviderStats) -> bool:
        if not s.circuit_open:
            return True
        elapsed = time.monotonic() - s.last_failure_time
        if elapsed >= self._config.cooldown_seconds:
            s.circuit_open = False
            s.consecutive_failures = 0
            return True
        return False

    def __call__(self, messages: list[dict]) -> str:
        """Invoke providers in order until one succeeds.

        Compatible with the ``generate_fn`` signature.
        """
        result = self.call(messages)
        if result.error is not None:
            raise RuntimeError(
                f"all providers failed (tried {result.tried_providers}): {result.error}"
            )
        return result.output

    def call(self, messages: list[dict]) -> ChainResult:
        """Invoke providers in order, returning a :class:`ChainResult`."""
        tried: list[str] = []
        last_error = ""

        with self._lock:
            ordered = list(self._providers.items())

        for provider_name, generate_fn in ordered:
            with self._lock:
                s = self._stats[provider_name]
                if not self._is_available(s):
                    continue

            tried.append(provider_name)
            try:
                output = generate_fn(messages)
                if not isinstance(output, str):
                    raise TypeError(
                        f"provider {provider_name!r} returned {type(output).__name__}, expected str"
                    )
                with self._lock:
                    s.calls += 1
                    s.successes += 1
                    s.consecutive_failures = 0
                    s.circuit_open = False
                return ChainResult(
                    provider_name=provider_name,
                    output=output,
                    attempts=len(tried),
                    tried_providers=tried,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"{provider_name}: {type(exc).__name__}: {exc}"
                with self._lock:
                    s.calls += 1
                    s.failures += 1
                    s.consecutive_failures += 1
                    s.last_failure_time = time.monotonic()
                    s.last_error = last_error
                    if s.consecutive_failures >= self._config.failure_threshold:
                        s.circuit_open = True

        return ChainResult(
            provider_name="",
            output="",
            attempts=len(tried),
            tried_providers=tried,
            error=last_error,
        )


PROVIDER_CHAIN_REGISTRY: dict[str, type[ProviderChain]] = {"default": ProviderChain}
