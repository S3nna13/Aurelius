"""Integration tests: CircuitBreaker registers and wraps fake service calls."""

from __future__ import annotations

import pytest

from src.model.config import AureliusConfig
from src.runtime.feature_flags import FeatureFlag, FEATURE_FLAG_REGISTRY
from src.serving import (
    API_SHAPE_REGISTRY,
    DECODER_REGISTRY,
    RESILIENCE_REGISTRY,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


def test_registry_contains_circuit_breaker_entry() -> None:
    assert "circuit_breaker" in RESILIENCE_REGISTRY
    assert RESILIENCE_REGISTRY["circuit_breaker"] is CircuitBreaker


def test_prior_serving_registries_intact() -> None:
    # Additive registration must leave sibling registries intact.
    assert "json_schema" in DECODER_REGISTRY
    assert "grammar" in DECODER_REGISTRY
    assert "structured_output.json_schema" in API_SHAPE_REGISTRY


def test_config_flag_defaults_off() -> None:
    cfg = AureliusConfig()
    assert hasattr(cfg, "serving_circuit_breaker_enabled")
    assert cfg.serving_circuit_breaker_enabled is False


def test_config_flag_can_be_enabled() -> None:
    FEATURE_FLAG_REGISTRY.register(FeatureFlag(name="serving.circuit_breaker", enabled=True))
    cfg = AureliusConfig()
    assert cfg.serving_circuit_breaker_enabled is True


class _FakeFlakyService:
    """Deterministic flaky service: fails first N calls, then succeeds."""

    def __init__(self, fail_first: int) -> None:
        self.fail_first = fail_first
        self.calls = 0

    def __call__(self, x: int) -> int:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError(f"service unavailable (call {self.calls})")
        return x * 2


def test_breaker_wraps_fake_service_and_trips_open() -> None:
    cb = CircuitBreaker(
        name="fake_service",
        failure_threshold=3,
        recovery_timeout_s=5.0,
        probe_successes_required=1,
    )
    svc = _FakeFlakyService(fail_first=10)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            cb.call(svc, 5)
    assert cb.state == CircuitState.OPEN
    # Now short-circuits without hitting the service.
    before = svc.calls
    with pytest.raises(CircuitOpenError):
        cb.call(svc, 5)
    assert svc.calls == before  # service not invoked


def test_breaker_recovers_after_timeout() -> None:
    clock = {"t": 0.0}

    def time_fn() -> float:
        return clock["t"]

    cb = CircuitBreaker(
        name="fake_service_recovery",
        failure_threshold=2,
        recovery_timeout_s=5.0,
        probe_successes_required=2,
        time_fn=time_fn,
    )
    svc = _FakeFlakyService(fail_first=2)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(svc, 3)
    assert cb.state == CircuitState.OPEN
    # Advance past recovery timeout.
    clock["t"] = 6.0
    # Probe succeeds; a second probe closes the breaker.
    assert cb.call(svc, 3) == 6
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.call(svc, 4) == 8
    assert cb.state == CircuitState.CLOSED


def test_config_flag_gates_construction_in_userland() -> None:
    # Downstream code pattern: gate construction behind the config flag.
    FEATURE_FLAG_REGISTRY.register(FeatureFlag(name="serving.circuit_breaker", enabled=True))
    cfg = AureliusConfig()
    cb = (
        RESILIENCE_REGISTRY["circuit_breaker"](name="gated")
        if cfg.serving_circuit_breaker_enabled
        else None
    )
    assert isinstance(cb, CircuitBreaker)
    assert cb.state == CircuitState.CLOSED
