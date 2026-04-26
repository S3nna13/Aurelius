"""Tests for message_retry_policy — pure delay computation."""

from __future__ import annotations

import pytest

from src.protocol.message_retry_policy import (
    DEFAULT_RETRY_POLICY,
    RETRY_POLICY_REGISTRY,
    BackoffStrategy,
    RetryPolicy,
)

# ---------------------------------------------------------------------------
# Fixed strategy
# ---------------------------------------------------------------------------


def test_fixed_delay_constant():
    p = RetryPolicy(strategy=BackoffStrategy.FIXED, base_delay_seconds=2.0, jitter=False)
    assert p.delay(1) == 2.0
    assert p.delay(2) == 2.0
    assert p.delay(3) == 2.0


# ---------------------------------------------------------------------------
# Linear strategy
# ---------------------------------------------------------------------------


def test_linear_delay_increases():
    p = RetryPolicy(strategy=BackoffStrategy.LINEAR, base_delay_seconds=1.0, jitter=False)
    assert p.delay(1) == 1.0
    assert p.delay(2) == 2.0
    assert p.delay(3) == 3.0


# ---------------------------------------------------------------------------
# Exponential strategy
# ---------------------------------------------------------------------------


def test_exponential_delay_doubles():
    p = RetryPolicy(strategy=BackoffStrategy.EXPONENTIAL, base_delay_seconds=1.0, jitter=False)
    assert p.delay(1) == 1.0
    assert p.delay(2) == 2.0
    assert p.delay(3) == 4.0


# ---------------------------------------------------------------------------
# Max delay cap
# ---------------------------------------------------------------------------


def test_delay_capped_at_max():
    p = RetryPolicy(
        strategy=BackoffStrategy.EXPONENTIAL,
        base_delay_seconds=10.0,
        max_delay_seconds=15.0,
        jitter=False,
    )
    assert p.delay(3) == 15.0


# ---------------------------------------------------------------------------
# Attempt bounds
# ---------------------------------------------------------------------------


def test_delay_zero_when_over_budget():
    p = RetryPolicy(max_attempts=2, jitter=False)
    assert p.delay(3) == 0.0


def test_should_retry_within_budget():
    p = RetryPolicy(max_attempts=3)
    assert p.should_retry(1) is True
    assert p.should_retry(2) is True
    assert p.should_retry(3) is False


def test_invalid_attempt_raises():
    p = RetryPolicy()
    with pytest.raises(ValueError):
        p.delay(0)


# ---------------------------------------------------------------------------
# Jitter
# ---------------------------------------------------------------------------


def test_jitter_adds_positive_offset():
    p = RetryPolicy(
        strategy=BackoffStrategy.FIXED,
        base_delay_seconds=1.0,
        jitter=True,
        jitter_max_seconds=0.5,
    )
    d = p.delay(1)
    assert 1.0 <= d <= 1.5


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in RETRY_POLICY_REGISTRY
    assert isinstance(RETRY_POLICY_REGISTRY["default"], RetryPolicy)


def test_registry_has_aggressive():
    aggressive = RETRY_POLICY_REGISTRY["aggressive"]
    assert aggressive.max_attempts == 5


def test_registry_has_gentle():
    gentle = RETRY_POLICY_REGISTRY["gentle"]
    assert gentle.strategy == BackoffStrategy.LINEAR


def test_default_is_retry_policy():
    assert isinstance(DEFAULT_RETRY_POLICY, RetryPolicy)
