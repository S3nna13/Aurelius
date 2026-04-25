"""Tests for src.backends.retry_policy."""
from __future__ import annotations

import pytest

from src.backends.retry_policy import RetryConfig, RetryPolicy, RetryResult, RETRY_POLICY_REGISTRY


def _no_sleep(s: float) -> None:
    pass


def test_registry_has_default():
    assert RETRY_POLICY_REGISTRY["default"] is RetryPolicy


def test_delay_attempt_zero_base_delay():
    cfg = RetryConfig(base_delay_s=2.0, backoff_factor=2.0, jitter=False)
    rp = RetryPolicy(cfg)
    assert rp.delay_for_attempt(0) == pytest.approx(2.0)


def test_delay_attempt_one_backoff():
    cfg = RetryConfig(base_delay_s=1.0, backoff_factor=3.0, jitter=False)
    rp = RetryPolicy(cfg)
    assert rp.delay_for_attempt(1) == pytest.approx(3.0)


def test_delay_clamped_to_max():
    cfg = RetryConfig(base_delay_s=1.0, backoff_factor=10.0, max_delay_s=5.0, jitter=False)
    rp = RetryPolicy(cfg)
    assert rp.delay_for_attempt(10) == pytest.approx(5.0)


def test_delay_jitter_disabled_exact():
    cfg = RetryConfig(base_delay_s=1.0, backoff_factor=2.0, jitter=False)
    rp = RetryPolicy(cfg)
    d = rp.delay_for_attempt(2)
    assert d == pytest.approx(4.0)


def test_delay_jitter_enabled_in_range():
    cfg = RetryConfig(base_delay_s=10.0, backoff_factor=1.0, jitter=True)
    rp = RetryPolicy(cfg)
    for _ in range(20):
        d = rp.delay_for_attempt(0)
        assert 5.0 <= d <= 10.0


def test_simulate_returns_correct_length():
    rp = RetryPolicy(RetryConfig(jitter=False))
    delays = rp.simulate(4)
    assert len(delays) == 4


def test_simulate_no_actual_sleep():
    rp = RetryPolicy(RetryConfig(base_delay_s=1.0, backoff_factor=2.0, jitter=False))
    delays = rp.simulate(3)
    assert delays == pytest.approx([1.0, 2.0, 4.0])


def test_execute_success_no_retry():
    rp = RetryPolicy(RetryConfig(max_retries=3))
    result = rp.execute(lambda: None, sleep_fn=_no_sleep)
    assert result.success is True
    assert result.attempts == 1


def test_execute_success_returns_result():
    rp = RetryPolicy()
    result = rp.execute(lambda: 42, sleep_fn=_no_sleep)
    assert isinstance(result, RetryResult)
    assert result.success is True


def test_execute_all_failures():
    cfg = RetryConfig(max_retries=2, jitter=False)
    rp = RetryPolicy(cfg)

    def always_fail():
        raise ValueError("boom")

    result = rp.execute(always_fail, sleep_fn=_no_sleep)
    assert result.success is False
    assert result.attempts == 3
    assert "boom" in result.last_error


def test_execute_retries_on_exception():
    cfg = RetryConfig(max_retries=3, jitter=False)
    rp = RetryPolicy(cfg)
    call_count = [0]

    def fail_twice():
        call_count[0] += 1
        if call_count[0] < 3:
            raise RuntimeError("not yet")

    result = rp.execute(fail_twice, sleep_fn=_no_sleep)
    assert result.success is True
    assert result.attempts == 3


def test_on_retry_called():
    cfg = RetryConfig(max_retries=2, jitter=False)
    rp = RetryPolicy(cfg)
    retry_log = []

    def always_fail():
        raise ValueError("err")

    rp.execute(always_fail, on_retry=lambda a, e: retry_log.append(a), sleep_fn=_no_sleep)
    assert retry_log == [0, 1]


def test_total_delay_accumulated():
    cfg = RetryConfig(max_retries=2, base_delay_s=1.0, backoff_factor=2.0, jitter=False)
    rp = RetryPolicy(cfg)

    def always_fail():
        raise ValueError("x")

    result = rp.execute(always_fail, sleep_fn=_no_sleep)
    # delays: attempt 0 → 1.0, attempt 1 → 2.0
    assert result.total_delay_s == pytest.approx(3.0)


def test_sleep_fn_injected():
    cfg = RetryConfig(max_retries=1, jitter=False)
    rp = RetryPolicy(cfg)
    slept = []

    def always_fail():
        raise ValueError("x")

    rp.execute(always_fail, sleep_fn=lambda s: slept.append(s))
    assert len(slept) == 1


def test_retry_result_frozen():
    r = RetryResult(attempts=1, success=True, last_error="", total_delay_s=0.0)
    with pytest.raises(Exception):
        r.success = False  # type: ignore[misc]


def test_retry_config_frozen():
    cfg = RetryConfig()
    with pytest.raises(Exception):
        cfg.max_retries = 99  # type: ignore[misc]


def test_config_defaults():
    cfg = RetryConfig()
    assert cfg.max_retries == 3
    assert cfg.base_delay_s == 1.0
    assert cfg.max_delay_s == 30.0
    assert cfg.backoff_factor == 2.0
    assert cfg.jitter is True


def test_execute_max_retries_zero():
    cfg = RetryConfig(max_retries=0)
    rp = RetryPolicy(cfg)
    calls = [0]

    def fail():
        calls[0] += 1
        raise ValueError("x")

    result = rp.execute(fail, sleep_fn=_no_sleep)
    assert result.attempts == 1
    assert result.success is False


def test_execute_success_total_delay_zero():
    rp = RetryPolicy()
    result = rp.execute(lambda: None, sleep_fn=_no_sleep)
    assert result.total_delay_s == pytest.approx(0.0)


def test_simulate_empty():
    rp = RetryPolicy()
    assert rp.simulate(0) == []


def test_on_retry_not_called_on_success():
    rp = RetryPolicy()
    called = []
    rp.execute(lambda: None, on_retry=lambda a, e: called.append(a), sleep_fn=_no_sleep)
    assert called == []


def test_last_error_captured():
    cfg = RetryConfig(max_retries=0)
    rp = RetryPolicy(cfg)

    def fail():
        raise TypeError("type_err")

    result = rp.execute(fail, sleep_fn=_no_sleep)
    assert "type_err" in result.last_error


def test_last_error_empty_on_success():
    rp = RetryPolicy()
    result = rp.execute(lambda: None, sleep_fn=_no_sleep)
    assert result.last_error == ""
