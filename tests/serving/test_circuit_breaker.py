"""Unit tests for :mod:`src.serving.circuit_breaker`."""

from __future__ import annotations

import threading

import pytest

from src.serving.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    CircuitStateTransition,
)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _make(clock: FakeClock | None = None, **kw) -> CircuitBreaker:
    c = clock or FakeClock()
    params = dict(
        name="test",
        failure_threshold=3,
        recovery_timeout_s=10.0,
        probe_successes_required=2,
        backoff_multiplier=2.0,
        max_recovery_timeout_s=100.0,
        time_fn=c,
    )
    params.update(kw)
    return CircuitBreaker(**params)


def test_starts_closed() -> None:
    cb = _make()
    assert cb.state == CircuitState.CLOSED
    assert cb.total_calls == 0
    assert cb.successes == 0
    assert cb.failures == 0


def test_closed_to_open_after_threshold_failures() -> None:
    cb = _make()
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.failures == 3


def test_open_rejects_calls() -> None:
    cb = _make()
    cb.force_open()
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: 1)


def test_open_to_half_open_after_timeout() -> None:
    clk = FakeClock()
    cb = _make(clock=clk)
    cb.force_open()
    assert cb.state == CircuitState.OPEN
    clk.advance(9.9)
    assert cb.state == CircuitState.OPEN
    clk.advance(0.2)  # now > recovery_timeout_s (10.0)
    assert cb.state == CircuitState.HALF_OPEN


def test_half_open_successes_close_circuit() -> None:
    clk = FakeClock()
    cb = _make(clock=clk)
    cb.force_open()
    clk.advance(11.0)
    assert cb.state == CircuitState.HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitState.HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_half_open_failure_reopens_with_backoff() -> None:
    clk = FakeClock()
    cb = _make(clock=clk)
    cb.force_open()
    clk.advance(11.0)
    assert cb.state == CircuitState.HALF_OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    # backoff: 10s * 2.0 = 20s -- half the time should NOT reopen to half.
    clk.advance(15.0)
    assert cb.state == CircuitState.OPEN
    clk.advance(6.0)  # total 21s
    assert cb.state == CircuitState.HALF_OPEN


def test_backoff_capped_at_max() -> None:
    clk = FakeClock()
    cb = _make(clock=clk, recovery_timeout_s=10.0, max_recovery_timeout_s=25.0)
    # trip open repeatedly; backoffs: 10 -> 20 -> 40(capped 25) -> 25
    cb.force_open()
    for _ in range(5):
        # Advance past whatever timeout; must advance enough to move to half.
        clk.advance(1000.0)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
    # final current timeout should be capped
    assert cb._current_recovery_timeout_s <= 25.0
    assert cb._current_recovery_timeout_s == 25.0


def test_record_success_resets_closed_failure_count() -> None:
    cb = _make()
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    cb.record_failure()
    # Only 2 consecutive failures after reset -- threshold is 3, should stay closed.
    assert cb.state == CircuitState.CLOSED


def test_force_open_and_force_close() -> None:
    cb = _make()
    cb.force_open("manual")
    assert cb.state == CircuitState.OPEN
    cb.force_close("manual")
    assert cb.state == CircuitState.CLOSED
    # force_close resets backoff timeout.
    assert cb._current_recovery_timeout_s == cb.recovery_timeout_s


def test_thread_safety_under_concurrent_calls() -> None:
    cb = _make(failure_threshold=1000)  # keep closed under load

    def worker() -> None:
        for _ in range(200):
            try:
                cb.call(lambda: 1)
            except Exception:
                pass

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cb.total_calls == 8 * 200
    assert cb.successes == 8 * 200
    assert cb.state == CircuitState.CLOSED


def test_call_wrapper_returns_value() -> None:
    cb = _make()
    assert cb.call(lambda x, y: x + y, 2, 3) == 5
    assert cb.successes == 1
    assert cb.total_calls == 1


def test_call_wrapper_records_failure_on_exception() -> None:
    cb = _make()

    def boom() -> None:
        raise RuntimeError("nope")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            cb.call(boom)
    assert cb.state == CircuitState.OPEN
    assert cb.failures == 3


def test_context_manager_success_path() -> None:
    cb = _make()
    with cb:
        pass
    assert cb.successes == 1
    assert cb.total_calls == 1
    assert cb.state == CircuitState.CLOSED


def test_context_manager_exception_path() -> None:
    cb = _make()
    for _ in range(3):
        with pytest.raises(ValueError):
            with cb:
                raise ValueError("boom")
    assert cb.failures == 3
    assert cb.state == CircuitState.OPEN


def test_context_manager_raises_when_open() -> None:
    cb = _make()
    cb.force_open()
    with pytest.raises(CircuitOpenError):
        with cb:
            pass


def test_state_transitions_logged() -> None:
    clk = FakeClock()
    cb = _make(clock=clk)
    for _ in range(3):
        cb.record_failure()
    clk.advance(11.0)
    _ = cb.state  # triggers half-open transition
    cb.record_success()
    cb.record_success()  # closes
    kinds = [(t.from_state, t.to_state) for t in cb.state_transitions]
    assert (CircuitState.CLOSED, CircuitState.OPEN) in kinds
    assert (CircuitState.OPEN, CircuitState.HALF_OPEN) in kinds
    assert (CircuitState.HALF_OPEN, CircuitState.CLOSED) in kinds
    # All transitions have typed entries.
    for tr in cb.state_transitions:
        assert isinstance(tr, CircuitStateTransition)
        assert isinstance(tr.at_time, float)
        assert tr.reason


def test_determinism_with_injected_time_fn() -> None:
    clk1 = FakeClock(start=100.0)
    clk2 = FakeClock(start=100.0)
    cb1 = _make(clock=clk1)
    cb2 = _make(clock=clk2)
    for cb, clk in ((cb1, clk1), (cb2, clk2)):
        for _ in range(3):
            cb.record_failure()
        clk.advance(11.0)
        _ = cb.state
    assert [t.at_time for t in cb1.state_transitions] == [
        t.at_time for t in cb2.state_transitions
    ]
    assert [(t.from_state, t.to_state) for t in cb1.state_transitions] == [
        (t.from_state, t.to_state) for t in cb2.state_transitions
    ]


def test_invalid_params_rejected() -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(name="x", failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(name="x", probe_successes_required=0)
    with pytest.raises(ValueError):
        CircuitBreaker(name="x", recovery_timeout_s=0.0)
    with pytest.raises(ValueError):
        CircuitBreaker(name="x", backoff_multiplier=0.5)
    with pytest.raises(ValueError):
        CircuitBreaker(name="x", recovery_timeout_s=100.0, max_recovery_timeout_s=10.0)


def test_half_open_close_resets_backoff() -> None:
    clk = FakeClock()
    cb = _make(clock=clk)
    cb.force_open()
    clk.advance(11.0)
    # bump backoff by failing then recovering
    cb.record_failure()  # reopens with 20s backoff
    clk.advance(21.0)
    assert cb.state == CircuitState.HALF_OPEN
    cb.record_success()
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    # On successful close, recovery timeout resets to base.
    assert cb._current_recovery_timeout_s == cb.recovery_timeout_s
