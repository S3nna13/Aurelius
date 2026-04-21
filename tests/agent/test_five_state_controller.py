"""Tests for the five-state agent lifecycle controller."""

from __future__ import annotations

import itertools
import re
import threading
import time

import pytest

from src.agent.five_state_controller import (
    AgentState,
    ControlContext,
    ControllerEvent,
    FiveStateController,
)


# ---------------------------------------------------------------------------
# Helper backends
# ---------------------------------------------------------------------------

def _echo_backend(chunks):
    def backend(messages, control: ControlContext):
        for c in chunks:
            control.wait_while_paused()
            if control.should_stop():
                return
            yield c

    return backend


def _instruction_aware_backend(seen: list[str]):
    """Emits 'a', records any pending_instruction, emits 'b'."""

    def backend(messages, control: ControlContext):
        yield "a"
        instr = control.pending_instruction()
        if instr is not None:
            seen.append(instr)
            yield f"[instr:{instr}]"
        yield "b"

    return backend


def _pausing_backend(barrier: threading.Event, chunks):
    def backend(messages, control: ControlContext):
        for c in chunks:
            yield c
            # Signal the test thread after first chunk.
            barrier.set()
            # Give the test thread a window to pause.
            time.sleep(0.02)
            control.wait_while_paused()
            if control.should_stop():
                return

    return backend


def _fake_clock():
    counter = itertools.count()
    return lambda: float(next(counter))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_initial_state_is_idle():
    ctrl = FiveStateController(_echo_backend(["x"]))
    assert ctrl.state is AgentState.IDLE


def test_start_transitions_to_running_then_completed():
    ctrl = FiveStateController(_echo_backend(["x", "y"]))
    ctrl.start([])
    out = ctrl.run_to_completion(timeout=2.0)
    assert out == "xy"
    assert ctrl.state is AgentState.COMPLETED


def test_double_start_raises():
    ctrl = FiveStateController(_echo_backend(["x"]))
    ctrl.start([])
    ctrl.run_to_completion(timeout=2.0)
    with pytest.raises(RuntimeError):
        ctrl.start([])


def test_pause_when_not_running_raises():
    ctrl = FiveStateController(_echo_backend(["x"]))
    with pytest.raises(RuntimeError):
        ctrl.pause()


def test_resume_when_not_paused_raises():
    ctrl = FiveStateController(_echo_backend(["x"]))
    with pytest.raises(RuntimeError):
        ctrl.resume()


def test_pause_and_resume_cycle():
    barrier = threading.Event()
    ctrl = FiveStateController(_pausing_backend(barrier, ["1", "2", "3"]))
    ctrl.start([])
    # Wait for first chunk.
    assert barrier.wait(timeout=2.0)
    ctrl.pause()
    assert ctrl.state is AgentState.PAUSED
    # Give the backend time to observe pause.
    time.sleep(0.05)
    ctrl.resume()
    assert ctrl.state is AgentState.RUNNING
    out = ctrl.run_to_completion(timeout=2.0)
    assert out == "123"
    assert ctrl.state is AgentState.COMPLETED


def test_stop_transitions_to_completed():
    # Long backend that we terminate early.
    def backend(messages, control):
        for i in range(1000):
            if control.should_stop():
                return
            yield f"{i},"
            time.sleep(0.001)

    ctrl = FiveStateController(backend)
    ctrl.start([])
    time.sleep(0.02)
    ctrl.stop()
    ctrl.run_to_completion(timeout=2.0)
    assert ctrl.state is AgentState.COMPLETED
    # should_stop was observed.
    assert any(e.event_type == "stop_requested" for e in ctrl.events)


def test_inject_instruction_visible_to_backend():
    seen: list[str] = []
    ctrl = FiveStateController(_instruction_aware_backend(seen))
    ctrl.inject_instruction("hello")
    ctrl.start([])
    out = ctrl.run_to_completion(timeout=2.0)
    assert seen == ["hello"]
    assert "[instr:hello]" in out


def test_terminal_matcher_triggers_completion():
    ctrl = FiveStateController(
        _echo_backend(["thinking...", "<DONE>", "should_not_see"]),
        terminal_matchers=[re.compile(r"<DONE>")],
    )
    ctrl.start([])
    ctrl.run_to_completion(timeout=2.0)
    assert ctrl.state is AgentState.COMPLETED
    # Completion event carries success=True from terminal match.
    transitions = [e for e in ctrl.events if e.event_type == "state_transition"]
    last = transitions[-1]
    assert last.payload["to"] == AgentState.COMPLETED.value
    assert last.payload.get("success") is True
    assert last.payload.get("reason") == "terminal_match"


def test_backend_exception_sets_error_and_reraises():
    def bad(messages, control):
        yield "ok"
        raise ValueError("boom")

    ctrl = FiveStateController(bad)
    ctrl.start([])
    with pytest.raises(ValueError, match="boom"):
        ctrl.run_to_completion(timeout=2.0)
    assert ctrl.state is AgentState.ERROR
    assert any(e.event_type == "exception" for e in ctrl.events)


def test_run_to_completion_returns_concatenated_output():
    ctrl = FiveStateController(_echo_backend(["foo", "bar", "baz"]))
    ctrl.start([])
    assert ctrl.run_to_completion(timeout=2.0) == "foobarbaz"


def test_events_log_populated():
    ctrl = FiveStateController(_echo_backend(["x"]))
    ctrl.start([])
    ctrl.run_to_completion(timeout=2.0)
    types = [e.event_type for e in ctrl.events]
    assert "start" in types
    assert "state_transition" in types
    assert "chunk" in types
    # All events are ControllerEvent instances.
    assert all(isinstance(e, ControllerEvent) for e in ctrl.events)


def test_thread_safe_concurrent_pause_from_another_thread():
    barrier = threading.Event()
    ctrl = FiveStateController(_pausing_backend(barrier, ["a", "b", "c", "d"]))
    ctrl.start([])
    assert barrier.wait(timeout=2.0)

    results: list[str] = []

    def pauser():
        try:
            ctrl.pause()
            results.append("paused")
        except RuntimeError:
            results.append("failed")

    t = threading.Thread(target=pauser)
    t.start()
    t.join(timeout=2.0)
    assert results == ["paused"]
    assert ctrl.state is AgentState.PAUSED
    ctrl.resume()
    ctrl.run_to_completion(timeout=2.0)


def test_time_fn_injectable_for_determinism():
    clock = _fake_clock()
    ctrl = FiveStateController(_echo_backend(["x", "y"]), time_fn=clock)
    ctrl.start([])
    ctrl.run_to_completion(timeout=2.0)
    # All event timestamps are strictly non-decreasing integers from our clock.
    ts = [e.at_time for e in ctrl.events]
    assert ts == sorted(ts)
    assert all(float(t).is_integer() for t in ts)


def test_stub_backend_deterministic_output():
    ctrl1 = FiveStateController(_echo_backend(["p", "q", "r"]))
    ctrl2 = FiveStateController(_echo_backend(["p", "q", "r"]))
    ctrl1.start([])
    ctrl2.start([])
    assert ctrl1.run_to_completion(timeout=2.0) == ctrl2.run_to_completion(timeout=2.0) == "pqr"


def test_agent_state_enum_has_five_members():
    assert {s.name for s in AgentState} == {
        "IDLE",
        "RUNNING",
        "PAUSED",
        "COMPLETED",
        "ERROR",
    }


def test_run_to_completion_before_start_raises():
    ctrl = FiveStateController(_echo_backend(["x"]))
    with pytest.raises(RuntimeError):
        ctrl.run_to_completion(timeout=0.1)
