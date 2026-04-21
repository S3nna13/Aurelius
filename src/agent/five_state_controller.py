"""PentestGPT-style five-state agent lifecycle controller.

This module provides a *lifecycle supervisor* that owns a finite-state
machine over ``IDLE / RUNNING / PAUSED / COMPLETED / ERROR`` and wraps
an arbitrary chunk-yielding backend. It is orthogonal to the ReAct
message-iteration loop (``src.agent.react_loop``): a controller can
drive any backend callable that respects the :class:`ControlContext`.

The controller is thread-safe for external actors (pause/resume/stop/
inject_instruction may be called from any thread). The backend itself
runs on whatever thread called :meth:`FiveStateController.start`.

Pure standard library only: ``threading, time, re, dataclasses, enum,
queue``. No foreign imports; additive within this file only.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator, Optional


class AgentState(str, Enum):
    """Lifecycle states for :class:`FiveStateController`."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


@dataclass
class ControllerEvent:
    """Audit-log entry describing a controller transition or signal."""

    event_type: str
    at_time: float
    payload: dict = field(default_factory=dict)


class ControlContext:
    """Cooperative control handle passed to backend callables.

    The backend is expected to poll :meth:`is_paused`, :meth:`should_stop`,
    and :meth:`pending_instruction` between chunks. It must not block
    indefinitely while paused; use :meth:`wait_while_paused` to yield.
    """

    def __init__(self, controller: "FiveStateController") -> None:
        self._c = controller

    def is_paused(self) -> bool:
        return self._c._pause_event.is_set()

    def should_stop(self) -> bool:
        return self._c._stop_event.is_set()

    def pending_instruction(self) -> Optional[str]:
        """Pop one pending injected instruction (FIFO), or None."""
        try:
            return self._c._instr_queue.get_nowait()
        except queue.Empty:
            return None

    def wait_while_paused(self, poll_interval: float = 0.01) -> None:
        """Block the backend while paused, returning when resumed/stopped."""
        while self._c._pause_event.is_set() and not self._c._stop_event.is_set():
            time.sleep(poll_interval)


class FiveStateController:
    """Lifecycle supervisor wrapping a chunk-yielding backend callable.

    Parameters
    ----------
    backend_fn:
        Callable ``(messages, control) -> Iterator[str]``. Yields response
        chunks. Receives a :class:`ControlContext` for cooperative pause/
        stop/instruction handling.
    terminal_matchers:
        Regex patterns; when any matches an emitted chunk the controller
        transitions to ``COMPLETED`` with ``success=True``.
    time_fn:
        Injectable clock for deterministic tests (default
        :func:`time.monotonic`).
    """

    def __init__(
        self,
        backend_fn: Callable[[list[dict], ControlContext], Iterator[str]],
        terminal_matchers: Optional[list[re.Pattern]] = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._backend_fn = backend_fn
        self._terminal_matchers: list[re.Pattern] = list(terminal_matchers or [])
        self._time_fn = time_fn

        self._state: AgentState = AgentState.IDLE
        self._state_lock = threading.RLock()

        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._instr_queue: "queue.Queue[str]" = queue.Queue()

        self._thread: Optional[threading.Thread] = None
        self._started_once = False

        self._chunks: list[str] = []
        self._output_lock = threading.Lock()
        self._exc: Optional[BaseException] = None

        self.events: list[ControllerEvent] = []

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    @property
    def state(self) -> AgentState:
        with self._state_lock:
            return self._state

    def _set_state(self, new: AgentState, **payload: object) -> None:
        with self._state_lock:
            old = self._state
            self._state = new
        self._record("state_transition", {"from": old.value, "to": new.value, **payload})

    def _record(self, event_type: str, payload: Optional[dict] = None) -> None:
        self.events.append(
            ControllerEvent(
                event_type=event_type,
                at_time=self._time_fn(),
                payload=dict(payload or {}),
            )
        )

    # ------------------------------------------------------------------
    # External controls (thread-safe)
    # ------------------------------------------------------------------
    def start(self, initial_messages: list[dict]) -> None:
        """Begin backend execution on a daemon thread."""
        with self._state_lock:
            if self._started_once:
                raise RuntimeError("FiveStateController.start() called twice")
            if self._state is not AgentState.IDLE:
                raise RuntimeError(
                    f"start() requires IDLE state, got {self._state.value}"
                )
            self._started_once = True
            self._state = AgentState.RUNNING
        self._record(
            "state_transition", {"from": AgentState.IDLE.value, "to": AgentState.RUNNING.value}
        )
        self._record("start", {"n_messages": len(initial_messages)})

        self._thread = threading.Thread(
            target=self._run, args=(initial_messages,), daemon=True
        )
        self._thread.start()

    def pause(self) -> None:
        with self._state_lock:
            if self._state is not AgentState.RUNNING:
                raise RuntimeError(
                    f"pause() requires RUNNING state, got {self._state.value}"
                )
            self._state = AgentState.PAUSED
        self._pause_event.set()
        self._record(
            "state_transition",
            {"from": AgentState.RUNNING.value, "to": AgentState.PAUSED.value},
        )

    def resume(self) -> None:
        with self._state_lock:
            if self._state is not AgentState.PAUSED:
                raise RuntimeError(
                    f"resume() requires PAUSED state, got {self._state.value}"
                )
            self._state = AgentState.RUNNING
        self._pause_event.clear()
        self._record(
            "state_transition",
            {"from": AgentState.PAUSED.value, "to": AgentState.RUNNING.value},
        )

    def stop(self) -> None:
        """Request cooperative termination. Transitions to COMPLETED on clean exit."""
        self._stop_event.set()
        # If paused, unblock the backend so it can observe stop.
        self._pause_event.clear()
        self._record("stop_requested", {})

    def inject_instruction(self, instr: str) -> None:
        """Queue an instruction visible to the next backend poll."""
        self._instr_queue.put(instr)
        self._record("instruction_injected", {"instruction": instr})

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _run(self, messages: list[dict]) -> None:
        control = ControlContext(self)
        try:
            iterator = self._backend_fn(list(messages), control)
            for chunk in iterator:
                if not isinstance(chunk, str):
                    chunk = str(chunk)
                with self._output_lock:
                    self._chunks.append(chunk)
                self._record("chunk", {"text": chunk})

                # Terminal-pattern detection.
                if self._matches_terminal(chunk):
                    self._set_state(AgentState.COMPLETED, reason="terminal_match", success=True)
                    return

                if self._stop_event.is_set():
                    break

            # Backend exhausted naturally.
            with self._state_lock:
                terminal = self._state in (AgentState.COMPLETED, AgentState.ERROR)
            if terminal:
                return
            if self._stop_event.is_set():
                self._set_state(AgentState.COMPLETED, reason="stop_requested", success=False)
            else:
                self._set_state(AgentState.COMPLETED, reason="backend_exhausted", success=True)
        except BaseException as exc:  # noqa: BLE001 — must capture all
            self._exc = exc
            self._set_state(AgentState.ERROR, error=repr(exc))
            self._record("exception", {"type": type(exc).__name__, "message": str(exc)})
            # Do not re-raise on the worker thread (would print to stderr);
            # run_to_completion() re-raises for the caller.
            return

    def _matches_terminal(self, chunk: str) -> bool:
        for pat in self._terminal_matchers:
            if pat.search(chunk):
                return True
        return False

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------
    def run_to_completion(self, timeout: Optional[float] = None) -> str:
        """Block until COMPLETED or ERROR; return concatenated output."""
        if self._thread is None:
            raise RuntimeError("run_to_completion() called before start()")
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("backend did not complete within timeout")
        if self._exc is not None:
            raise self._exc
        with self._output_lock:
            return "".join(self._chunks)

    @property
    def output(self) -> str:
        with self._output_lock:
            return "".join(self._chunks)


__all__ = [
    "AgentState",
    "ControllerEvent",
    "ControlContext",
    "FiveStateController",
]
