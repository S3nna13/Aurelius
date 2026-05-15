"""Parallel fan-out dispatch abstraction for Aurelius agent tasks.

A ``DispatchTask`` represents a batch of independent LLM calls (one per
input item) that can be executed in parallel. The ``Dispatcher`` runs
the task via a thread pool with per-call timeouts, classifies failures,
and returns a structured ``DispatchReport``.

Pure stdlib; no foreign imports. Inspired by gadievron/raptor's
``packages/llm_analysis/dispatch.py`` but reimplemented from scratch
with Aurelius-native typing and failure-threshold circuit breaker.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

# Canonical status strings. ``success`` is used for the happy path; the
# others are error classifications derived from the raised exception or
# returned message.
_STATUS_VALUES = ("blocked", "auth", "timeout", "quota", "error", "success")


# Substring/regex patterns used by ``classify_error``. Order matters:
# the first matching pattern wins, so more specific ones come first.
_ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("auth", re.compile(r"\b(401|403|unauthorized|forbidden)\b", re.IGNORECASE)),
    ("quota", re.compile(r"\b(quota|rate[\s_-]?limit|429|too many requests)\b", re.IGNORECASE)),
    ("timeout", re.compile(r"\b(timed[\s_-]?out|timeout|deadline exceeded)\b", re.IGNORECASE)),
    ("blocked", re.compile(r"\b(blocked|content filter|safety|refus(?:ed|al))\b", re.IGNORECASE)),
)


def classify_error(exc_or_msg: Any) -> str:
    """Classify an exception or error string into a dispatch status.

    Returns one of ``blocked | auth | timeout | quota | error | success``.
    A ``None`` or empty input returns ``success``.
    """
    if exc_or_msg is None:
        return "success"
    if isinstance(exc_or_msg, TimeoutError):
        return "timeout"
    if isinstance(exc_or_msg, BaseException):
        text = f"{type(exc_or_msg).__name__}: {exc_or_msg}"
    else:
        text = str(exc_or_msg)
    if not text.strip():
        return "success"
    for status, pattern in _ERROR_PATTERNS:
        if pattern.search(text):
            return status
    return "error"


@dataclass
class DispatchOutcome:
    """Result of a single fan-out call."""

    input_item: Any
    raw_result: Any
    processed: Any
    status: str
    duration_s: float
    error_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_item": self.input_item,
            "raw_result": self.raw_result,
            "processed": self.processed,
            "status": self.status,
            "duration_s": self.duration_s,
            "error_class": self.error_class,
        }


@dataclass
class DispatchReport:
    """Aggregate report produced by ``Dispatcher.dispatch``."""

    task_name: str
    outcomes: list[DispatchOutcome] = field(default_factory=list)
    total_duration_s: float = 0.0
    status_counts: dict[str, int] = field(default_factory=dict)
    finalized: Any = None
    circuit_open: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "total_duration_s": self.total_duration_s,
            "status_counts": dict(self.status_counts),
            "finalized": self.finalized,
            "circuit_open": self.circuit_open,
        }


class DispatchTask(ABC):
    """Abstract base class for parallel fan-out LLM tasks."""

    name: str = "dispatch_task"

    @abstractmethod
    def build_prompt(self, input_item: Any) -> str:
        """Construct the LLM prompt for a single input item."""

    def get_schema(self) -> dict | None:
        """Optional structured-output schema. ``None`` = freeform text."""
        return None

    def validate_input(self, item: Any) -> bool:
        """Return True if ``item`` is acceptable. Default: not None."""
        return item is not None

    @abstractmethod
    def process_result(self, input_item: Any, result: Any) -> Any:
        """Transform the raw LLM output into a structured record."""

    @abstractmethod
    def finalize(self, processed_results: list[Any]) -> Any:
        """Combine per-item processed results into a final artifact."""


class Dispatcher:
    """Runs a ``DispatchTask`` over a list of inputs in parallel."""

    def __init__(
        self,
        max_workers: int = 4,
        per_task_timeout_s: float = 30.0,
        failure_threshold: float = 0.5,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if per_task_timeout_s <= 0:
            raise ValueError("per_task_timeout_s must be > 0")
        if not (0.0 < failure_threshold <= 1.0):
            raise ValueError("failure_threshold must be in (0, 1]")
        self.max_workers = max_workers
        self.per_task_timeout_s = per_task_timeout_s
        self.failure_threshold = failure_threshold

    # ------------------------------------------------------------------
    def _run_one(
        self,
        task: DispatchTask,
        item: Any,
        llm_fn: Callable[[str, dict | None], str],
    ) -> DispatchOutcome:
        start = time.monotonic()
        if not task.validate_input(item):
            return DispatchOutcome(
                input_item=item,
                raw_result=None,
                processed=None,
                status="error",
                duration_s=time.monotonic() - start,
                error_class="ValidationError",
            )
        prompt = task.build_prompt(item)
        schema = task.get_schema()
        try:
            raw = llm_fn(prompt, schema)
        except BaseException as exc:  # pragma: no cover - defensive
            status = classify_error(exc)
            return DispatchOutcome(
                input_item=item,
                raw_result=None,
                processed=None,
                status=status,
                duration_s=time.monotonic() - start,
                error_class=type(exc).__name__,
            )
        try:
            processed = task.process_result(item, raw)
        except BaseException as exc:
            return DispatchOutcome(
                input_item=item,
                raw_result=raw,
                processed=None,
                status=classify_error(exc),
                duration_s=time.monotonic() - start,
                error_class=type(exc).__name__,
            )
        return DispatchOutcome(
            input_item=item,
            raw_result=raw,
            processed=processed,
            status="success",
            duration_s=time.monotonic() - start,
            error_class=None,
        )

    # ------------------------------------------------------------------
    def dispatch(
        self,
        task: DispatchTask,
        inputs: list,
        llm_fn: Callable[[str, dict | None], str],
    ) -> DispatchReport:
        report = DispatchReport(task_name=task.name)
        if not inputs:
            report.finalized = task.finalize([])
            report.status_counts = {}
            return report

        t0 = time.monotonic()
        outcomes: list[DispatchOutcome | None] = [None] * len(inputs)
        midpoint = max(1, len(inputs) // 2)
        circuit_open = False

        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            future_to_idx: dict[Future, int] = {}
            for idx, item in enumerate(inputs):
                fut = executor.submit(self._run_one, task, item, llm_fn)
                future_to_idx[fut] = idx

            pending = set(future_to_idx.keys())
            completed_count = 0
            failures = 0
            deadline_base = time.monotonic()

            while pending:
                elapsed = time.monotonic() - deadline_base
                remaining_timeout = max(0.0, self.per_task_timeout_s - elapsed)
                done, pending = wait(
                    pending,
                    timeout=remaining_timeout if pending else 0.0,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    # Timeout: mark all still-pending futures as timed out.
                    for fut in pending:
                        idx = future_to_idx[fut]
                        outcomes[idx] = DispatchOutcome(
                            input_item=inputs[idx],
                            raw_result=None,
                            processed=None,
                            status="timeout",
                            duration_s=self.per_task_timeout_s,
                            error_class="TimeoutError",
                        )
                        fut.cancel()
                    pending = set()
                    break

                for fut in done:
                    idx = future_to_idx[fut]
                    try:
                        outcome = fut.result()
                    except BaseException as exc:  # pragma: no cover
                        outcome = DispatchOutcome(
                            input_item=inputs[idx],
                            raw_result=None,
                            processed=None,
                            status=classify_error(exc),
                            duration_s=0.0,
                            error_class=type(exc).__name__,
                        )
                    outcomes[idx] = outcome
                    completed_count += 1
                    if outcome.status != "success":
                        failures += 1

                # Circuit-breaker: at midpoint, if failure rate too high, abort.
                # Require at least 3 completions for a statistically meaningful
                # sample, and require there still be pending work to abort.
                if (
                    not circuit_open
                    and completed_count >= midpoint
                    and completed_count >= 3
                    and pending
                    and (failures / completed_count) > self.failure_threshold
                ):
                    circuit_open = True
                    for fut in pending:
                        idx = future_to_idx[fut]
                        outcomes[idx] = DispatchOutcome(
                            input_item=inputs[idx],
                            raw_result=None,
                            processed=None,
                            status="error",
                            duration_s=0.0,
                            error_class="CircuitOpen",
                        )
                        fut.cancel()
                    pending = set()
                    break
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        # Fill any holes defensively (should not occur).
        for idx, out in enumerate(outcomes):
            if out is None:
                outcomes[idx] = DispatchOutcome(
                    input_item=inputs[idx],
                    raw_result=None,
                    processed=None,
                    status="error",
                    duration_s=0.0,
                    error_class="Unknown",
                )

        final_outcomes: list[DispatchOutcome] = [o for o in outcomes if o is not None]
        status_counts: dict[str, int] = {}
        for o in final_outcomes:
            status_counts[o.status] = status_counts.get(o.status, 0) + 1

        processed_only = [o.processed for o in final_outcomes if o.status == "success"]
        finalized = task.finalize(processed_only)

        report.outcomes = final_outcomes
        report.total_duration_s = time.monotonic() - t0
        report.status_counts = status_counts
        report.finalized = finalized
        report.circuit_open = circuit_open
        return report


__all__ = [
    "DispatchOutcome",
    "DispatchReport",
    "DispatchTask",
    "Dispatcher",
    "classify_error",
]
