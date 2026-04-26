"""
retry_workflow.py — Retry-aware workflow step executor.
Stdlib-only. Exports RETRY_WORKFLOW_REGISTRY.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass


class RetryStrategy(enum.Enum):
    IMMEDIATE = "immediate"
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"


@dataclass(frozen=True)
class StepResult:
    step_name: str
    success: bool
    attempts: int
    output: object
    error: str = ""


class RetryWorkflow:
    """Retry-aware workflow step executor."""

    def __init__(
        self,
        max_retries: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay_s: float = 1.0,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.strategy = strategy
        self.base_delay_s = base_delay_s
        # Default is a no-op lambda so tests run instantly.
        self.sleep_fn: Callable[[float], None] = (
            sleep_fn if sleep_fn is not None else lambda x: None
        )

    def delay_for_attempt(self, attempt: int) -> float:
        """Return the delay in seconds before retrying after *attempt* failures."""
        if self.strategy is RetryStrategy.IMMEDIATE:
            return 0.0
        if self.strategy is RetryStrategy.FIXED_DELAY:
            return self.base_delay_s
        # EXPONENTIAL_BACKOFF
        return self.base_delay_s * (2**attempt)

    def run_step(self, name: str, fn: Callable[[], object]) -> StepResult:
        """
        Try fn() up to max_retries+1 times.
        Sleep between attempts using sleep_fn(delay_for_attempt(attempt)).
        Returns a StepResult capturing outcome.
        """
        last_error: str = ""
        last_output: object = None
        total_attempts = self.max_retries + 1

        for attempt in range(total_attempts):
            try:
                result = fn()
                return StepResult(
                    step_name=name,
                    success=True,
                    attempts=attempt + 1,
                    output=result,
                    error="",
                )
            except Exception as exc:
                last_error = str(exc)
                last_output = None
                # Sleep before next retry (not after the last attempt)
                if attempt < total_attempts - 1:
                    delay = self.delay_for_attempt(attempt)
                    self.sleep_fn(delay)

        return StepResult(
            step_name=name,
            success=False,
            attempts=total_attempts,
            output=last_output,
            error=last_error,
        )

    def run_pipeline(self, steps: list[tuple[str, Callable[[], object]]]) -> list[StepResult]:
        """
        Run steps sequentially. Stop on the first failure.
        Returns the list of StepResults executed so far.
        """
        results: list[StepResult] = []
        for name, fn in steps:
            step_result = self.run_step(name, fn)
            results.append(step_result)
            if not step_result.success:
                break
        return results


# Public registry
RETRY_WORKFLOW_REGISTRY: dict = {"default": RetryWorkflow}
