"""Skill composition and pipeline execution for Aurelius.

Provides sequential and parallel skill execution pipelines with context
passing between steps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.agent.skill_executor import (
    DEFAULT_SKILL_EXECUTOR,
    ExecutionResult,
    SkillContext,
    SkillExecutionError,
    SkillExecutor,
)


class SkillCompositionError(Exception):
    """Raised when pipeline composition or execution fails."""


@dataclass
class CompositionStep:
    """A single step in a skill composition pipeline."""

    skill_id: str
    instructions: str
    output_key: str = "output"


@dataclass
class CompositionResult:
    """Result of executing a composition pipeline."""

    outputs: dict[str, ExecutionResult] = field(default_factory=dict)
    overall_success: bool = True
    total_duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillComposer:
    """Composes multiple skills into sequential or parallel pipelines."""

    executor: SkillExecutor | None = None

    def __post_init__(self) -> None:
        if self.executor is None:
            self.executor = DEFAULT_SKILL_EXECUTOR

    def execute_pipeline(
        self,
        steps: list[CompositionStep],
        initial_context: SkillContext | None = None,
    ) -> CompositionResult:
        """Execute *steps* sequentially, passing outputs between steps.

        Each step's ``ExecutionResult.output`` is injected into the next
        step's context variables under the step's *output_key*.

        Fail-fast: stops at the first failed step.
        """
        if not isinstance(steps, list):
            raise TypeError(f"steps must be a list, got {type(steps).__name__}")
        if not steps:
            raise SkillCompositionError("steps must not be empty")

        context = initial_context if initial_context is not None else SkillContext()
        outputs: dict[str, ExecutionResult] = {}
        overall_success = True
        start = time.perf_counter()

        for step in steps:
            try:
                result = self.executor.execute(
                    step.skill_id, step.instructions, context
                )
            except SkillExecutionError as exc:
                result = ExecutionResult(
                    output=str(exc),
                    success=False,
                    duration_ms=0.0,
                    metadata={"error": str(exc)},
                )

            outputs[step.output_key] = result

            if not result.success:
                overall_success = False
                break

            # Feed output into next step's context
            context.variables[step.output_key] = result.output

        total_duration_ms = (time.perf_counter() - start) * 1000

        return CompositionResult(
            outputs=outputs,
            overall_success=overall_success,
            total_duration_ms=total_duration_ms,
            metadata={"step_count": len(steps), "mode": "pipeline"},
        )

    def execute_parallel(
        self,
        steps: list[CompositionStep],
        initial_context: SkillContext | None = None,
    ) -> CompositionResult:
        """Execute *steps* independently with copies of *initial_context*.

        Outputs are NOT shared between parallel steps.  ``overall_success``
        is ``True`` only if every step succeeds.
        """
        if not isinstance(steps, list):
            raise TypeError(f"steps must be a list, got {type(steps).__name__}")
        if not steps:
            raise SkillCompositionError("steps must not be empty")

        outputs: dict[str, ExecutionResult] = {}
        overall_success = True
        start = time.perf_counter()

        for step in steps:
            # Each step gets its own copy of the initial context
            context = initial_context if initial_context is not None else SkillContext()

            try:
                result = self.executor.execute(
                    step.skill_id, step.instructions, context
                )
            except SkillExecutionError as exc:
                result = ExecutionResult(
                    output=str(exc),
                    success=False,
                    duration_ms=0.0,
                    metadata={"error": str(exc)},
                )

            outputs[step.output_key] = result

            if not result.success:
                overall_success = False

        total_duration_ms = (time.perf_counter() - start) * 1000

        return CompositionResult(
            outputs=outputs,
            overall_success=overall_success,
            total_duration_ms=total_duration_ms,
            metadata={"step_count": len(steps), "mode": "parallel"},
        )


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

DEFAULT_SKILL_COMPOSER: SkillComposer = SkillComposer()

SKILL_COMPOSER_REGISTRY: dict[str, SkillComposer] = {
    "default": DEFAULT_SKILL_COMPOSER,
}


__all__ = [
    "DEFAULT_SKILL_COMPOSER",
    "SKILL_COMPOSER_REGISTRY",
    "CompositionResult",
    "CompositionStep",
    "SkillComposer",
    "SkillCompositionError",
]
