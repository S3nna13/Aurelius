"""Skill execution engine for Aurelius.

Handles simple variable substitution, special placeholders, and
execution-time tracking for skill instructions.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SkillExecutionError",
    "SkillContext",
    "ExecutionResult",
    "SkillExecutor",
    "DEFAULT_SKILL_EXECUTOR",
    "SKILL_EXECUTOR_REGISTRY",
]


_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


class SkillExecutionError(Exception):
    """Raised when a skill cannot be executed."""


@dataclass
class SkillContext:
    """Runtime context provided to a skill during execution."""

    variables: dict[str, str] = field(default_factory=dict)
    memory: list[str] = field(default_factory=list)
    available_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Outcome of a single skill execution."""

    output: str
    success: bool
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillExecutor:
    """Executes skill instructions with variable substitution."""

    _max_instructions_length: int = 100_000

    def execute(
        self,
        skill_id: str,
        instructions: str,
        context: SkillContext | None = None,
    ) -> ExecutionResult:
        """Run ``instructions`` through variable substitution and return result.

        Args:
            skill_id: Identifier of the skill being executed.
            instructions: Raw instruction text with optional ``{placeholders}``.
            context: Runtime context for substitution.  Defaults to empty.

        Returns:
            An :class:`ExecutionResult` with the substituted output.

        Raises:
            SkillExecutionError: If inputs are invalid or exceed limits.
        """
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise SkillExecutionError("skill_id must be a non-empty string")
        if not isinstance(instructions, str) or not instructions.strip():
            raise SkillExecutionError("instructions must be a non-empty string")
        if len(instructions) > self._max_instructions_length:
            raise SkillExecutionError(
                f"instructions exceed max length of {self._max_instructions_length}"
            )

        ctx = context if context is not None else SkillContext()

        start = time.perf_counter()
        output = self._substitute(instructions, ctx)
        duration_ms = (time.perf_counter() - start) * 1000.0

        return ExecutionResult(
            output=output,
            success=True,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _substitute(instructions: str, context: SkillContext) -> str:
        """Replace ``{placeholders}`` in *instructions* using *context*."""

        def _replacer(match: re.Match[str]) -> str:
            key = match.group(1)
            if key == "memory":
                return "\n".join(context.memory)
            if key == "tools":
                return ", ".join(context.available_tools)
            if key in context.variables:
                return context.variables[key]
            return match.group(0)

        return _PLACEHOLDER_RE.sub(_replacer, instructions)


DEFAULT_SKILL_EXECUTOR = SkillExecutor()

SKILL_EXECUTOR_REGISTRY: dict[str, SkillExecutor] = {
    "default": DEFAULT_SKILL_EXECUTOR,
}
