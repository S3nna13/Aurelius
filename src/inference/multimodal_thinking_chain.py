"""Multimodal Thinking Chain — interleaved think/tool/vision token builder.

Inspired by Kimi K2.6's interleaved <think> reasoning tokens with tool_call
invocations and vision tokens during generation. The chain builder tracks the
step sequence and enforces step limits (50 steps for vision, 98,304 thinking
tokens total).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class StepType(Enum):
    THINK = "think"              # reasoning tokens
    TOOL_CALL = "tool_call"      # tool invocation (text)
    TOOL_RESULT = "tool_result"  # tool response tokens
    VISION = "vision"            # vision feature tokens
    TEXT = "text"                # regular output text


@dataclass
class ChainStep:
    step_idx: int
    step_type: StepType
    tokens: list[int]           # token IDs for this step
    token_count: int = 0        # auto-computed from len(tokens)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.token_count = len(self.tokens)


@dataclass
class MultimodalThinkingConfig:
    max_steps: int = 50                # max steps (vision mode limit)
    max_thinking_tokens: int = 98304   # total THINK tokens across all steps
    max_tokens_per_step: int = 4096    # per-step token limit
    vision_step_limit: int = 50        # separate limit on VISION steps
    allow_interleave: bool = True      # if False, thinking must come before all tool calls


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class StepLimitError(ValueError):
    """Raised when adding a step would exceed max_steps."""


class ThinkingBudgetError(ValueError):
    """Raised when adding THINK tokens would exceed max_thinking_tokens."""


class VisionStepLimitError(ValueError):
    """Raised when adding a VISION step would exceed vision_step_limit."""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MultimodalThinkingChain:
    """Builds and manages an interleaved multimodal thinking chain.

    Tracks the step sequence and enforces:
    - max_steps: total number of steps
    - max_thinking_tokens: total THINK tokens across all steps
    - vision_step_limit: total number of VISION steps
    - max_tokens_per_step: per-step token limit (truncates silently)
    """

    def __init__(self, config: Optional[MultimodalThinkingConfig] = None) -> None:
        self.config = config if config is not None else MultimodalThinkingConfig()
        self._steps: list[ChainStep] = []

    # ------------------------------------------------------------------
    # Core mutation
    # ------------------------------------------------------------------

    def add_step(
        self,
        step_type: StepType,
        tokens: list[int],
        metadata: Optional[dict] = None,
    ) -> ChainStep:
        """Add a step to the chain.

        Raises:
            StepLimitError: if max_steps would be exceeded.
            ThinkingBudgetError: if THINK tokens would exceed max_thinking_tokens.
            VisionStepLimitError: if VISION steps would exceed vision_step_limit.

        Tokens are silently truncated to max_tokens_per_step if too long.
        Returns the created ChainStep.
        """
        cfg = self.config

        # 1. Check step count limit before adding
        if len(self._steps) >= cfg.max_steps:
            raise StepLimitError(
                f"Cannot add step: max_steps={cfg.max_steps} already reached."
            )

        # 2. Truncate tokens to per-step limit
        if len(tokens) > cfg.max_tokens_per_step:
            tokens = tokens[: cfg.max_tokens_per_step]

        # 3. Budget checks AFTER truncation (use actual tokens that would be added)
        if step_type is StepType.THINK:
            new_total = self.thinking_tokens_used() + len(tokens)
            if new_total > cfg.max_thinking_tokens:
                raise ThinkingBudgetError(
                    f"Cannot add THINK step: would use {new_total} thinking tokens "
                    f"(budget={cfg.max_thinking_tokens})."
                )

        if step_type is StepType.VISION:
            if self.vision_steps_used() >= cfg.vision_step_limit:
                raise VisionStepLimitError(
                    f"Cannot add VISION step: vision_step_limit={cfg.vision_step_limit} already reached."
                )

        # 4. Create and store the step
        step = ChainStep(
            step_idx=len(self._steps),
            step_type=step_type,
            tokens=list(tokens),
            metadata=metadata if metadata is not None else {},
        )
        self._steps.append(step)
        return step

    # ------------------------------------------------------------------
    # Accessors / metrics
    # ------------------------------------------------------------------

    def get_flat_tokens(self) -> list[int]:
        """Return all tokens concatenated in step order."""
        result: list[int] = []
        for step in self._steps:
            result.extend(step.tokens)
        return result

    def thinking_tokens_used(self) -> int:
        """Total THINK tokens across all steps."""
        return sum(s.token_count for s in self._steps if s.step_type is StepType.THINK)

    def vision_steps_used(self) -> int:
        """Count of VISION steps."""
        return sum(1 for s in self._steps if s.step_type is StepType.VISION)

    def step_count(self) -> int:
        """Total number of steps in the chain."""
        return len(self._steps)

    def budget_remaining(self) -> dict:
        """Return remaining budget.

        Returns:
            {"steps": int, "thinking_tokens": int, "vision_steps": int}
        """
        cfg = self.config
        return {
            "steps": cfg.max_steps - len(self._steps),
            "thinking_tokens": cfg.max_thinking_tokens - self.thinking_tokens_used(),
            "vision_steps": cfg.vision_step_limit - self.vision_steps_used(),
        }

    def validate_interleave(self) -> bool:
        """Validate interleave constraint.

        If allow_interleave=False: returns False if any TOOL_CALL comes before
        the last THINK step (i.e., thinking must precede all tool calls).
        If allow_interleave=True: always returns True.
        """
        if self.config.allow_interleave:
            return True

        # Find indices of THINK and TOOL_CALL steps
        last_think_idx: int = -1
        first_tool_call_idx: int = -1

        for step in self._steps:
            if step.step_type is StepType.THINK:
                last_think_idx = step.step_idx
            elif step.step_type is StepType.TOOL_CALL:
                if first_tool_call_idx == -1:
                    first_tool_call_idx = step.step_idx

        # If there are no tool calls, always valid
        if first_tool_call_idx == -1:
            return True

        # Tool call must come AFTER all thinks (i.e., first tool call > last think)
        # If there are no thinks, tool calls are fine (no thinks to be violated)
        if last_think_idx == -1:
            return True

        # Invalid if the first tool call appears BEFORE the last think
        return first_tool_call_idx > last_think_idx

    def to_summary(self) -> dict:
        """Return a summary of the chain.

        Returns:
            {
                "total_steps": int,
                "by_type": {type_name: count},
                "total_tokens": int,
                "thinking_tokens": int,
            }
        """
        by_type: dict[str, int] = {}
        for step in self._steps:
            key = step.step_type.value
            by_type[key] = by_type.get(key, 0) + 1

        return {
            "total_steps": len(self._steps),
            "by_type": by_type,
            "total_tokens": sum(s.token_count for s in self._steps),
            "thinking_tokens": self.thinking_tokens_used(),
        }
