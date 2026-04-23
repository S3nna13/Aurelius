"""Tool orchestrator: parallel tool dispatch, result aggregation, retry logic."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict
    max_retries: int = 2
    id: str = field(default_factory=_short_uuid)


@dataclass
class ToolCallOutcome:
    call_id: str
    tool_name: str
    success: bool
    output: str
    attempts: int = 1
    error: str = ""


class ToolOrchestrator:
    """Dispatch tool calls with retry logic and aggregate outcomes."""

    def __init__(self, tool_registry=None) -> None:
        self._tool_registry = tool_registry

    def build_call(self, tool_name: str, **arguments) -> ToolCall:
        return ToolCall(tool_name=tool_name, arguments=arguments)

    def dispatch(self, call: ToolCall, handler: Callable) -> ToolCallOutcome:
        """Invoke handler(**call.arguments), retrying up to max_retries on exception."""
        max_attempts = call.max_retries + 1
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            try:
                result = handler(**call.arguments)
                return ToolCallOutcome(
                    call_id=call.id,
                    tool_name=call.tool_name,
                    success=True,
                    output=str(result) if result is not None else "",
                    attempts=attempt,
                    error="",
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        return ToolCallOutcome(
            call_id=call.id,
            tool_name=call.tool_name,
            success=False,
            output="",
            attempts=max_attempts,
            error=last_error,
        )

    def dispatch_batch(
        self,
        calls: list[ToolCall],
        handlers: dict[str, Callable],
    ) -> list[ToolCallOutcome]:
        """Dispatch each call sequentially; aggregate and return all outcomes."""
        outcomes: list[ToolCallOutcome] = []
        for call in calls:
            handler = handlers[call.tool_name]
            outcomes.append(self.dispatch(call, handler))
        return outcomes

    @staticmethod
    def success_rate(outcomes: list[ToolCallOutcome]) -> float:
        if not outcomes:
            return 1.0
        return sum(1 for o in outcomes if o.success) / len(outcomes)

    @staticmethod
    def failed_outcomes(outcomes: list[ToolCallOutcome]) -> list[ToolCallOutcome]:
        return [o for o in outcomes if not o.success]


TOOL_ORCHESTRATOR = ToolOrchestrator()
