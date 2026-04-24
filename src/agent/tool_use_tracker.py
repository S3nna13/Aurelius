"""Tool-use tracker module for the Aurelius agent surface.

Tracks agent tool calls for analysis, budget control, and reporting.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCall:
    """Immutable record of a single tool invocation."""
    call_id:     str
    tool_name:   str
    input_size:  int
    output_size: int
    duration_ms: float
    success:     bool
    error:       str = ""

    @classmethod
    def create(
        cls,
        tool_name:   str,
        input_size:  int,
        output_size: int,
        duration_ms: float,
        success:     bool = True,
        error:       str  = "",
        call_id:     str | None = None,
    ) -> "ToolCall":
        """Factory that auto-generates call_id when not provided."""
        cid = call_id if call_id is not None else uuid.uuid4().hex[:8]
        return cls(
            call_id=cid,
            tool_name=tool_name,
            input_size=input_size,
            output_size=output_size,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )


@dataclass(frozen=True)
class ToolBudget:
    """Hard limits for tool usage within a session."""
    max_calls:           int   = 100
    max_total_duration_ms: float = 30_000.0


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ToolUseTracker:
    """Record tool calls and enforce an optional budget."""

    def __init__(self, budget: ToolBudget | None = None) -> None:
        self._budget: ToolBudget | None = budget
        self._calls:  list[ToolCall]    = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        tool_name:   str,
        input_size:  int,
        output_size: int,
        duration_ms: float,
        success:     bool = True,
        error:       str  = "",
    ) -> ToolCall:
        """Create and store a ToolCall record."""
        tc = ToolCall.create(
            tool_name=tool_name,
            input_size=input_size,
            output_size=output_size,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        self._calls.append(tc)
        return tc

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def calls_for(self, tool_name: str) -> list[ToolCall]:
        """Return all recorded calls for the given tool name."""
        return [c for c in self._calls if c.tool_name == tool_name]

    def total_calls(self) -> int:
        """Return the total number of recorded calls."""
        return len(self._calls)

    def total_duration_ms(self) -> float:
        """Return the sum of all call durations in milliseconds."""
        return sum(c.duration_ms for c in self._calls)

    def budget_status(self) -> dict[str, Any]:
        """Return current budget consumption and pass/fail flags."""
        used_calls    = self.total_calls()
        used_duration = self.total_duration_ms()

        if self._budget is None:
            calls_ok    = True
            duration_ok = True
        else:
            calls_ok    = used_calls    <= self._budget.max_calls
            duration_ok = used_duration <= self._budget.max_total_duration_ms

        return {
            "calls_ok":        calls_ok,
            "duration_ok":     duration_ok,
            "all_ok":          calls_ok and duration_ok,
            "calls_used":      used_calls,
            "duration_used_ms": used_duration,
        }

    def most_used(self, n: int = 5) -> list[tuple[str, int]]:
        """Return the top-n (tool_name, call_count) pairs sorted descending."""
        counter: Counter[str] = Counter(c.tool_name for c in self._calls)
        return counter.most_common(n)

    def success_rate(self, tool_name: str | None = None) -> float:
        """Return the fraction of successful calls (0.0 if no calls)."""
        if tool_name is not None:
            subset = self.calls_for(tool_name)
        else:
            subset = self._calls
        if not subset:
            return 0.0
        return sum(1 for c in subset if c.success) / len(subset)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_USE_TRACKER_REGISTRY: dict[str, Any] = {
    "default": ToolUseTracker,
}
