"""Execution trace analyzer.

Reads :class:`~src.agent.execution_tracer.ExecutionEvent` streams and
produces actionable insights: failure patterns, performance bottlenecks,
tool-success rates, and repeated-hammer detection.  Pure stdlib.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.agent.execution_tracer import ExecutionEvent, ExecutionEventType


@dataclass
class ToolSummary:
    name: str
    calls: int
    errors: int
    total_duration_ms: float
    avg_duration_ms: float


@dataclass
class TraceAnalysis:
    total_events: int
    total_steps: int
    error_count: int
    unique_errors: list[str]
    tool_summaries: list[ToolSummary]
    slowest_event: dict[str, Any] | None
    failure_patterns: list[str]
    recommendations: list[str]


class TraceAnalyzer:
    """Analyze execution traces for debugging and self-improvement."""

    def __init__(self, slow_threshold_ms: float = 500.0) -> None:
        self.slow_threshold_ms = slow_threshold_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, events: list[ExecutionEvent]) -> TraceAnalysis:
        if not events:
            return TraceAnalysis(
                total_events=0,
                total_steps=0,
                error_count=0,
                unique_errors=[],
                tool_summaries=[],
                slowest_event=None,
                failure_patterns=[],
                recommendations=[],
            )

        total_steps = max(e.step for e in events) + 1
        error_events = [e for e in events if e.event_type == ExecutionEventType.ERROR]
        error_count = len(error_events)
        unique_errors = list({self._error_signature(e) for e in error_events})

        tool_summaries = self._summarize_tools(events)
        slowest_event = self._find_slowest(events)
        failure_patterns = self._detect_failure_patterns(events)
        recommendations = self._generate_recommendations(
            error_count, tool_summaries, failure_patterns, slowest_event
        )

        return TraceAnalysis(
            total_events=len(events),
            total_steps=total_steps,
            error_count=error_count,
            unique_errors=unique_errors,
            tool_summaries=tool_summaries,
            slowest_event=slowest_event,
            failure_patterns=failure_patterns,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error_signature(event: ExecutionEvent) -> str:
        content = event.content or {}
        msg = str(content.get("message", content))
        # Truncate early to prevent regex DoS on megabyte-long error strings
        msg = msg[:2_048]
        # Normalize numbers and paths so trivial variations collapse
        msg = re.sub(r"\d+", "N", msg)
        msg = re.sub(r"/[\w./-]+", "<PATH>", msg)
        return msg[:200]

    def _summarize_tools(self, events: list[ExecutionEvent]) -> list[ToolSummary]:
        tool_events = [e for e in events if e.event_type == ExecutionEventType.TOOL_CALL]
        tool_results = {
            e.content.get("tool_call_id", e.event_id): e
            for e in events
            if e.event_type == ExecutionEventType.TOOL_RESULT
        }

        by_name: dict[str, dict[str, Any]] = {}
        for evt in tool_events:
            name = evt.content.get("tool_name", "unknown")
            bucket = by_name.setdefault(name, {"calls": 0, "errors": 0, "duration_ms": 0.0})
            bucket["calls"] += 1
            dur = evt.duration_ms or 0.0
            bucket["duration_ms"] += dur

            # Match result by event_id heuristic
            result = tool_results.get(evt.content.get("tool_call_id", evt.event_id))
            if result is not None:
                bucket["errors"] += bool(result.content.get("error"))

        summaries = []
        for name, data in sorted(by_name.items()):
            avg = data["duration_ms"] / data["calls"] if data["calls"] else 0.0
            summaries.append(
                ToolSummary(
                    name=name,
                    calls=data["calls"],
                    errors=data["errors"],
                    total_duration_ms=data["duration_ms"],
                    avg_duration_ms=avg,
                )
            )
        return summaries

    def _find_slowest(self, events: list[ExecutionEvent]) -> dict[str, Any] | None:
        with_dur = [e for e in events if e.duration_ms is not None]
        if not with_dur:
            return None
        slowest = max(with_dur, key=lambda e: e.duration_ms)  # type: ignore[type-var]
        return {
            "event_type": slowest.event_type.value,
            "step": slowest.step,
            "duration_ms": slowest.duration_ms,
            "content_keys": list(slowest.content.keys()),
        }

    def _detect_failure_patterns(self, events: list[ExecutionEvent]) -> list[str]:
        patterns: list[str] = []

        # Pattern: rapid retry loop (same tool called ≥3 times in ≤5 steps)
        tool_calls = [
            (i, e) for i, e in enumerate(events) if e.event_type == ExecutionEventType.TOOL_CALL
        ]
        for idx, (pos, evt) in enumerate(tool_calls):
            name = evt.content.get("tool_name", "unknown")
            window = tool_calls[idx : idx + 3]
            if len(window) >= 3 and all(w[1].content.get("tool_name") == name for w in window):
                if window[-1][0] - window[0][0] <= 5:
                    patterns.append(
                        f"rapid_retry_loop: tool '{name}' called {len(window)}x in "
                        f"{window[-1][0] - window[0][0]} steps"
                    )
                    break  # report once per trace

        # Pattern: error immediately after tool call (likely tool failure)
        for i in range(len(events) - 1):
            if (
                events[i].event_type == ExecutionEventType.TOOL_CALL
                and events[i + 1].event_type == ExecutionEventType.ERROR
            ):
                name = events[i].content.get("tool_name", "unknown")
                patterns.append(f"tool_error_chain: '{name}' call followed by error")
                break

        # Pattern: think-heavy trace (many THINK events, few ACT)
        think_count = sum(1 for e in events if e.event_type == ExecutionEventType.THINK)
        act_count = sum(1 for e in events if e.event_type == ExecutionEventType.ACT)
        if think_count > act_count * 3 and think_count > 5:
            patterns.append(f"think_heavy: {think_count} think events vs {act_count} act events")

        return patterns

    def _generate_recommendations(
        self,
        error_count: int,
        tool_summaries: list[ToolSummary],
        failure_patterns: list[str],
        slowest_event: dict[str, Any] | None,
    ) -> list[str]:
        recs: list[str] = []

        if error_count > 0:
            recs.append(
                f"{error_count} error(s) detected — review tool inputs and "
                "add pre-validation where possible."
            )

        for ts in tool_summaries:
            if ts.errors > 0 and ts.errors / ts.calls >= 0.5:
                recs.append(
                    f"Tool '{ts.name}' fails {ts.errors}/{ts.calls} times — "
                    "consider adding fallback logic or tightening schemas."
                )

        if slowest_event and slowest_event.get("duration_ms", 0) > self.slow_threshold_ms:
            recs.append(
                f"Slowest step ({slowest_event['event_type']} at step "
                f"{slowest_event['step']}) took {slowest_event['duration_ms']:.1f} ms — "
                "investigate for blocking I/O or large payload processing."
            )

        for pat in failure_patterns:
            if pat.startswith("rapid_retry_loop"):
                recs.append("Retry loop detected — add exponential backoff or circuit-breaker.")
            elif pat.startswith("tool_error_chain"):
                recs.append("Tool calls consistently erroring — review tool description clarity.")
            elif pat.startswith("think_heavy"):
                recs.append(
                    "Excessive thinking steps — tighten system prompt or add max-think limit."
                )

        return recs


#: Registry for tests / DI.
TRACE_ANALYZER_REGISTRY: dict[str, TraceAnalyzer] = {"default": TraceAnalyzer()}
