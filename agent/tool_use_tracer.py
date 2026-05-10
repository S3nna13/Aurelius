from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ToolCallStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ToolCall:
    call_id: str
    tool_name: str
    args: dict
    status: ToolCallStatus
    result: Any | None = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    latency_ms: float | None = None


@dataclass
class ToolTrace:
    trace_id: str
    calls: list[ToolCall]
    total_calls: int
    success_rate: float
    mean_latency_ms: float | None


class ToolUseTracer:
    """Trace all tool calls within an agent run."""

    def __init__(self) -> None:
        self._traces: dict[str, list[ToolCall]] = {}
        self._calls: dict[str, ToolCall] = {}

    def start_trace(self) -> str:
        trace_id = str(uuid.uuid4())
        self._traces[trace_id] = []
        return trace_id

    def record_call(self, trace_id: str, tool_name: str, args: dict) -> ToolCall:
        if trace_id not in self._traces:
            raise KeyError(f"Unknown trace_id: {trace_id}")
        call = ToolCall(
            call_id=str(uuid.uuid4()),
            tool_name=tool_name,
            args=args,
            status=ToolCallStatus.PENDING,
        )
        self._traces[trace_id].append(call)
        self._calls[call.call_id] = call
        return call

    def start_call(self, call_id: str) -> ToolCall:
        call = self._get_call(call_id)
        call.status = ToolCallStatus.RUNNING
        call.started_at = time.monotonic()
        return call

    def complete_call(self, call_id: str, result: Any) -> ToolCall:
        call = self._get_call(call_id)
        call.completed_at = time.monotonic()
        call.result = result
        call.status = ToolCallStatus.SUCCESS
        if call.started_at is not None:
            call.latency_ms = (call.completed_at - call.started_at) * 1000.0
        return call

    def fail_call(self, call_id: str, error: str, timeout: bool = False) -> ToolCall:
        call = self._get_call(call_id)
        call.completed_at = time.monotonic()
        call.error = error
        call.status = ToolCallStatus.TIMEOUT if timeout else ToolCallStatus.ERROR
        if call.started_at is not None:
            call.latency_ms = (call.completed_at - call.started_at) * 1000.0
        return call

    def get_trace(self, trace_id: str) -> ToolTrace | None:
        calls = self._traces.get(trace_id)
        if calls is None:
            return None
        total = len(calls)
        successes = sum(1 for c in calls if c.status == ToolCallStatus.SUCCESS)
        rate = (successes / total) if total else 0.0
        latencies = [c.latency_ms for c in calls if c.latency_ms is not None]
        mean_lat = (sum(latencies) / len(latencies)) if latencies else None
        return ToolTrace(
            trace_id=trace_id,
            calls=list(calls),
            total_calls=total,
            success_rate=rate,
            mean_latency_ms=mean_lat,
        )

    def export_trace(self, trace_id: str) -> dict:
        trace = self.get_trace(trace_id)
        if trace is None:
            return {}
        return {
            "trace_id": trace.trace_id,
            "total_calls": trace.total_calls,
            "success_rate": trace.success_rate,
            "mean_latency_ms": trace.mean_latency_ms,
            "calls": [
                {
                    "call_id": c.call_id,
                    "tool_name": c.tool_name,
                    "args": c.args,
                    "status": c.status.value,
                    "result": c.result
                    if isinstance(c.result, (str, int, float, bool, list, dict, type(None)))
                    else str(c.result),
                    "error": c.error,
                    "started_at": c.started_at,
                    "completed_at": c.completed_at,
                    "latency_ms": c.latency_ms,
                }
                for c in trace.calls
            ],
        }

    def active_calls(self, trace_id: str) -> list[ToolCall]:
        calls = self._traces.get(trace_id, [])
        return [c for c in calls if c.status == ToolCallStatus.RUNNING]

    def _get_call(self, call_id: str) -> ToolCall:
        call = self._calls.get(call_id)
        if call is None:
            raise KeyError(f"Unknown call_id: {call_id}")
        return call
