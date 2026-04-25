"""Tool-use tracing for agent tool-call latencies, success rates, and traces."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum


class ToolCallStatus(StrEnum):
    ACTIVE = "active"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class ToolCall:
    call_id: str
    trace_id: str
    tool_name: str
    status: ToolCallStatus
    start_time: float
    end_time: float | None = None
    error: str | None = None


@dataclass
class ToolTrace:
    trace_id: str
    calls: list[ToolCall] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class ToolUseTracer:
    max_traces: int = 100
    max_calls_per_trace: int = 5000
    _traces: dict[str, ToolTrace] = field(default_factory=dict)
    _active_calls: dict[str, ToolCall] = field(default_factory=dict)

    @property
    def active_calls(self) -> list[ToolCall]:
        return list(self._active_calls.values())

    def start_trace(self, trace_id: str | None = None) -> str:
        tid = trace_id or str(uuid.uuid4())[:8]
        if len(self._traces) >= self.max_traces:
            oldest = min(self._traces)
            del self._traces[oldest]
        self._traces[tid] = ToolTrace(trace_id=tid)
        return tid

    def get_trace(self, trace_id: str) -> ToolTrace:
        if trace_id not in self._traces:
            raise KeyError(f"Trace {trace_id!r} not found")
        return self._traces[trace_id]

    def start_call(
        self,
        trace_id: str,
        tool_name: str,
        call_id: str | None = None,
    ) -> str:
        if trace_id not in self._traces:
            raise KeyError(f"Trace {trace_id!r} not found")
        cid = call_id or str(uuid.uuid4())[:8]
        call = ToolCall(
            call_id=cid,
            trace_id=trace_id,
            tool_name=tool_name,
            status=ToolCallStatus.ACTIVE,
            start_time=time.time(),
        )
        self._active_calls[cid] = call
        return cid

    def complete_call(self, call_id: str) -> None:
        if call_id not in self._active_calls:
            raise KeyError(f"Call {call_id!r} not found")
        call = self._active_calls[call_id]
        call.status = ToolCallStatus.SUCCESS
        call.end_time = time.time()
        trace = self._traces[call.trace_id]
        trace.calls.append(call)
        del self._active_calls[call_id]

    def fail_call(self, call_id: str, error: str) -> None:
        if call_id not in self._active_calls:
            raise KeyError(f"Call {call_id!r} not found")
        call = self._active_calls[call_id]
        call.status = ToolCallStatus.FAILURE
        call.end_time = time.time()
        call.error = error
        trace = self._traces[call.trace_id]
        trace.calls.append(call)
        del self._active_calls[call_id]

    def record_call(
        self,
        trace_id: str,
        tool_name: str,
        success: bool = True,
        error: str = "",
        duration_ms: float | None = None,
    ) -> ToolCall:
        call_id = self.start_call(trace_id, tool_name)
        if success:
            self.complete_call(call_id)
        else:
            self.fail_call(call_id, error)
        return self._traces[trace_id].calls[-1]

    def export_trace(self, trace_id: str) -> str:
        if trace_id not in self._traces:
            raise KeyError(f"Trace {trace_id!r} not found")
        trace = self._traces[trace_id]
        calls_data = []
        for c in trace.calls:
            calls_data.append({
                "call_id": c.call_id,
                "tool_name": c.tool_name,
                "status": c.status.value,
                "duration_ms": (c.end_time - c.start_time) * 1000 if c.end_time else None,
                "error": c.error,
            })
        return json.dumps({"trace_id": trace_id, "calls": calls_data}, indent=2)


TOOL_USE_TRACER_REGISTRY: dict[str, object] = {"default": ToolUseTracer()}