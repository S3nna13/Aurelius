"""Safe tool registry and dispatcher for agent loops.

This module sits above the raw tool callables consumed by
:class:`ReActLoop` and friends. Tools are registered with a JSON-schema
argument description, a per-call timeout, and optional rate-limiting
metadata. The dispatcher performs schema validation, enforces a
session-wide budget, and returns structured :class:`ToolInvocationResult`
envelopes instead of raising. Every dispatch attempt is written to an
audit log so failed calls can be replayed offline.

The JSON-schema validator is a deliberately minimal, hand-written
subset (type, required, additionalProperties=false) with a recursion
depth cap so malicious payloads cannot stall the executor.

Dependencies: standard library only.
"""

from __future__ import annotations

import concurrent.futures
import copy
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ToolSpec",
    "ToolInvocationResult",
    "SessionBudget",
    "ToolRegistryDispatcher",
]


_MAX_SCHEMA_DEPTH = 16

_JSON_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """Registration record for a single tool."""

    name: str
    fn: Callable[..., Any]
    schema: dict
    description: str = ""
    rate_limit_per_minute: int | None = None
    per_call_timeout: float = 5.0
    max_result_chars: int = 16384


@dataclass
class ToolInvocationResult:
    """Structured envelope returned by every dispatch."""

    name: str
    ok: bool
    value: Any
    error: str | None
    duration_ms: float
    truncated: bool


@dataclass
class SessionBudget:
    """Aggregate limits applied across an entire dispatcher session."""

    total_calls: int = 32
    total_wall_seconds: float = 60.0
    per_tool_calls: dict[str, int] | None = None


# ---------------------------------------------------------------------------
# Token bucket rate limiter
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Minute-scaled token bucket: `capacity` tokens refilled over 60s."""

    def __init__(self, capacity: int, clock: Callable[[], float] = time.monotonic):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._refill_rate = float(capacity) / 60.0  # tokens per second
        self._clock = clock
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
            self._last = now

    def try_consume(self) -> bool:
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


# ---------------------------------------------------------------------------
# Minimal JSON-schema validator
# ---------------------------------------------------------------------------


def _validate_schema(value: Any, schema: dict, depth: int = 0) -> str | None:
    """Return an error message or ``None`` if value conforms to schema."""

    if depth > _MAX_SCHEMA_DEPTH:
        return "schema_depth_exceeded"
    if not isinstance(schema, dict):
        return "invalid_schema"

    expected = schema.get("type")
    if expected is not None:
        types = _JSON_TYPE_MAP.get(expected)
        if types is None:
            return f"unsupported_type:{expected}"
        # Reject bool where integer/number required (bool is subclass of int).
        if expected in ("integer", "number") and isinstance(value, bool):
            return f"type_mismatch:expected_{expected}_got_boolean"
        if not isinstance(value, types):
            return f"type_mismatch:expected_{expected}"

    if expected == "object" or (expected is None and isinstance(value, dict)):
        if not isinstance(value, dict):
            return "type_mismatch:expected_object"
        properties = schema.get("properties", {}) or {}
        required = schema.get("required", []) or []
        for key in required:
            if key not in value:
                return f"missing_required:{key}"
        additional = schema.get("additionalProperties", True)
        if additional is False:
            for key in value.keys():
                if key not in properties:
                    return f"unexpected_property:{key}"
        for key, sub_value in value.items():
            sub_schema = properties.get(key)
            if sub_schema is not None:
                err = _validate_schema(sub_value, sub_schema, depth + 1)
                if err is not None:
                    return f"{key}.{err}"

    if expected == "array" or (expected is None and isinstance(value, list)):
        if not isinstance(value, list):
            return "type_mismatch:expected_array"
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for idx, item in enumerate(value):
                err = _validate_schema(item, items_schema, depth + 1)
                if err is not None:
                    return f"[{idx}].{err}"

    return None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


@dataclass
class _AuditEntry:
    name: str
    arguments: dict
    ok: bool
    error: str | None
    duration_ms: float
    truncated: bool
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
            "timestamp": self.timestamp,
        }


class ToolRegistryDispatcher:
    """Thread-safe-ish dispatcher that wraps tool calls with guardrails.

    The dispatcher is designed for single-session usage inside an agent
    loop. It is not re-entrant across threads; callers should construct
    a fresh instance per conversation.
    """

    def __init__(
        self,
        budget: SessionBudget | None = None,
        redactor: Callable[[str], str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._specs: dict[str, ToolSpec] = {}
        self._order: list[str] = []
        self._budget = budget or SessionBudget()
        self._redactor = redactor
        self._clock = clock
        self._call_count = 0
        self._per_tool_count: dict[str, int] = {}
        self._wall_start: float | None = None
        self._rate_buckets: dict[str, _TokenBucket] = {}
        self._audit: list[_AuditEntry] = []
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="tool-dispatch"
        )

    # -- registration -------------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        if not isinstance(spec, ToolSpec):
            raise TypeError("register() requires a ToolSpec")
        if not spec.name or not isinstance(spec.name, str):
            raise ValueError("ToolSpec.name must be a non-empty string")
        if not callable(spec.fn):
            raise TypeError("ToolSpec.fn must be callable")
        if not isinstance(spec.schema, dict):
            raise TypeError("ToolSpec.schema must be a dict")
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name}")
        self._specs[spec.name] = spec
        self._order.append(spec.name)
        if spec.rate_limit_per_minute is not None:
            self._rate_buckets[spec.name] = _TokenBucket(
                spec.rate_limit_per_minute, clock=self._clock
            )

    def list_tools(self) -> list[dict]:
        """Return tool metadata in registration order (deterministic)."""

        out: list[dict] = []
        for name in self._order:
            spec = self._specs[name]
            out.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "schema": copy.deepcopy(spec.schema),
                }
            )
        return out

    def audit_log(self) -> list[dict]:
        return [entry.to_dict() for entry in self._audit]

    def reset(self) -> None:
        self._call_count = 0
        self._per_tool_count.clear()
        self._wall_start = None
        self._audit.clear()
        # Rebuild rate buckets so tokens refresh.
        self._rate_buckets = {
            name: _TokenBucket(spec.rate_limit_per_minute, clock=self._clock)
            for name, spec in self._specs.items()
            if spec.rate_limit_per_minute is not None
        }

    # -- dispatch -----------------------------------------------------------

    def dispatch(self, name: str, arguments: dict) -> ToolInvocationResult:
        start = self._clock()
        if self._wall_start is None:
            self._wall_start = start

        spec = self._specs.get(name)
        if spec is None:
            return self._finish(name, arguments, None, "unknown_tool", start, truncated=False)

        # Budget: total calls.
        if self._call_count >= self._budget.total_calls:
            return self._finish(
                name,
                arguments,
                spec,
                "budget:total_calls_exhausted",
                start,
                truncated=False,
            )
        # Budget: wall seconds.
        if (start - self._wall_start) >= self._budget.total_wall_seconds:
            return self._finish(
                name,
                arguments,
                spec,
                "budget:wall_seconds_exhausted",
                start,
                truncated=False,
            )
        # Budget: per-tool cap.
        per_tool = self._budget.per_tool_calls or {}
        cap = per_tool.get(name)
        if cap is not None and self._per_tool_count.get(name, 0) >= cap:
            return self._finish(
                name,
                arguments,
                spec,
                "budget:per_tool_exhausted",
                start,
                truncated=False,
            )

        # Argument validation.
        if not isinstance(arguments, dict):
            return self._finish(name, arguments, spec, "invalid_arguments:not_a_dict", start, False)
        schema_err = _validate_schema(arguments, spec.schema)
        if schema_err is not None:
            return self._finish(
                name,
                arguments,
                spec,
                f"schema_error:{schema_err}",
                start,
                truncated=False,
            )

        # Rate limit check (consumes a token).
        bucket = self._rate_buckets.get(name)
        if bucket is not None and not bucket.try_consume():
            return self._finish(name, arguments, spec, "rate_limited", start, truncated=False)

        # Count this call against the budget regardless of outcome.
        self._call_count += 1
        self._per_tool_count[name] = self._per_tool_count.get(name, 0) + 1

        # Invoke under timeout.
        future = self._executor.submit(self._call_tool, spec, arguments)
        try:
            value = future.result(timeout=spec.per_call_timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return self._finish(
                name,
                arguments,
                spec,
                f"timeout:exceeded_{spec.per_call_timeout}s",
                start,
                truncated=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._finish(
                name,
                arguments,
                spec,
                f"executor_error:{type(exc).__name__}:{exc}",
                start,
                truncated=False,
            )

        if isinstance(value, _ToolFailure):
            return self._finish(name, arguments, spec, value.message, start, truncated=False)

        # Truncation.
        truncated = False
        if isinstance(value, str) and len(value) > spec.max_result_chars:
            value = value[: spec.max_result_chars]
            truncated = True

        return self._finish(name, arguments, spec, None, start, truncated=truncated, value=value)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _call_tool(spec: ToolSpec, arguments: dict) -> Any:
        try:
            return spec.fn(**arguments)
        except TypeError as exc:
            return _ToolFailure(f"tool_argument_error:{exc}")
        except Exception as exc:
            tb = traceback.format_exception_only(type(exc), exc)
            msg = "".join(tb).strip().splitlines()[-1] if tb else str(exc)
            return _ToolFailure(f"tool_raised:{msg}")

    def _finish(
        self,
        name: str,
        arguments: dict,
        spec: ToolSpec | None,
        error: str | None,
        start: float,
        truncated: bool,
        value: Any = None,
    ) -> ToolInvocationResult:
        duration_ms = max(0.0, (self._clock() - start) * 1000.0)
        ok = error is None

        # Apply redaction to serialisable string fields.
        red_value = value
        red_error = error
        if self._redactor is not None:
            try:
                if isinstance(red_value, str):
                    red_value = self._redactor(red_value)
                if isinstance(red_error, str):
                    red_error = self._redactor(red_error)
            except Exception as exc:  # pragma: no cover - redactor must not escape
                red_error = f"redactor_failure:{type(exc).__name__}:{exc}"
                ok = False

        # Audit every attempt (pre-redaction arguments are deep-copied to
        # freeze the snapshot, then redacted lazily for readability).
        try:
            audit_args = (
                copy.deepcopy(arguments)
                if isinstance(arguments, dict)
                else {"_raw": repr(arguments)}
            )
        except Exception:
            audit_args = {"_raw": "<uncopyable>"}
        if self._redactor is not None:
            audit_args = _redact_structure(audit_args, self._redactor)

        self._audit.append(
            _AuditEntry(
                name=name,
                arguments=audit_args,
                ok=ok,
                error=red_error,
                duration_ms=duration_ms,
                truncated=truncated,
                timestamp=self._clock(),
            )
        )
        return ToolInvocationResult(
            name=name,
            ok=ok,
            value=red_value if ok else None,
            error=red_error,
            duration_ms=duration_ms,
            truncated=truncated,
        )

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: S110
            pass


class _ToolFailure:
    """Sentinel returned from worker when tool raises."""

    __slots__ = ("message",)

    def __init__(self, message: str):
        self.message = message


def _redact_structure(obj: Any, redactor: Callable[[str], str], depth: int = 0) -> Any:
    if depth > _MAX_SCHEMA_DEPTH:
        return obj
    if isinstance(obj, str):
        try:
            return redactor(obj)
        except Exception:
            return obj
    if isinstance(obj, dict):
        return {k: _redact_structure(v, redactor, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_structure(v, redactor, depth + 1) for v in obj]
    return obj
