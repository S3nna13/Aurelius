"""Streaming tool-call accumulation: partial JSON, delta events, finalization."""

import json
from enum import StrEnum


class ToolCallState(StrEnum):
    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETE = "complete"
    ERROR = "error"


class ToolCallBuffer:
    """Accumulates streamed argument fragments for a single tool call."""

    def __init__(self, tool_call_id: str, function_name: str) -> None:
        self._tool_call_id = tool_call_id
        self._function_name = function_name
        self._state: ToolCallState = ToolCallState.PENDING
        self._raw_arguments: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tool_call_id(self) -> str:
        return self._tool_call_id

    @property
    def function_name(self) -> str:
        return self._function_name

    @property
    def state(self) -> ToolCallState:
        return self._state

    @property
    def raw_arguments(self) -> str:
        return self._raw_arguments

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def append_argument_delta(self, delta: str) -> None:
        """Accumulate a raw JSON fragment arriving from the stream."""
        self._raw_arguments += delta
        if self._state == ToolCallState.PENDING:
            self._state = ToolCallState.STREAMING

    def finalize(self) -> dict | None:
        """Attempt to parse the accumulated JSON.

        Returns the parsed dict and sets state=COMPLETE on success.
        Returns None and sets state=ERROR on parse failure.
        """
        try:
            parsed = json.loads(self._raw_arguments)
            self._state = ToolCallState.COMPLETE
            return parsed
        except (json.JSONDecodeError, ValueError):
            self._state = ToolCallState.ERROR
            return None


class ToolCallStreamAccumulator:
    """Manages multiple concurrent ToolCallBuffer instances."""

    def __init__(self) -> None:
        self._buffers: dict[str, ToolCallBuffer] = {}

    def start_tool_call(self, tool_call_id: str, function_name: str) -> ToolCallBuffer:
        """Create and register a new buffer for the given tool call."""
        buf = ToolCallBuffer(tool_call_id, function_name)
        self._buffers[tool_call_id] = buf
        return buf

    def append_delta(self, tool_call_id: str, delta: str) -> bool:
        """Append a delta to the named buffer.

        Returns False if the tool_call_id is unknown, True otherwise.
        """
        buf = self._buffers.get(tool_call_id)
        if buf is None:
            return False
        buf.append_argument_delta(delta)
        return True

    def finalize_all(self) -> list[dict]:
        """Finalize every buffer; return successfully parsed dicts (skip ERROR)."""
        results: list[dict] = []
        for buf in self._buffers.values():
            parsed = buf.finalize()
            if parsed is not None:
                results.append(parsed)
        return results

    def pending_count(self) -> int:
        """Return count of buffers whose state is neither COMPLETE nor ERROR."""
        return sum(
            1
            for buf in self._buffers.values()
            if buf.state not in (ToolCallState.COMPLETE, ToolCallState.ERROR)
        )

    def get_buffer(self, tool_call_id: str) -> ToolCallBuffer | None:
        """Return the buffer for *tool_call_id*, or None if not found."""
        return self._buffers.get(tool_call_id)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_CALL_ACCUMULATOR_REGISTRY: dict[str, type] = {
    "default": ToolCallStreamAccumulator,
}
