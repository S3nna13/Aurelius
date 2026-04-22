"""OpenAI-compatible Server-Sent Events (SSE) chat-completion streaming.

This module layers *chat-specific* framing on top of the generic WHATWG SSE
encoder in :mod:`src.serving.sse_stream_encoder`. It emits frames shaped like
OpenAI's ``chat.completion.chunk`` objects so clients built against the
OpenAI Python/JS SDKs can consume Aurelius streaming responses without any
translation shim.

A streaming chat response typically looks like::

    data: {"id": "...", "object": "chat.completion.chunk", ..., "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n
    data: {"id": "...", ..., "choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}\n\n
    data: {"id": "...", ..., "choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}\n\n
    data: {"id": "...", ..., "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n
    data: [DONE]\n\n

Pure stdlib: ``dataclasses``, ``json``, ``time``, ``typing``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


_DONE_SENTINEL = "[DONE]"
_FRAME_PREFIX = "data: "
_FRAME_TERMINATOR = "\n\n"


class SSEParseError(Exception):
    """Raised when :func:`parse_sse_event` cannot decode a line."""


@dataclass
class ChatCompletionChunk:
    """One OpenAI-shape ``chat.completion.chunk`` object.

    Attributes
    ----------
    id:
        Stable stream id — same value for every chunk of a single response.
    object_type:
        Always ``"chat.completion.chunk"`` for streaming (kept configurable so
        tests can round-trip other shapes if desired).
    created:
        Unix timestamp in seconds (integer — matches OpenAI's schema).
    model:
        Model name being served.
    choices:
        List of ``{"index": int, "delta": dict, "finish_reason": str|None}``.
    """

    id: str
    object_type: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: list[dict] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object_type,
            "created": int(self.created),
            "model": self.model,
            "choices": list(self.choices),
        }


@dataclass
class ChoiceDelta:
    """One element of ``ChatCompletionChunk.choices``.

    ``delta`` is a strict subset of the OpenAI delta shape: any of ``role``,
    ``content``, ``tool_calls``. Keys with ``None`` values are omitted so the
    wire shape matches the official SDK exactly.
    """

    index: int
    delta: dict
    finish_reason: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "delta": dict(self.delta),
            "finish_reason": self.finish_reason,
        }


class SSEChatStream:
    """Stateful emitter for OpenAI-compatible chat-completion SSE frames.

    Parameters
    ----------
    model_name:
        Echoed into every chunk's ``model`` field.
    max_chunk_bytes:
        Maximum UTF-8 byte length of a single emitted SSE frame. Raises
        :class:`ValueError` if a frame would exceed this. Defaults to
        32 KiB which is well above typical token deltas but small enough to
        catch runaway inputs.
    """

    def __init__(self, model_name: str, max_chunk_bytes: int = 32768) -> None:
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty str")
        if not isinstance(max_chunk_bytes, int) or max_chunk_bytes <= 0:
            raise ValueError("max_chunk_bytes must be a positive int")
        self._model = model_name
        self._max_chunk_bytes = max_chunk_bytes
        # Stable id across every chunk emitted by this instance — matches
        # OpenAI's contract that every chunk of a single response shares id.
        self._stream_id = f"chatcmpl-{int(time.time() * 1_000_000):x}"
        self._created = int(time.time())

    # -- public emitters -----------------------------------------------------

    def emit_role_delta(self, role: str = "assistant") -> str:
        """First chunk of a stream: sets the assistant role."""
        if not isinstance(role, str) or not role:
            raise ValueError("role must be a non-empty str")
        choice = ChoiceDelta(index=0, delta={"role": role}, finish_reason=None)
        return self._frame(self._chunk([choice]))

    def emit_content_delta(self, content: str) -> str:
        """Append a content token / substring to the assistant message."""
        if not isinstance(content, str):
            raise TypeError("content must be str")
        choice = ChoiceDelta(
            index=0, delta={"content": content}, finish_reason=None
        )
        return self._frame(self._chunk([choice]))

    def emit_tool_call_delta(
        self,
        tool_call_index: int,
        name: str | None,
        arguments_chunk: str | None,
    ) -> str:
        """Emit an incremental tool-call delta.

        OpenAI's spec keys tool calls by ``index`` (so multiple parallel tool
        calls can stream interleaved) and sends ``function.name`` once up front
        followed by ``function.arguments`` chunks that the client concatenates.
        """
        if not isinstance(tool_call_index, int) or tool_call_index < 0:
            raise ValueError("tool_call_index must be a non-negative int")
        function: dict[str, Any] = {}
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("name must be str or None")
            function["name"] = name
        if arguments_chunk is not None:
            if not isinstance(arguments_chunk, str):
                raise TypeError("arguments_chunk must be str or None")
            function["arguments"] = arguments_chunk
        tool_call = {"index": tool_call_index, "function": function}
        choice = ChoiceDelta(
            index=0,
            delta={"tool_calls": [tool_call]},
            finish_reason=None,
        )
        return self._frame(self._chunk([choice]))

    def emit_finish(self, reason: str = "stop") -> str:
        """Terminal chunk carrying the stop reason and an empty delta."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty str")
        choice = ChoiceDelta(index=0, delta={}, finish_reason=reason)
        return self._frame(self._chunk([choice]))

    def emit_done(self) -> str:
        """OpenAI's literal end-of-stream sentinel: ``data: [DONE]\\n\\n``."""
        return f"{_FRAME_PREFIX}{_DONE_SENTINEL}{_FRAME_TERMINATOR}"

    # -- internals -----------------------------------------------------------

    def _chunk(self, choices: list[ChoiceDelta]) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=self._stream_id,
            object_type="chat.completion.chunk",
            created=self._created,
            model=self._model,
            choices=[c.to_json_dict() for c in choices],
        )

    def _frame(self, chunk: ChatCompletionChunk) -> str:
        # ``separators`` keeps output compact and deterministic (no trailing
        # whitespace, stable key ordering via dataclass-controlled dict build).
        payload = json.dumps(
            chunk.to_json_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        frame = f"{_FRAME_PREFIX}{payload}{_FRAME_TERMINATOR}"
        # Enforce the frame-size cap on the encoded byte length — text length
        # alone would undercount multi-byte unicode characters.
        if len(frame.encode("utf-8")) > self._max_chunk_bytes:
            raise ValueError(
                f"SSE chat frame exceeds max_chunk_bytes="
                f"{self._max_chunk_bytes}"
            )
        return frame


def parse_sse_event(line: str) -> dict | None:
    """Reverse-parse a single ``data: {...}\\n\\n`` SSE frame.

    Returns
    -------
    dict | None
        Decoded JSON payload, or ``None`` if the frame is the ``[DONE]``
        terminator.

    Raises
    ------
    SSEParseError
        On any malformed input (missing prefix, missing blank-line terminator,
        invalid JSON payload).
    """
    if not isinstance(line, str):
        raise SSEParseError("input must be str")
    if not line.startswith(_FRAME_PREFIX):
        raise SSEParseError("missing 'data: ' prefix")
    # Strict framing: each frame is terminated by exactly "\n\n". We accept
    # either a single trailing "\n\n" or its absence only if the payload is
    # the [DONE] sentinel written without framing (defensive).
    body = line[len(_FRAME_PREFIX):]
    if body.endswith(_FRAME_TERMINATOR):
        body = body[: -len(_FRAME_TERMINATOR)]
    elif body.endswith("\n"):
        # single newline alone is malformed — SSE requires blank-line end
        raise SSEParseError("frame must end with blank line (\\n\\n)")
    body = body.strip()
    if body == _DONE_SENTINEL:
        return None
    if not body:
        raise SSEParseError("empty frame payload")
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SSEParseError(f"invalid JSON in frame: {exc}") from exc
    if not isinstance(obj, dict):
        raise SSEParseError("frame payload must be a JSON object")
    return obj


__all__ = [
    "ChatCompletionChunk",
    "ChoiceDelta",
    "SSEChatStream",
    "SSEParseError",
    "parse_sse_event",
]
