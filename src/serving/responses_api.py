"""Inspired by OpenAI Responses API (GPT-5 agentic API shape, 2025), OpenAI function-calling API (Apache-2.0 compatible spec), clean-room Aurelius implementation.
GPT-5-style agentic Responses API shape for Aurelius.  Provides stateful
multi-turn request/response types, tool-call support, reasoning chains, and
a streaming SSE event generator — all implemented with stdlib dataclasses
only (no FastAPI, no starlette, no httpx).
"""  # noqa: E501

from __future__ import annotations

import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Literal

__all__ = [
    "ResponsesAPIModel",
    "InputItem",
    "ResponseTool",
    "ResponseOutputItem",
    "ResponsesAPIRequest",
    "ResponsesAPIResponse",
    "ResponseStreamEvent",
    "RESPONSE_CREATED",
    "RESPONSE_IN_PROGRESS",
    "RESPONSE_OUTPUT_ITEM_ADDED",
    "RESPONSE_OUTPUT_TEXT_DELTA",
    "RESPONSE_COMPLETED",
    "RESPONSE_FAILED",
    "ResponsesAPIValidator",
    "ResponsesAPIHandler",
    "RESPONSES_API_REGISTRY",
]


# ---------------------------------------------------------------------------
# Model enum
# ---------------------------------------------------------------------------


class ResponsesAPIModel(StrEnum):
    """Supported Aurelius model identifiers for the Responses API."""

    base = "aurelius-base"
    chat = "aurelius-chat"
    coding = "aurelius-coding"
    long = "aurelius-long"


# ---------------------------------------------------------------------------
# Streaming event type constants
# ---------------------------------------------------------------------------

RESPONSE_CREATED: str = "response.created"
RESPONSE_IN_PROGRESS: str = "response.in_progress"
RESPONSE_OUTPUT_ITEM_ADDED: str = "response.output_item.added"
RESPONSE_OUTPUT_TEXT_DELTA: str = "response.output_text.delta"
RESPONSE_COMPLETED: str = "response.completed"
RESPONSE_FAILED: str = "response.failed"


# ---------------------------------------------------------------------------
# Request / response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InputItem:
    """A single input item in a Responses API request (message or tool result)."""

    role: Literal["user", "assistant", "system", "tool"]
    content: str | list[dict]
    input_type: Literal["message", "tool_result"] = "message"


@dataclass
class ResponseTool:
    """A tool offered to the model in a Responses API request."""

    type: Literal["function", "computer_use", "web_search"]
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)


@dataclass
class ResponseOutputItem:
    """A single item in the Responses API output list."""

    id: str
    type: Literal["message", "reasoning", "tool_call", "tool_result"]
    content: str | None = None
    tool_name: str | None = None
    tool_input: dict | None = None
    status: Literal["in_progress", "completed", "failed"] = "in_progress"


@dataclass
class ResponsesAPIRequest:
    """Inbound request for the Aurelius Responses API."""

    model: str
    input: list[InputItem]
    tools: list[ResponseTool] | None = None
    max_output_tokens: int | None = None
    temperature: float = 1.0
    stream: bool = False
    previous_response_id: str | None = None
    reasoning: dict | None = None
    store: bool = True


@dataclass
class ResponsesAPIResponse:
    """Outbound response from the Aurelius Responses API."""

    id: str
    model: str
    status: Literal["completed", "in_progress", "failed"]
    output: list[ResponseOutputItem]
    usage: dict
    created_at: float
    object: Literal["response"] = "response"


# ---------------------------------------------------------------------------
# Streaming event dataclass
# ---------------------------------------------------------------------------


@dataclass
class ResponseStreamEvent:
    """A single SSE streaming event emitted during a Responses API stream."""

    event_type: str
    data: dict
    sequence_number: int


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class ResponsesAPIValidator:
    """Validates :class:`ResponsesAPIRequest` and :class:`ResponsesAPIResponse`
    objects, returning a list of human-readable error strings.  An empty list
    means the object is valid.
    """

    def validate_request(self, req: ResponsesAPIRequest) -> list[str]:
        errors: list[str] = []

        if not req.model:
            errors.append("model must not be empty")

        if not req.input:
            errors.append("input must not be empty")

        if not (0.0 <= req.temperature <= 2.0):
            errors.append(f"temperature must be in [0.0, 2.0], got {req.temperature}")

        if req.max_output_tokens is not None and req.max_output_tokens <= 0:
            errors.append(f"max_output_tokens must be > 0 if set, got {req.max_output_tokens}")

        if req.tools:
            for tool in req.tools:
                if not tool.name:
                    errors.append("all tools must have a non-empty name")
                    break

        return errors

    def validate_response(self, resp: ResponsesAPIResponse) -> list[str]:
        errors: list[str] = []

        if not resp.id:
            errors.append("response id must not be empty")

        if not isinstance(resp.output, list):
            errors.append("response output must be a list")

        return errors


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class ResponsesAPIHandler:
    """Stub handler that creates :class:`ResponsesAPIResponse` objects and
    yields :class:`ResponseStreamEvent` generators.

    ``RESPONSE_COUNTER`` is a simple class-level integer used to construct
    deterministic response ids without requiring external state.
    """

    RESPONSE_COUNTER: ClassVar[int] = 0

    @classmethod
    def _next_id(cls) -> str:
        cls.RESPONSE_COUNTER += 1
        return f"resp_{cls.RESPONSE_COUNTER}_{uuid.uuid4().hex[:8]}"

    def create_response(self, req: ResponsesAPIRequest) -> ResponsesAPIResponse:
        """Build a stub :class:`ResponsesAPIResponse` for *req*."""
        response_id = self._next_id()
        output_item = ResponseOutputItem(
            id=f"item_{uuid.uuid4().hex[:8]}",
            type="message",
            content=f"Stub response from {req.model}",
            status="completed",
        )
        usage = {
            "input_tokens": len(str(req.input)),
            "output_tokens": 10,
        }
        return ResponsesAPIResponse(
            id=response_id,
            model=req.model,
            status="completed",
            output=[output_item],
            usage=usage,
            created_at=time.time(),
        )

    def stream_events(self, req: ResponsesAPIRequest) -> Generator[ResponseStreamEvent, None, None]:
        """Yield a minimal sequence of SSE events for *req*."""
        response_id = self._next_id()
        seq = 0

        # 1. response.created
        yield ResponseStreamEvent(
            event_type=RESPONSE_CREATED,
            data={"id": response_id, "model": req.model, "status": "in_progress"},
            sequence_number=seq,
        )
        seq += 1

        # 2. response.in_progress
        yield ResponseStreamEvent(
            event_type=RESPONSE_IN_PROGRESS,
            data={"id": response_id, "status": "in_progress"},
            sequence_number=seq,
        )
        seq += 1

        # 3-5. output_text.delta (3 chunks)
        chunks = [
            f"Stub response from {req.model}",
            " (chunk 2)",
            " (chunk 3)",
        ]
        for chunk in chunks:
            yield ResponseStreamEvent(
                event_type=RESPONSE_OUTPUT_TEXT_DELTA,
                data={"id": response_id, "delta": chunk},
                sequence_number=seq,
            )
            seq += 1

        # 6. response.completed
        yield ResponseStreamEvent(
            event_type=RESPONSE_COMPLETED,
            data={
                "id": response_id,
                "model": req.model,
                "status": "completed",
                "usage": {
                    "input_tokens": len(str(req.input)),
                    "output_tokens": 10,
                },
            },
            sequence_number=seq,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RESPONSES_API_REGISTRY: dict[str, type[ResponsesAPIHandler]] = {
    "default": ResponsesAPIHandler,
}

# Additive: register into the shared API_SHAPE_REGISTRY so the Responses API
# shape is discoverable alongside existing validators/decoders.
try:
    from src.serving.openai_api_validator import API_SHAPE_REGISTRY  # type: ignore[import]

    API_SHAPE_REGISTRY["responses"] = ResponsesAPIHandler
except Exception:  # noqa: S110
    pass
