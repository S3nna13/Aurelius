"""OpenAI Chat Completions API schema validator.

Pre-flight validation for /v1/chat/completions request and response shapes,
including the Function-Calling extension. Pure stdlib implementation — no
jsonschema, pydantic, or fastapi dependencies.

The validator is intentionally strict by default: unknown top-level fields
produce errors. Callers (e.g. src/serving/api_server.py) can opt into a
lenient mode via ``strict=False`` for forward compatibility with upstream
OpenAI schema additions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


__all__ = [
    "APIValidationError",
    "OpenAIChatRequestValidator",
    "OpenAIChatResponseValidator",
    "API_SHAPE_REGISTRY",
]


@dataclass(frozen=True)
class APIValidationError:
    """Structured validation failure.

    Attributes:
        field: Dotted JSON path to the offending element (e.g. ``messages[0].role``).
        message: Human-readable failure reason.
        code: Stable machine-readable error code (e.g. ``missing_field``).
    """

    field: str
    message: str
    code: str


# ---------------------------------------------------------------------------
# Request validator
# ---------------------------------------------------------------------------

_REQUEST_KNOWN_FIELDS = frozenset({
    "model",
    "messages",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stream_options",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "tools",
    "tool_choice",
    "response_format",
    "logprobs",
    "top_logprobs",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "user",
    "logit_bias",
    "parallel_tool_calls",
    "functions",
    "function_call",
    "service_tier",
    "metadata",
    "store",
})

_VALID_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})
_VALID_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call", None})


def _is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def _is_number(x: Any) -> bool:
    # Reject bool (subclass of int) for numeric fields.
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


class OpenAIChatRequestValidator:
    """Validator for OpenAI ``/v1/chat/completions`` request payloads."""

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def validate(self, payload: dict) -> list[APIValidationError]:
        errors: list[APIValidationError] = []

        if not isinstance(payload, dict):
            return [APIValidationError("", "payload must be a JSON object", "invalid_type")]

        # model (required, str)
        if "model" not in payload:
            errors.append(APIValidationError("model", "field is required", "missing_field"))
        elif not isinstance(payload["model"], str) or not payload["model"]:
            errors.append(APIValidationError("model", "must be a non-empty string", "invalid_type"))

        # messages (required, list)
        if "messages" not in payload:
            errors.append(APIValidationError("messages", "field is required", "missing_field"))
        elif not isinstance(payload["messages"], list):
            errors.append(APIValidationError("messages", "must be a list", "invalid_type"))
        elif len(payload["messages"]) == 0:
            errors.append(APIValidationError("messages", "must contain at least one message", "empty_list"))
        else:
            for idx, msg in enumerate(payload["messages"]):
                errors.extend(self._validate_message(msg, f"messages[{idx}]"))

        # temperature (0-2)
        if "temperature" in payload:
            v = payload["temperature"]
            if not _is_number(v):
                errors.append(APIValidationError("temperature", "must be a number", "invalid_type"))
            elif v < 0 or v > 2:
                errors.append(APIValidationError("temperature", "must be between 0 and 2", "out_of_range"))

        # top_p (0-1)
        if "top_p" in payload:
            v = payload["top_p"]
            if not _is_number(v):
                errors.append(APIValidationError("top_p", "must be a number", "invalid_type"))
            elif v < 0 or v > 1:
                errors.append(APIValidationError("top_p", "must be between 0 and 1", "out_of_range"))

        # n (>=1)
        if "n" in payload:
            v = payload["n"]
            if not _is_int(v):
                errors.append(APIValidationError("n", "must be an integer", "invalid_type"))
            elif v < 1:
                errors.append(APIValidationError("n", "must be >= 1", "out_of_range"))

        # stream (bool)
        if "stream" in payload and not _is_bool(payload["stream"]):
            errors.append(APIValidationError("stream", "must be a boolean", "invalid_type"))

        # max_tokens (>=1)
        if "max_tokens" in payload:
            v = payload["max_tokens"]
            if v is None:
                pass
            elif not _is_int(v):
                errors.append(APIValidationError("max_tokens", "must be an integer", "invalid_type"))
            elif v < 1:
                errors.append(APIValidationError("max_tokens", "must be >= 1", "out_of_range"))

        # stop (str|list|null)
        if "stop" in payload:
            v = payload["stop"]
            if v is None or isinstance(v, str):
                pass
            elif isinstance(v, list):
                for i, s in enumerate(v):
                    if not isinstance(s, str):
                        errors.append(APIValidationError(f"stop[{i}]", "must be a string", "invalid_type"))
                if len(v) > 4:
                    errors.append(APIValidationError("stop", "at most 4 stop sequences", "out_of_range"))
            else:
                errors.append(APIValidationError("stop", "must be string, list of strings, or null", "invalid_type"))

        # tools (list of ToolSchema)
        if "tools" in payload:
            v = payload["tools"]
            if not isinstance(v, list):
                errors.append(APIValidationError("tools", "must be a list", "invalid_type"))
            else:
                for i, tool in enumerate(v):
                    errors.extend(self._validate_tool(tool, f"tools[{i}]"))

        # tool_choice: "none"|"auto"|"required"|{"type":"function","function":{"name":...}}
        if "tool_choice" in payload:
            errors.extend(self._validate_tool_choice(payload["tool_choice"], "tool_choice"))

        # response_format: {"type": "text"|"json_object"|"json_schema", ...}
        if "response_format" in payload:
            errors.extend(self._validate_response_format(payload["response_format"], "response_format"))

        # logprobs (bool)
        if "logprobs" in payload and not _is_bool(payload["logprobs"]):
            errors.append(APIValidationError("logprobs", "must be a boolean", "invalid_type"))

        # top_logprobs (0-20 int)
        if "top_logprobs" in payload:
            v = payload["top_logprobs"]
            if not _is_int(v):
                errors.append(APIValidationError("top_logprobs", "must be an integer", "invalid_type"))
            elif v < 0 or v > 20:
                errors.append(APIValidationError("top_logprobs", "must be between 0 and 20", "out_of_range"))

        # presence_penalty / frequency_penalty (-2..2)
        for key in ("presence_penalty", "frequency_penalty"):
            if key in payload:
                v = payload[key]
                if not _is_number(v):
                    errors.append(APIValidationError(key, "must be a number", "invalid_type"))
                elif v < -2 or v > 2:
                    errors.append(APIValidationError(key, "must be between -2 and 2", "out_of_range"))

        # seed (int)
        if "seed" in payload:
            v = payload["seed"]
            if v is not None and not _is_int(v):
                errors.append(APIValidationError("seed", "must be an integer or null", "invalid_type"))

        # user (str)
        if "user" in payload and not isinstance(payload["user"], str):
            errors.append(APIValidationError("user", "must be a string", "invalid_type"))

        # strict: unknown top-level fields
        if self.strict:
            for key in payload:
                if key not in _REQUEST_KNOWN_FIELDS:
                    errors.append(APIValidationError(key, f"unknown field '{key}' (strict mode)", "unknown_field"))

        return errors

    # ------------------------------------------------------------------ helpers

    def _validate_message(self, msg: Any, path: str) -> list[APIValidationError]:
        errs: list[APIValidationError] = []
        if not isinstance(msg, dict):
            return [APIValidationError(path, "message must be an object", "invalid_type")]

        if "role" not in msg:
            errs.append(APIValidationError(f"{path}.role", "field is required", "missing_field"))
        elif not isinstance(msg["role"], str):
            errs.append(APIValidationError(f"{path}.role", "must be a string", "invalid_type"))
        elif msg["role"] not in _VALID_MESSAGE_ROLES:
            errs.append(APIValidationError(
                f"{path}.role",
                f"must be one of {sorted(_VALID_MESSAGE_ROLES)}",
                "invalid_value",
            ))

        has_content = "content" in msg and msg["content"] is not None
        has_tool_calls = "tool_calls" in msg and msg["tool_calls"] is not None
        role = msg.get("role")

        # Assistant may omit content if tool_calls present. Everyone else must have content.
        if not has_content and not (role == "assistant" and has_tool_calls):
            errs.append(APIValidationError(f"{path}.content", "field is required", "missing_field"))

        if has_content:
            content = msg["content"]
            if isinstance(content, str):
                pass
            elif isinstance(content, list):
                for i, part in enumerate(content):
                    if not isinstance(part, dict):
                        errs.append(APIValidationError(
                            f"{path}.content[{i}]", "content part must be an object", "invalid_type",
                        ))
                        continue
                    if "type" not in part:
                        errs.append(APIValidationError(
                            f"{path}.content[{i}].type", "field is required", "missing_field",
                        ))
            else:
                errs.append(APIValidationError(
                    f"{path}.content", "must be a string or list of content parts", "invalid_type",
                ))

        if has_tool_calls:
            tc = msg["tool_calls"]
            if not isinstance(tc, list):
                errs.append(APIValidationError(f"{path}.tool_calls", "must be a list", "invalid_type"))
            else:
                for i, call in enumerate(tc):
                    errs.extend(self._validate_tool_call(call, f"{path}.tool_calls[{i}]"))

        # tool role requires tool_call_id
        if role == "tool" and "tool_call_id" not in msg:
            errs.append(APIValidationError(f"{path}.tool_call_id", "required for role='tool'", "missing_field"))

        return errs

    def _validate_tool_call(self, call: Any, path: str) -> list[APIValidationError]:
        errs: list[APIValidationError] = []
        if not isinstance(call, dict):
            return [APIValidationError(path, "tool_call must be an object", "invalid_type")]
        for key in ("id", "type", "function"):
            if key not in call:
                errs.append(APIValidationError(f"{path}.{key}", "field is required", "missing_field"))
        if call.get("type") not in (None, "function"):
            errs.append(APIValidationError(f"{path}.type", "must be 'function'", "invalid_value"))
        fn = call.get("function")
        if fn is not None:
            if not isinstance(fn, dict):
                errs.append(APIValidationError(f"{path}.function", "must be an object", "invalid_type"))
            else:
                if "name" not in fn:
                    errs.append(APIValidationError(f"{path}.function.name", "field is required", "missing_field"))
                elif not isinstance(fn["name"], str) or not fn["name"]:
                    errs.append(APIValidationError(f"{path}.function.name", "must be a non-empty string", "invalid_type"))
                if "arguments" in fn and not isinstance(fn["arguments"], str):
                    errs.append(APIValidationError(
                        f"{path}.function.arguments",
                        "must be a JSON-encoded string",
                        "invalid_type",
                    ))
        return errs

    def _validate_tool(self, tool: Any, path: str) -> list[APIValidationError]:
        errs: list[APIValidationError] = []
        if not isinstance(tool, dict):
            return [APIValidationError(path, "tool must be an object", "invalid_type")]
        if tool.get("type") != "function":
            errs.append(APIValidationError(f"{path}.type", "must be 'function'", "invalid_value"))
        if "function" not in tool:
            errs.append(APIValidationError(f"{path}.function", "field is required", "missing_field"))
            return errs
        fn = tool["function"]
        if not isinstance(fn, dict):
            errs.append(APIValidationError(f"{path}.function", "must be an object", "invalid_type"))
            return errs
        if "name" not in fn:
            errs.append(APIValidationError(f"{path}.function.name", "field is required", "missing_field"))
        elif not isinstance(fn["name"], str) or not fn["name"]:
            errs.append(APIValidationError(f"{path}.function.name", "must be a non-empty string", "invalid_type"))
        if "description" in fn and not isinstance(fn["description"], str):
            errs.append(APIValidationError(f"{path}.function.description", "must be a string", "invalid_type"))
        if "parameters" in fn and not isinstance(fn["parameters"], dict):
            errs.append(APIValidationError(
                f"{path}.function.parameters",
                "must be a JSON-schema object",
                "invalid_type",
            ))
        return errs

    def _validate_tool_choice(self, v: Any, path: str) -> list[APIValidationError]:
        if isinstance(v, str):
            if v not in ("none", "auto", "required"):
                return [APIValidationError(path, "must be 'none', 'auto', or 'required'", "invalid_value")]
            return []
        if isinstance(v, dict):
            errs: list[APIValidationError] = []
            if v.get("type") != "function":
                errs.append(APIValidationError(f"{path}.type", "must be 'function'", "invalid_value"))
            fn = v.get("function")
            if not isinstance(fn, dict):
                errs.append(APIValidationError(f"{path}.function", "must be an object", "invalid_type"))
            elif "name" not in fn:
                errs.append(APIValidationError(f"{path}.function.name", "field is required", "missing_field"))
            return errs
        return [APIValidationError(path, "must be a string or tool-choice object", "invalid_type")]

    def _validate_response_format(self, v: Any, path: str) -> list[APIValidationError]:
        if not isinstance(v, dict):
            return [APIValidationError(path, "must be an object", "invalid_type")]
        rtype = v.get("type")
        if rtype not in ("text", "json_object", "json_schema"):
            return [APIValidationError(f"{path}.type", "must be 'text', 'json_object', or 'json_schema'", "invalid_value")]
        if rtype == "json_schema":
            if "json_schema" not in v:
                return [APIValidationError(f"{path}.json_schema", "field is required", "missing_field")]
            if not isinstance(v["json_schema"], dict):
                return [APIValidationError(f"{path}.json_schema", "must be an object", "invalid_type")]
        return []


# ---------------------------------------------------------------------------
# Response validator
# ---------------------------------------------------------------------------


class OpenAIChatResponseValidator:
    """Validator for OpenAI ``/v1/chat/completions`` response payloads.

    Handles both non-streaming completion objects (``object='chat.completion'``)
    and streaming chunks (``object='chat.completion.chunk'``).
    """

    def validate(self, payload: dict) -> list[APIValidationError]:
        errors: list[APIValidationError] = []
        if not isinstance(payload, dict):
            return [APIValidationError("", "payload must be a JSON object", "invalid_type")]

        obj = payload.get("object")
        is_chunk = obj == "chat.completion.chunk"

        for key in ("id", "object", "created", "model", "choices"):
            if key not in payload:
                errors.append(APIValidationError(key, "field is required", "missing_field"))

        if "id" in payload and not isinstance(payload["id"], str):
            errors.append(APIValidationError("id", "must be a string", "invalid_type"))
        if "object" in payload and payload["object"] not in ("chat.completion", "chat.completion.chunk"):
            errors.append(APIValidationError(
                "object", "must be 'chat.completion' or 'chat.completion.chunk'", "invalid_value",
            ))
        if "created" in payload and not _is_int(payload["created"]):
            errors.append(APIValidationError("created", "must be an integer unix timestamp", "invalid_type"))
        if "model" in payload and not isinstance(payload["model"], str):
            errors.append(APIValidationError("model", "must be a string", "invalid_type"))

        if "choices" in payload:
            if not isinstance(payload["choices"], list):
                errors.append(APIValidationError("choices", "must be a list", "invalid_type"))
            else:
                for i, ch in enumerate(payload["choices"]):
                    errors.extend(self._validate_choice(ch, f"choices[{i}]", is_chunk))

        if "usage" in payload and payload["usage"] is not None:
            errors.extend(self._validate_usage(payload["usage"], "usage"))

        return errors

    def _validate_choice(self, ch: Any, path: str, is_chunk: bool) -> list[APIValidationError]:
        errs: list[APIValidationError] = []
        if not isinstance(ch, dict):
            return [APIValidationError(path, "choice must be an object", "invalid_type")]

        if "index" not in ch:
            errs.append(APIValidationError(f"{path}.index", "field is required", "missing_field"))
        elif not _is_int(ch["index"]) or ch["index"] < 0:
            errs.append(APIValidationError(f"{path}.index", "must be a non-negative integer", "invalid_type"))

        msg_key = "delta" if is_chunk else "message"
        if msg_key not in ch:
            errs.append(APIValidationError(f"{path}.{msg_key}", "field is required", "missing_field"))
        else:
            errs.extend(self._validate_response_message(ch[msg_key], f"{path}.{msg_key}", is_chunk))

        # finish_reason: required for non-streaming, may be null in streaming chunks
        if "finish_reason" in ch:
            fr = ch["finish_reason"]
            if fr not in _VALID_FINISH_REASONS:
                errs.append(APIValidationError(
                    f"{path}.finish_reason",
                    f"must be one of {sorted(r for r in _VALID_FINISH_REASONS if r is not None)} or null",
                    "invalid_value",
                ))
        elif not is_chunk:
            errs.append(APIValidationError(f"{path}.finish_reason", "field is required", "missing_field"))

        return errs

    def _validate_response_message(self, msg: Any, path: str, is_chunk: bool) -> list[APIValidationError]:
        errs: list[APIValidationError] = []
        if not isinstance(msg, dict):
            return [APIValidationError(path, "must be an object", "invalid_type")]

        # In streaming deltas, role only appears on first chunk; content may be absent.
        if not is_chunk:
            if "role" not in msg:
                errs.append(APIValidationError(f"{path}.role", "field is required", "missing_field"))
            elif msg["role"] not in _VALID_MESSAGE_ROLES:
                errs.append(APIValidationError(f"{path}.role", "invalid role", "invalid_value"))

        if "role" in msg and not isinstance(msg["role"], str):
            errs.append(APIValidationError(f"{path}.role", "must be a string", "invalid_type"))

        if "content" in msg:
            c = msg["content"]
            if c is not None and not isinstance(c, str):
                errs.append(APIValidationError(f"{path}.content", "must be a string or null", "invalid_type"))

        if "tool_calls" in msg:
            tc = msg["tool_calls"]
            if not isinstance(tc, list):
                errs.append(APIValidationError(f"{path}.tool_calls", "must be a list", "invalid_type"))
            else:
                for i, call in enumerate(tc):
                    if not isinstance(call, dict):
                        errs.append(APIValidationError(
                            f"{path}.tool_calls[{i}]", "must be an object", "invalid_type",
                        ))
                        continue
                    # Streaming deltas may omit id/type; require function object presence if any.
                    if not is_chunk:
                        for key in ("id", "type", "function"):
                            if key not in call:
                                errs.append(APIValidationError(
                                    f"{path}.tool_calls[{i}].{key}", "field is required", "missing_field",
                                ))
                    if "function" in call and not isinstance(call["function"], dict):
                        errs.append(APIValidationError(
                            f"{path}.tool_calls[{i}].function", "must be an object", "invalid_type",
                        ))

        return errs

    def _validate_usage(self, u: Any, path: str) -> list[APIValidationError]:
        errs: list[APIValidationError] = []
        if not isinstance(u, dict):
            return [APIValidationError(path, "must be an object", "invalid_type")]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in u and (not _is_int(u[key]) or u[key] < 0):
                errs.append(APIValidationError(f"{path}.{key}", "must be a non-negative integer", "invalid_type"))
        return errs


# ---------------------------------------------------------------------------
# Public registry (additive hook for src/serving/__init__.py)
# ---------------------------------------------------------------------------

API_SHAPE_REGISTRY: dict[str, Any] = {
    "openai.chat.request": OpenAIChatRequestValidator,
    "openai.chat.response": OpenAIChatResponseValidator,
}


def _selftest() -> None:  # pragma: no cover - smoke
    v = OpenAIChatRequestValidator()
    assert v.validate({"model": "m", "messages": [{"role": "user", "content": "hi"}]}) == []


if __name__ == "__main__":  # pragma: no cover
    _selftest()
    print(json.dumps({"ok": True}))
