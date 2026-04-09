"""Tool use / function calling interface for Aurelius — GPT-4-style structured tool invocation."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ToolSchema:
    """Schema describing a callable tool (function) for prompt injection and validation."""

    name: str
    description: str
    parameters: dict  # JSON Schema dict describing parameters
    required: list[str] = field(default_factory=list)

    def to_prompt_str(self) -> str:
        """Format as a readable description for prompt injection."""
        lines = [
            f"Tool: {self.name}",
            f"  Description: {self.description}",
        ]
        if self.parameters:
            lines.append(f"  Parameters: {json.dumps(self.parameters)}")
        if self.required:
            lines.append(f"  Required: {', '.join(self.required)}")
        return "\n".join(lines)


@dataclass
class ToolCall:
    """Represents a single tool invocation parsed from model output."""

    tool_name: str
    arguments: dict
    call_id: str = ""  # unique call identifier

    def __post_init__(self) -> None:
        if not self.call_id:
            self.call_id = str(uuid.uuid4())[:8]

    def to_json(self) -> str:
        """Return JSON string representation."""
        return json.dumps({
            "name": self.tool_name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        })


@dataclass
class ToolResult:
    """Result returned from a tool execution."""

    call_id: str
    tool_name: str
    result: str  # string result from tool execution
    error: str | None = None

    def to_prompt_str(self) -> str:
        """Format for injection back into context."""
        if self.error:
            content = f"ERROR: {self.error}"
        else:
            content = self.result
        return (
            f"<tool_result call_id=\"{self.call_id}\" tool=\"{self.tool_name}\">"
            f"{content}"
            f"</tool_result>"
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>(.*?)</tool_call>",
    re.DOTALL,
)


def parse_tool_call(text: str) -> ToolCall | None:
    """Parse a tool call from model output.

    Expected format::

        <tool_call>{"name": "...", "arguments": {...}}</tool_call>

    Returns:
        ToolCall if a valid tool call is found, None otherwise.
    """
    match = _TOOL_CALL_PATTERN.search(text)
    if not match:
        return None

    raw = match.group(1).strip()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    name = data.get("name")
    if not isinstance(name, str) or not name:
        return None

    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}

    call_id = data.get("call_id", "")
    return ToolCall(tool_name=name, arguments=arguments, call_id=call_id)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_tools_for_prompt(tools: list[ToolSchema]) -> str:
    """Format tool descriptions for injection into a system prompt.

    Returns a multi-line string with all tool schemas.
    """
    if not tools:
        return "No tools available."
    sections = ["Available Tools:"]
    for tool in tools:
        sections.append(tool.to_prompt_str())
    return "\n".join(sections)


def format_tool_result_for_context(result: ToolResult) -> str:
    """Format result for re-injection into conversation.

    Returns: ``<tool_result call_id="{id}">{result}</tool_result>``
    """
    content = result.result if result.error is None else f"ERROR: {result.error}"
    return f'<tool_result call_id="{result.call_id}">{content}</tool_result>'


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry that maps tool names to their schemas and callable implementations."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSchema, Callable]] = {}

    def register(self, schema: ToolSchema, fn: Callable) -> None:
        """Register a tool with its schema and callable."""
        self._tools[schema.name] = (schema, fn)

    def get_schema(self, name: str) -> ToolSchema | None:
        """Return the ToolSchema for the named tool, or None if not registered."""
        entry = self._tools.get(name)
        return entry[0] if entry is not None else None

    def execute(self, call: ToolCall) -> ToolResult:
        """Look up tool by name, call fn(**call.arguments), catch exceptions.

        Returns ToolResult with result string on success, or error string on failure.
        """
        entry = self._tools.get(call.tool_name)
        if entry is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                result="",
                error=f"Tool '{call.tool_name}' not found in registry.",
            )

        _, fn = entry
        try:
            raw = fn(**call.arguments)
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                result=str(raw),
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                result="",
                error=str(exc),
            )

    def all_schemas(self) -> list[ToolSchema]:
        """Return list of all registered ToolSchema objects."""
        return [schema for schema, _ in self._tools.values()]


# ---------------------------------------------------------------------------
# Tool use session
# ---------------------------------------------------------------------------


class ToolUseSession:
    """Manages a multi-turn conversation with tool calls.

    Each call to ``run`` drives a prompt-generate-execute cycle:
    1. Build prompt with tool schemas + user message.
    2. Generate response; if it contains ``<tool_call>``, parse and execute the tool.
    3. Inject the result back and generate again (up to max_tool_rounds).
    4. Return the final text response.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenize_fn: Callable[[str], list[int]],
        registry: ToolRegistry,
        max_new_tokens: int = 64,
        max_tool_rounds: int = 5,
    ) -> None:
        self.model = model
        self.tokenize_fn = tokenize_fn
        self.registry = registry
        self.max_new_tokens = max_new_tokens
        self.max_tool_rounds = max_tool_rounds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_message: str, system_prompt: str = "") -> str:
        """Run the tool-use loop for a single user message.

        Args:
            user_message: The user's input text.
            system_prompt: Optional system prompt prefix.

        Returns:
            Final assistant text response.
        """
        # Build initial prompt
        tool_descriptions = format_tools_for_prompt(self.registry.all_schemas())

        parts: list[str] = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        parts.append(tool_descriptions)
        parts.append(f"User: {user_message}")
        parts.append("Assistant:")
        prompt = "\n\n".join(parts)

        for _round in range(self.max_tool_rounds):
            response = self._generate(prompt)

            tool_call = parse_tool_call(response)
            if tool_call is None:
                # No tool call — return the response as final answer
                return response.strip()

            # Execute the tool and inject the result
            result = self.registry.execute(tool_call)
            result_str = format_tool_result_for_context(result)

            # Append the model's response (including the tool call) and the result
            prompt = prompt + response + "\n" + result_str + "\nAssistant:"

        # Max rounds reached — generate one final response
        return self._generate(prompt).strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """Greedy decode up to max_new_tokens tokens from the prompt.

        Returns decoded text string.
        """
        input_ids = self.tokenize_fn(prompt)
        current_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # (1, S)

        generated: list[int] = []

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                output = self.model(current_ids)   # (loss, logits, past_key_values)
                logits = output[1]                 # (1, S, V)
                next_token = int(torch.argmax(logits[0, -1, :]).item())
                generated.append(next_token)
                current_ids = torch.cat(
                    [current_ids, torch.tensor([[next_token]], dtype=torch.long)],
                    dim=1,
                )

        # Decode bytes — tokenize_fn produces UTF-8 byte values
        raw_bytes = bytes([b & 0xFF for b in generated])
        return raw_bytes.decode("utf-8", errors="replace")
