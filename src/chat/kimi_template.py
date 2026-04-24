"""Kimi-style chat template encoder for Aurelius.

Implements the multi-role token wire format used by Moonshot/Kimi models:

    <|im_system|>...<|im_middle|>...<|im_end|>
    <|im_user|>...<|im_middle|>...<|im_end|>
    <|im_assistant|>...<|im_middle|>...<|im_end|>

Tool declarations are injected via a special ``tool_declare`` system block.
Tool calls are wrapped in section/call delimiters.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Role tokens
# ---------------------------------------------------------------------------

IM_USER = "<|im_user|>"
IM_ASSISTANT = "<|im_assistant|>"
IM_SYSTEM = "<|im_system|>"
IM_MIDDLE = "<|im_middle|>"
IM_END = "<|im_end|>"

# ---------------------------------------------------------------------------
# Tool call tokens
# ---------------------------------------------------------------------------

TOOL_CALLS_BEGIN = "<|tool_calls_section_begin|>"
TOOL_CALLS_END = "<|tool_calls_section_end|>"
TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_ARG_BEGIN = "<|tool_call_argument_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"

# Role → token mapping
_ROLE_TOKEN: dict[str, str] = {
    "user": IM_USER,
    "assistant": IM_ASSISTANT,
    "system": IM_SYSTEM,
    "observation": IM_SYSTEM,  # observations share system token in Kimi style
}


def render_content(msg: dict) -> str:
    """Extract displayable text from a message dict.

    Args:
        msg: A message dict with a ``content`` field that is either a
            plain string or a list of content-block dicts.

    Returns:
        The joined text content as a plain string.
    """
    content = msg.get("content") or ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def set_role(message: dict) -> str:
    """Build the role-opener token sequence for *message*.

    The format is ``<|im_{role}|>{name}<|im_middle|>`` where *name*
    defaults to the role string when not explicitly specified.

    Args:
        message: A message dict with at minimum a ``role`` key.

    Returns:
        The role-opener string.
    """
    role = message.get("role", "user")
    name = message.get("name") or role
    token = _ROLE_TOKEN.get(role, IM_USER)
    return f"{token}{name}{IM_MIDDLE}"


class KimiChatTemplate:
    """Kimi / Moonshot chat template.

    Encodes a list of messages into a single prompt string using the
    Kimi multi-role token format.  Supports:

    * System, user, and assistant roles.
    * Tool declaration injection.
    * Tool call encoding.
    * History/suffix splitting for partial-generation use-cases.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_hist_suffix(
        self,
        messages: list[dict],
        preserve_thinking: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        """Split *messages* into history and a suffix starting at the
        last non-tool-call assistant message.

        The last assistant message that does NOT have ``tool_calls``
        becomes the start of the *suffix*; everything before it is the
        *hist*.

        Args:
            messages: The full message list.
            preserve_thinking: When True, thinking tokens in the suffix
                are preserved (currently informational — callers decide
                how to use this flag).

        Returns:
            A ``(hist, suffix)`` tuple where both are sub-lists of
            *messages*.  If no qualifying assistant message is found,
            returns ``(messages, [])``.
        """
        split_idx: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                split_idx = i
                break

        if split_idx is None:
            return list(messages), []

        hist = list(messages[:split_idx])
        suffix = list(messages[split_idx:])
        return hist, suffix

    def encode(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tools_ts_str: str | None = None,
        preserve_thinking: bool = False,
    ) -> str:
        """Encode a list of messages into a Kimi prompt string.

        Args:
            messages: List of message dicts.
            tools: Optional list of tool-schema dicts.  When provided,
                a ``tool_declare`` system block is prepended.
            tools_ts_str: Optional pre-rendered tool declaration string.
                Takes precedence over ``tools`` when both are supplied.
            preserve_thinking: Passed through to
                :meth:`split_hist_suffix` (informational).

        Returns:
            The fully-formatted prompt string.
        """
        parts: list[str] = []

        # Prepend tool declarations.
        if tools:
            tool_text = tools_ts_str if tools_ts_str is not None else self.render_tools_as_json(tools)
            parts.append(
                f"{IM_SYSTEM}tool_declare{IM_MIDDLE}{tool_text}{IM_END}\n"
            )

        for msg in messages:
            role_opener = set_role(msg)
            content_text = render_content(msg)
            tool_calls_text = ""

            if msg.get("tool_calls"):
                tool_calls_text = self.render_toolcalls(msg)

            parts.append(f"{role_opener}{content_text}{tool_calls_text}{IM_END}")

        return "".join(parts)

    def render_toolcalls(self, message: dict) -> str:
        """Encode the ``tool_calls`` list of *message* as a token block.

        Args:
            message: A message dict that has a ``tool_calls`` list.

        Returns:
            A string containing all tool calls wrapped in section
            delimiters.
        """
        tool_calls = message.get("tool_calls", [])
        call_parts: list[str] = []
        for tc in tool_calls:
            call_id = tc.get("id", "")
            fn = tc.get("function", tc)
            arguments = fn.get("arguments", "")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            call_parts.append(
                f"{TOOL_CALL_BEGIN}{call_id}{TOOL_CALL_ARG_BEGIN}{arguments}{TOOL_CALL_END}"
            )
        return f"{TOOL_CALLS_BEGIN}{''.join(call_parts)}{TOOL_CALLS_END}"

    def render_tools_as_json(self, tools: list[dict]) -> str:
        """Render *tools* as a pretty-printed JSON string.

        Args:
            tools: List of tool-schema dicts.

        Returns:
            JSON-encoded string with 2-space indentation.
        """
        return json.dumps(tools, indent=2)


# ---------------------------------------------------------------------------
# Module-level singleton and registry entry
# ---------------------------------------------------------------------------

CHAT_TEMPLATE_REGISTRY: dict[str, Any] = {}
CHAT_TEMPLATE_REGISTRY["kimi"] = KimiChatTemplate()

__all__ = [
    "IM_USER",
    "IM_ASSISTANT",
    "IM_SYSTEM",
    "IM_MIDDLE",
    "IM_END",
    "TOOL_CALLS_BEGIN",
    "TOOL_CALLS_END",
    "TOOL_CALL_BEGIN",
    "TOOL_CALL_ARG_BEGIN",
    "TOOL_CALL_END",
    "render_content",
    "set_role",
    "KimiChatTemplate",
    "CHAT_TEMPLATE_REGISTRY",
]
