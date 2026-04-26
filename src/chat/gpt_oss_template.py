"""GPT-OSS-style chat template encoder/decoder for Aurelius.

Implements the channel-based wire format used by OpenAI's open-source
reasoning models:

    <|start|>{role}<|channel|>{channel}<|message|>{content}<|end|>

Reasoning is separated into an ``analysis`` channel; final answers use
the ``final`` channel.  Tool calls use ``<|call|>`` / ``<|return|>``
delimiters instead of ``<|end|>``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .typescript_tool_renderer import render_namespace

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

START = "<|start|>"
CHANNEL = "<|channel|>"
MESSAGE = "<|message|>"
END = "<|end|>"
CALL = "<|call|>"
RETURN = "<|return|>"

# ---------------------------------------------------------------------------
# Channel / role constants
# ---------------------------------------------------------------------------

ANALYSIS_CHANNEL = "analysis"
FINAL_CHANNEL = "final"

# Decoder pattern for tool-call blocks.
_TOOL_CALL_DECODE_RE = re.compile(
    r"<\|start\|>assistant to=functions\.(?P<name>[^\s<|]+)"
    r"<\|channel\|>[^<]*<\|message\|>(?P<args>.*?)<\|call\|>",
    re.DOTALL,
)


class GptOssTemplate:
    """GPT-OSS channel-based chat template.

    Encodes a list of messages into the GPT-OSS wire format.  Supports:

    * Developer (replaces system), user, and assistant roles.
    * Reasoning / analysis channel with look-ahead suppression.
    * Tool call encoding and decoding.
    * TypeScript namespace rendering for tool schemas.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        builtin_tools: list[Any] | None = None,
        reasoning_effort: str = "medium",
        add_generation_prompt: bool = True,
    ) -> str:
        """Encode *messages* into a GPT-OSS prompt string.

        Args:
            messages: List of message dicts.  Supported roles:
                ``developer``, ``user``, ``assistant``, ``tool``.
            tools: Optional list of tool-schema dicts.  When provided,
                a TypeScript namespace block is prepended as a developer
                message.
            builtin_tools: Reserved for future use (ignored for now).
            reasoning_effort: One of ``"low"``, ``"medium"``, ``"high"``.
                Currently informational; passed through in the prompt.
            add_generation_prompt: When True, a bare
                ``<|start|>assistant<|channel|>`` opener is appended to
                signal the model should continue.

        Returns:
            The fully-formatted prompt string.
        """
        parts: list[str] = []

        # Prepend tool namespace as a developer message.
        if tools:
            ts_block = render_namespace("functions", tools)
            parts.append(f"{START}developer{CHANNEL}{FINAL_CHANNEL}{MESSAGE}{ts_block}{END}")

        # Build a look-ahead index: for each assistant message that has
        # reasoning content, is there a LATER non-tool-call assistant msg?
        assistant_indices = [
            i
            for i, m in enumerate(messages)
            if m.get("role") == "assistant" and not m.get("tool_calls")
        ]

        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            content = self._get_content(msg)

            if role == "developer":
                parts.append(f"{START}developer{CHANNEL}{FINAL_CHANNEL}{MESSAGE}{content}{END}")

            elif role == "system":
                # Treat system as developer.
                parts.append(f"{START}developer{CHANNEL}{FINAL_CHANNEL}{MESSAGE}{content}{END}")

            elif role == "user":
                parts.append(f"{START}user{CHANNEL}{FINAL_CHANNEL}{MESSAGE}{content}{END}")

            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # Emit one call block per tool call.
                    for tc in tool_calls:
                        fn = tc.get("function", tc)
                        fn_name = fn.get("name", "")
                        args = fn.get("arguments", {})
                        if isinstance(args, dict):
                            args_str = json.dumps(args)
                        else:
                            args_str = str(args)
                        parts.append(
                            f"{START}assistant to=functions.{fn_name}"
                            f"{CHANNEL}commentary json{MESSAGE}{args_str}{CALL}"
                        )
                else:
                    reasoning = msg.get("reasoning_content", "") or msg.get("thinking", "") or ""

                    # Look-ahead: is there a later non-tool-call assistant msg?
                    future_assistant = any(j > idx for j in assistant_indices)

                    if reasoning and not future_assistant:
                        # Last CoT message — emit analysis channel first.
                        parts.append(
                            f"{START}assistant{CHANNEL}{ANALYSIS_CHANNEL}{MESSAGE}{reasoning}{END}"
                        )

                    # Always emit final channel.
                    parts.append(f"{START}assistant{CHANNEL}{FINAL_CHANNEL}{MESSAGE}{content}{END}")

            elif role == "tool":
                # Tool result / observation.
                fn_name = msg.get("name", msg.get("tool_call_id", "tool"))
                parts.append(f"{START}functions.{fn_name}{CHANNEL}result{MESSAGE}{content}{RETURN}")

            else:
                # Unknown role — pass through as user.
                parts.append(f"{START}user{CHANNEL}{FINAL_CHANNEL}{MESSAGE}{content}{END}")

        if add_generation_prompt:
            parts.append(f"{START}assistant{CHANNEL}{FINAL_CHANNEL}{MESSAGE}")

        return "".join(parts)

    def decode_tool_call(self, text: str) -> dict | None:
        """Parse the first tool call block in *text*.

        Looks for the pattern::

            <|start|>assistant to=functions.{name}<|channel|>…<|message|>{args}<|call|>

        Args:
            text: Raw text that may contain a tool-call block.

        Returns:
            A dict ``{"name": str, "arguments": dict | str}`` on
            success, or ``None`` if no valid tool-call block is found.
        """
        match = _TOOL_CALL_DECODE_RE.search(text)
        if not match:
            return None

        name = match.group("name")
        args_str = match.group("args").strip()

        try:
            arguments = json.loads(args_str)
        except (json.JSONDecodeError, ValueError):
            arguments = args_str

        return {"name": name, "arguments": arguments}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_content(msg: dict) -> str:
        """Extract the text content from *msg*."""
        content = msg.get("content") or ""
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)


# ---------------------------------------------------------------------------
# Module-level singleton and registry entry
# ---------------------------------------------------------------------------

CHAT_TEMPLATE_REGISTRY: dict[str, Any] = {}
CHAT_TEMPLATE_REGISTRY["gpt_oss"] = GptOssTemplate()

__all__ = [
    "START",
    "CHANNEL",
    "MESSAGE",
    "END",
    "CALL",
    "RETURN",
    "ANALYSIS_CHANNEL",
    "FINAL_CHANNEL",
    "GptOssTemplate",
    "CHAT_TEMPLATE_REGISTRY",
]
