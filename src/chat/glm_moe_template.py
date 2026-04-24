"""GLM-MoE chat template encoder/decoder for Aurelius.

Implements the GLM-style wire format used by ChatGLM models:

    [gMASK]<sop><|system|>\n{system_content}<|user|>\n{content}<|assistant|>\n{content}

Supports thinking-indices-based Chain-of-Thought (CoT) via reasoning_content,
tool call encoding/decoding, and tool schema rendering.
"""

from __future__ import annotations

import re
from typing import Any

GMASK_TOKEN = "[gMASK]"
SOP_TOKEN = "<sop>"

ROLES: dict[str, str] = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "system": "<|system|>",
    "observation": "<|observation|>",
}

# Tool call pattern
_TOOL_CALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>",
    re.DOTALL,
)
_ARG_KEY_RE = re.compile(r"<arg_key>(.*?)</arg_key>", re.DOTALL)
_ARG_VALUE_RE = re.compile(r"<arg_value>(.*?)</arg_value>", re.DOTALL)
_TOOL_NAME_RE = re.compile(
    r"<tool_call>\s*([^\s<]+)",
    re.DOTALL,
)


def visible_text(content: str | list[dict]) -> str:
    """Extract visible text from a content field.

    Args:
        content: Either a plain string or a list of content-block dicts.
            For lists, only blocks with ``type == "text"`` are joined.

    Returns:
        The extracted/joined text as a plain string.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


class GlmMoeTemplate:
    """GLM-MoE chat template.

    Encodes a sequence of messages into a single string suitable for
    tokenisation by a GLM-family model.  Supports:

    * System, user, assistant, and observation roles.
    * Tool call encoding and decoding.
    * Optional Chain-of-Thought wrapping via ``reasoning_content``.
    * Lazy tool-schema injection controlled by ``defer_loading``.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        enable_thinking: bool = True,
        defer_loading: bool = False,
    ) -> str:
        """Encode a list of messages into a GLM-MoE prompt string.

        Args:
            messages: List of message dicts.  Each dict must have at
                minimum a ``role`` key.  Supported roles: ``system``,
                ``user``, ``assistant``, ``observation``.
            tools: Optional list of tool-schema dicts.  When provided
                and ``defer_loading`` is False, a synthetic system
                message describing the tools is prepended.
            enable_thinking: When True, ``reasoning_content`` on
                assistant messages is wrapped in ``<think>…</think>``.
            defer_loading: When True, tool descriptions are NOT
                prepended even if ``tools`` is given (the caller will
                inject them later).

        Returns:
            The fully-formatted prompt string.
        """
        parts: list[str] = [GMASK_TOKEN + SOP_TOKEN]

        # Optionally prepend tool descriptions as a synthetic system message.
        if tools and not defer_loading:
            tool_desc = "\n".join(self.render_tool_schema(t) for t in tools)
            parts.append(f"{ROLES['system']}\n{tool_desc}")

        for msg in messages:
            role = msg.get("role", "")
            content = visible_text(msg.get("content") or "")

            if role == "system":
                parts.append(f"{ROLES['system']}\n{content}")

            elif role == "user":
                parts.append(f"{ROLES['user']}\n{content}")

            elif role == "assistant":
                reasoning = msg.get("reasoning_content", "")
                tool_calls = msg.get("tool_calls")

                if tool_calls:
                    # Encode each tool call.
                    for tc in tool_calls:
                        fn = tc.get("function", tc)
                        fn_name = fn.get("name", "")
                        arguments: dict[str, Any] = fn.get("arguments") or {}
                        if isinstance(arguments, str):
                            # Try to keep as-is if it's raw JSON string
                            parts.append(
                                f"{ROLES['assistant']}\n<tool_call>{fn_name}"
                                f"<arg_key>__raw__</arg_key>"
                                f"<arg_value>{arguments}</arg_value></tool_call>"
                            )
                        else:
                            arg_parts = "".join(
                                f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>"
                                for k, v in arguments.items()
                            )
                            parts.append(
                                f"{ROLES['assistant']}\n<tool_call>{fn_name}"
                                f"{arg_parts}</tool_call>"
                            )
                else:
                    if reasoning and enable_thinking:
                        parts.append(
                            f"{ROLES['assistant']}\n"
                            f"<think>\n{reasoning}\n</think>\n{content}"
                        )
                    else:
                        parts.append(f"{ROLES['assistant']}\n{content}")

            elif role == "observation":
                parts.append(f"{ROLES['observation']}\n{content}")

            else:
                # Unknown role — pass through with a best-effort prefix.
                parts.append(f"{content}")

        return "".join(parts)

    def decode_tool_call(self, text: str) -> dict | None:
        """Parse the first ``<tool_call>…</tool_call>`` block in *text*.

        Args:
            text: Raw text that may contain a tool-call block.

        Returns:
            A dict ``{"name": str, "arguments": dict}`` on success, or
            ``None`` if no valid tool-call block is found.
        """
        match = _TOOL_CALL_RE.search(text)
        if not match:
            return None

        inner = match.group(1)

        # Extract tool name (first non-whitespace token before any tag).
        name_match = re.match(r"\s*([^\s<]+)", inner)
        name = name_match.group(1) if name_match else ""

        keys = _ARG_KEY_RE.findall(inner)
        values = _ARG_VALUE_RE.findall(inner)
        arguments: dict[str, str] = dict(zip(keys, values))

        return {"name": name, "arguments": arguments}

    def render_tool_schema(self, tool: dict) -> str:
        """Render a single tool schema as a human-readable description.

        Args:
            tool: A tool-schema dict with at minimum a ``name`` key and
                optionally ``description`` and ``parameters``.

        Returns:
            A compact string describing the tool.
        """
        name = tool.get("name", "")
        description = tool.get("description", "")
        parameters = tool.get("parameters", {})
        props = parameters.get("properties", {})
        param_names = ", ".join(props.keys()) if props else "(none)"
        return f"{name}: {description}\nParameters: {param_names}"


# ---------------------------------------------------------------------------
# Module-level singleton and registry entry
# ---------------------------------------------------------------------------

CHAT_TEMPLATE_REGISTRY: dict[str, Any] = {}
CHAT_TEMPLATE_REGISTRY["glm_moe"] = GlmMoeTemplate()

__all__ = [
    "GMASK_TOKEN",
    "SOP_TOKEN",
    "ROLES",
    "visible_text",
    "GlmMoeTemplate",
    "CHAT_TEMPLATE_REGISTRY",
]
