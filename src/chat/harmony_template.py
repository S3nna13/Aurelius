"""Harmony Response Format — GPT-OSS-120B (arXiv:2508.10925).
Chat template with reasoning scratchpad, tool-call format, and system prompt support.
Both gpt-oss-120b and gpt-oss-20b were trained exclusively on this format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


class HarmonyFormatError(ValueError):
    """Raised for any malformed Harmony payload or rejected input."""


@dataclass
class HarmonyMessage:
    """Thin compat shim kept for registry consumers that stored HarmonyMessage.

    New code should pass plain dicts to HarmonyTemplate.render().
    """

    role: str
    content: str
    channel: str | None = None
    name: str | None = None


@dataclass
class HarmonyTemplate:
    """Harmony chat template used by GPT-OSS-120B / GPT-OSS-20B.

    Wire format differs from ChatML:
      - System prompt:   <|system|>...<|end_system|>
      - User turns:      <|user|>...<|end_user|>
      - Assistant turns: <|assistant|>...<|end_assistant|>
      - Tool results:    <tool_result>...</tool_result>
      - Chain-of-thought scratchpad inside assistant turns: <think>...</think>
      - Tool calls inside assistant turns: <tool_call>...</tool_call>
      - Sequence starts with <|begin_of_text|>; optional <|end_of_text|> at end.

    Reference: OpenAI open-weights GPT-OSS family (arXiv:2508.10925).
    """

    bos: str = "<|begin_of_text|>"
    eos: str = "<|end_of_text|>"
    think_open: str = "<think>"
    think_close: str = "</think>"
    add_eos: bool = False

    # Keep a stable name attribute so CHAT_TEMPLATE_REGISTRY consumers can
    # inspect the template's canonical identifier.
    name: str = "harmony"

    def render(self, messages: list[dict]) -> str:
        """Render a list of message dicts into a Harmony-format string.

        Args:
            messages: Ordered list of dicts with at least a ``role`` key.
                      Supported roles: system, user, assistant, tool.
                      Assistant dicts may include:
                        - ``thinking`` (str): chain-of-thought; wrapped in
                          <think>...</think>.
                        - ``tool_calls`` (list[str]): each entry wrapped in
                          <tool_call>...</tool_call>.
                      Any unknown role raises ``ValueError``.

        Returns:
            Harmony-formatted string beginning with <|begin_of_text|>.
        """
        parts = [self.bos]

        for msg in messages:
            role = msg.get("role", "")
            if role not in VALID_ROLES:
                raise ValueError(f"Unknown role {role!r}. Valid roles: {sorted(VALID_ROLES)}")
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"<|system|>\n{content}\n<|end_system|>")

            elif role == "user":
                parts.append(f"<|user|>\n{content}\n<|end_user|>")

            elif role == "assistant":
                thinking = msg.get("thinking", "")
                tool_calls = msg.get("tool_calls", [])
                body = ""
                if thinking:
                    body += f"{self.think_open}{thinking}{self.think_close}\n"
                if tool_calls:
                    for tc in tool_calls:
                        body += f"<tool_call>{tc}</tool_call>\n"
                body += content
                parts.append(f"<|assistant|>\n{body.rstrip()}\n<|end_assistant|>")

            elif role == "tool":
                parts.append(f"<tool_result>\n{content}\n</tool_result>")

        rendered = "\n".join(parts)
        if self.add_eos:
            rendered += self.eos
        return rendered

    def parse_roles(self, rendered: str) -> list[str]:
        """Extract the role sequence from a rendered Harmony string.

        Used for round-trip validation: render a conversation, then confirm
        the role sequence survives the serialisation unchanged.

        Args:
            rendered: A string previously produced by ``render()``.

        Returns:
            Ordered list of role strings as they appear in ``rendered``.
        """
        roles: list[str] = []
        if "<|system|>" in rendered:
            roles.append("system")
        for m in re.finditer(r"<\|user\|>|<\|assistant\|>|<tool_result>", rendered):
            tag = m.group()
            if tag == "<|user|>":
                roles.append("user")
            elif tag == "<|assistant|>":
                roles.append("assistant")
            elif tag == "<tool_result>":
                roles.append("tool")
        return roles
