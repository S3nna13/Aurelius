"""Tool-message formatter for Aurelius chat surface.

Converts tool invocation results into ``Message`` objects compatible
with the ChatML and Llama-3 templates, preserving the tool ``call_id``
for correlation with the assistant's prior tool-call request and
handling large outputs via explicit truncation.

This parallels how Anthropic and OpenAI format ``tool_result`` blocks
in their respective chat APIs: each tool output becomes its own turn
with a dedicated role (``tool`` in ChatML / OpenAI, ``ipython`` in
Llama-3's agentic format) and an identifier that ties it back to the
call that produced it.

Security posture
----------------
Tool output is untrusted: it comes from sandboxed code, web fetches,
shell subprocesses, etc. Any chat-template control tokens appearing in
the content are **rejected**, not silently stripped — silent escaping
is a well-known vector for role-confusion prompt injection. Callers
that genuinely need to pass such literals must sanitize upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .chatml_template import (
    IM_END,
    IM_START,
    ChatMLFormatError,
    ChatMLTemplate,
    Message,
)
from .llama3_template import (
    BEGIN_OF_TEXT,
    END_HEADER,
    EOT,
    Llama3FormatError,
    Llama3Template,
    START_HEADER,
)

__all__ = ["ToolResult", "ToolMessageFormatter"]


_TRUNCATION_MARKER = "\n...[truncated: {dropped} chars omitted]"

_CHATML_CONTROL = (IM_START, IM_END)
_LLAMA3_CONTROL = (BEGIN_OF_TEXT, START_HEADER, END_HEADER, EOT)

_SUPPORTED_TEMPLATES = frozenset({"chatml", "llama3"})


@dataclass
class ToolResult:
    """A single completed tool invocation result.

    Attributes:
        call_id: Correlator matching the assistant's tool_use id.
        name: Logical tool name (e.g. ``"bash"``, ``"read_file"``).
        content: Raw textual output of the tool (may be large).
        is_error: True if the tool raised / returned a failure.
    """

    call_id: str
    name: str
    content: str
    is_error: bool = False


class ToolMessageFormatter:
    """Format ``ToolResult`` objects as chat-template messages.

    Args:
        template: Either ``"chatml"`` or ``"llama3"``. Unknown values
            raise ``ValueError``.
        max_content_chars: Hard cap (in characters) on the *output*
            content body (after the header prefix, before truncation
            marker). Must be > 0. Content exceeding this length is
            truncated and a marker is appended indicating how many
            characters were dropped.
    """

    def __init__(
        self,
        template: str = "chatml",
        max_content_chars: int = 16384,
    ) -> None:
        if template not in _SUPPORTED_TEMPLATES:
            raise ValueError(
                f"unknown template {template!r}; supported: "
                f"{sorted(_SUPPORTED_TEMPLATES)}"
            )
        if not isinstance(max_content_chars, int) or isinstance(
            max_content_chars, bool
        ):
            raise ValueError(
                f"max_content_chars must be int, got "
                f"{type(max_content_chars).__name__}"
            )
        if max_content_chars <= 0:
            raise ValueError(
                f"max_content_chars must be > 0, got {max_content_chars}"
            )
        self.template = template
        self.max_content_chars = max_content_chars

    # ------------------------------------------------------------------ core

    def _role(self) -> str:
        # ChatML has no "ipython" role; Llama-3's agentic format uses it.
        return "ipython" if self.template == "llama3" else "tool"

    def _control_tokens(self) -> tuple:
        return _LLAMA3_CONTROL if self.template == "llama3" else _CHATML_CONTROL

    def _format_error(self, err_cls):
        # Choose the right error class so callers get a consistent type
        # per template.
        return err_cls

    def _validate_no_control_tokens(self, text: str) -> None:
        if self.template == "llama3":
            for tok in _LLAMA3_CONTROL:
                if tok in text:
                    raise Llama3FormatError(
                        f"tool content contains Llama-3 control token "
                        f"{tok!r}; rejecting to prevent role-confusion "
                        "attack. Sanitize upstream if legitimate."
                    )
        else:
            for tok in _CHATML_CONTROL:
                if tok in text:
                    raise ChatMLFormatError(
                        f"tool content contains ChatML control token "
                        f"{tok!r}; rejecting to prevent role-confusion "
                        "attack. Sanitize upstream if legitimate."
                    )

    def _truncate(self, body: str) -> str:
        if len(body) <= self.max_content_chars:
            return body
        dropped = len(body) - self.max_content_chars
        return body[: self.max_content_chars] + _TRUNCATION_MARKER.format(
            dropped=dropped
        )

    # ---------------------------------------------------------------- format

    def format(self, tool_result: ToolResult) -> Message:
        """Format a single tool result as a ``Message``.

        The emitted content is:

            [tool: <name> id=<call_id>]\n<body>[truncation marker]

        For error results the body is prefixed with ``[ERROR]``. The
        role is ``tool`` for ChatML and ``ipython`` for Llama-3.
        """
        if not isinstance(tool_result, ToolResult):
            raise TypeError(
                f"tool_result must be ToolResult, got "
                f"{type(tool_result).__name__}"
            )
        if not isinstance(tool_result.call_id, str):
            raise TypeError("ToolResult.call_id must be str")
        if not isinstance(tool_result.name, str):
            raise TypeError("ToolResult.name must be str")
        if not isinstance(tool_result.content, str):
            raise TypeError("ToolResult.content must be str")

        # Reject control tokens in *any* user-controlled field so they
        # cannot escape the envelope via call_id / name either.
        self._validate_no_control_tokens(tool_result.call_id)
        self._validate_no_control_tokens(tool_result.name)
        self._validate_no_control_tokens(tool_result.content)

        header = f"[tool: {tool_result.name} id={tool_result.call_id}]"
        body = tool_result.content
        if tool_result.is_error:
            body = f"[ERROR] {body}" if body else "[ERROR]"

        body = self._truncate(body)

        if body == "" and not tool_result.is_error:
            # Preserve an explicit empty body — still emit the header so
            # the call_id survives the round-trip.
            content = header
        else:
            content = f"{header}\n{body}"

        return Message(role=self._role(), content=content, name=tool_result.name)

    def format_batch(self, results: List[ToolResult]) -> List[Message]:
        """Format a list of tool results, preserving order."""
        if not isinstance(results, list):
            raise TypeError("results must be a list[ToolResult]")
        return [self.format(r) for r in results]

    # --------------------------------------------------------- prompt turn

    def to_prompt_turn(
        self,
        results: List[ToolResult],
        template_obj: Optional[object] = None,
    ) -> str:
        """Render results as a ready-to-feed prompt fragment.

        If ``template_obj`` is supplied it must be a ``ChatMLTemplate``
        or ``Llama3Template`` instance compatible with ``self.template``;
        otherwise a fresh instance is constructed. The returned string
        is the ``encode()`` output of that template over the formatted
        messages (no generation prompt appended).
        """
        messages = self.format_batch(results)

        if template_obj is None:
            template_obj = (
                Llama3Template() if self.template == "llama3" else ChatMLTemplate()
            )
        else:
            expected = (
                Llama3Template if self.template == "llama3" else ChatMLTemplate
            )
            if not isinstance(template_obj, expected):
                raise TypeError(
                    f"template_obj must be {expected.__name__} for "
                    f"template={self.template!r}, got "
                    f"{type(template_obj).__name__}"
                )

        # ChatML's VALID_ROLES does not include "ipython"; the formatter
        # only emits "ipython" when self.template == "llama3", so this
        # is consistent by construction.
        return template_obj.encode(messages, add_generation_prompt=False)
