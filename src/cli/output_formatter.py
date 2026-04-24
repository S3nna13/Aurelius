"""Output formatter — render model responses in various display formats."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum


class OutputFormat(str, Enum):
    PLAIN = "plain"
    RICH_MARKDOWN = "rich_markdown"
    JSON = "json"
    COMPACT = "compact"
    STREAM = "stream"


@dataclass
class FormatterConfig:
    format: OutputFormat
    max_width: int = 120
    highlight_code: bool = True
    show_tokens: bool = False


class OutputFormatter:
    """Format model output for display."""

    # ------------------------------------------------------------------
    # Main formatting entry point
    # ------------------------------------------------------------------

    def format(self, text: str, config: FormatterConfig) -> str:  # noqa: A003
        """Format *text* according to *config*."""
        handlers = {
            OutputFormat.PLAIN: self._fmt_plain,
            OutputFormat.RICH_MARKDOWN: self._fmt_rich_markdown,
            OutputFormat.JSON: self._fmt_json,
            OutputFormat.COMPACT: self._fmt_compact,
            OutputFormat.STREAM: self._fmt_stream,
        }
        handler = handlers.get(config.format, self._fmt_plain)
        return handler(text, config)

    # ------------------------------------------------------------------
    # Format-specific handlers
    # ------------------------------------------------------------------

    def _fmt_plain(self, text: str, config: FormatterConfig) -> str:  # noqa: ARG002
        return text

    def _fmt_rich_markdown(self, text: str, config: FormatterConfig) -> str:  # noqa: ARG002
        """Ensure inline code snippets are wrapped in fenced code blocks."""
        # If text already contains a fenced block, leave it alone.
        if re.search(r"^```", text, re.MULTILINE):
            return text

        # Wrap lines that look like code (indented 4+ spaces or contain
        # common code patterns) inside a generic code fence.
        lines = text.splitlines()
        result: list[str] = []
        in_code = False

        for line in lines:
            is_code_line = line.startswith("    ") or line.startswith("\t")
            if is_code_line and not in_code:
                result.append("```")
                in_code = True
            elif not is_code_line and in_code:
                result.append("```")
                in_code = False
            result.append(line)

        if in_code:
            result.append("```")

        return "\n".join(result)

    def _fmt_json(self, text: str, config: FormatterConfig) -> str:
        if config.show_tokens:
            payload: dict = {
                "response": text,
                "tokens": len(text.split()),
            }
        else:
            payload = {"response": text}
        return json.dumps(payload, ensure_ascii=False)

    def _fmt_compact(self, text: str, config: FormatterConfig) -> str:
        """Strip blank lines and hard-wrap at max_width."""
        lines = [ln for ln in text.splitlines() if ln.strip()]
        wrapped: list[str] = []
        for line in lines:
            while len(line) > config.max_width:
                wrapped.append(line[: config.max_width])
                line = line[config.max_width :]
            wrapped.append(line)
        return "\n".join(wrapped)

    def _fmt_stream(self, text: str, config: FormatterConfig) -> str:  # noqa: ARG002
        """Return text unchanged — streaming is handled by the caller."""
        return text

    # ------------------------------------------------------------------
    # Utility formatters
    # ------------------------------------------------------------------

    def format_error(self, error: Exception, config: FormatterConfig) -> str:
        """Format an exception for display."""
        msg = f"[ERROR] {type(error).__name__}: {error}"
        if config.format == OutputFormat.JSON:
            return json.dumps({"error": str(error), "type": type(error).__name__})
        if config.format == OutputFormat.COMPACT:
            return msg.strip()
        return msg

    def format_tokens_used(
        self, prompt_tokens: int, completion_tokens: int
    ) -> str:
        """Return a human-readable token-usage summary."""
        total = prompt_tokens + completion_tokens
        return (
            f"Tokens used — prompt: {prompt_tokens}, "
            f"completion: {completion_tokens}, "
            f"total: {total}"
        )
