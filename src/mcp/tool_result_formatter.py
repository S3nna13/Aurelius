"""Aurelius MCP — tool_result_formatter.py

Formats tool results into various string representations.
All logic is pure Python (stdlib only); no external frameworks required.
"""

from __future__ import annotations

import json
import traceback
from enum import Enum
from typing import Any

_TRUNCATION_LIMIT = 10_000  # characters

# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class ResultFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    CODE = "code"
    ERROR = "error"


# ---------------------------------------------------------------------------
# ToolResultFormatter
# ---------------------------------------------------------------------------

class ToolResultFormatter:
    """Convert arbitrary tool results to formatted strings."""

    # ------------------------------------------------------------------
    # Single-result formatting
    # ------------------------------------------------------------------

    def format(self, result: Any, fmt: ResultFormat = ResultFormat.TEXT) -> str:
        """Format *result* according to *fmt*."""
        if fmt == ResultFormat.TEXT:
            return self._format_text(result)
        if fmt == ResultFormat.JSON:
            return self._format_json(result)
        if fmt == ResultFormat.MARKDOWN:
            return self._format_markdown(result)
        if fmt == ResultFormat.CODE:
            return self._format_code(result)
        if fmt == ResultFormat.ERROR:
            return self._format_error(result)
        # Fallback
        return self._format_text(result)

    # ------------------------------------------------------------------
    # Batch formatting
    # ------------------------------------------------------------------

    def format_batch(self, results: list[Any], fmt: ResultFormat) -> list[str]:
        """Format each result in *results* with *fmt*."""
        return [self.format(r, fmt) for r in results]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(text: str, limit: int = _TRUNCATION_LIMIT) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"

    def _format_text(self, result: Any) -> str:
        return self._truncate(str(result))

    def _format_json(self, result: Any) -> str:
        try:
            serialised = json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError) as exc:
            serialised = json.dumps({"error": str(exc), "repr": repr(result)}, indent=2)
        return self._truncate(serialised)

    def _format_markdown(self, result: Any) -> str:
        if isinstance(result, (dict, list)):
            try:
                body = json.dumps(result, indent=2, default=str)
            except (TypeError, ValueError):
                body = repr(result)
            return self._truncate(f"```json\n{body}\n```")
        return self._truncate(str(result))

    @staticmethod
    def _format_code(result: Any, language: str = "") -> str:
        body = str(result)
        block = f"```{language}\n{body}\n```"
        if len(block) > _TRUNCATION_LIMIT:
            allowed = _TRUNCATION_LIMIT - len(f"```{language}\n\n```") - 30
            body = body[:allowed] + f"\n... [truncated]"
            block = f"```{language}\n{body}\n```"
        return block

    def _format_error(self, result: Any) -> str:
        if isinstance(result, BaseException):
            exc_type = type(result).__name__
            message = str(result)
            tb_lines = traceback.format_exception(type(result), result, result.__traceback__)
            tb_text = "".join(tb_lines)
            formatted = f"[{exc_type}] {message}\n\nTraceback:\n{tb_text}"
        else:
            formatted = f"[Error] {result}"
        return self._truncate(formatted)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MCP_REGISTRY: dict[str, Any] = {}
MCP_REGISTRY["tool_result_formatter"] = ToolResultFormatter()
