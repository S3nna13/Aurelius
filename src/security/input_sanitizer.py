"""Input sanitizer for LLM agent inputs — treat all inputs as hostile.

Implements Trail of Bits security principles: validate early, fail closed,
reject unexpected characters, and enforce strict type/range boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SanitizerRule:
    """A single validation rule for input sanitization."""

    field: str
    pattern: str | None = None
    max_length: int | None = None
    allowed_values: set[str] | None = None

    def validate(self, value: str) -> str | None:
        if self.max_length is not None and len(value) > self.max_length:
            return f"exceeds max length {self.max_length}"
        if self.pattern is not None and not re.match(self.pattern, value):
            return f"does not match pattern {self.pattern}"
        if self.allowed_values is not None and value not in self.allowed_values:
            return f"not in allowed values: {value}"
        return None


@dataclass
class InputSanitizer:
    """Strict input sanitizer — rejects, never coerces."""

    strip_control_chars: bool = True
    reject_sql_keywords: bool = True
    reject_html_tags: bool = True
    max_input_length: int = 100000
    _sql_keywords: set[str] | None = None
    _html_tag_re: re.Pattern | None = None

    def __post_init__(self) -> None:
        if self.reject_sql_keywords:
            self._sql_keywords = {
                "SELECT",
                "DROP",
                "DELETE",
                "INSERT",
                "UPDATE",
                "ALTER",
                "CREATE",
                "EXEC",
                "--",
                "/*",
                "*/",
                "UNION",
            }
        if self.reject_html_tags:
            self._html_tag_re = re.compile(r"<[^>]*>")

    def sanitize(self, text: str) -> str:
        if len(text) > self.max_input_length:
            raise ValueError(f"input exceeds {self.max_input_length} chars")
        if self.strip_control_chars:
            text = "".join(c for c in text if c.isprintable() or c in "\n\r\t")
        if self.reject_html_tags and self._html_tag_re:
            if self._html_tag_re.search(text):
                raise ValueError("HTML tags rejected")
        if self.reject_sql_keywords and self._sql_keywords:
            upper = text.upper()
            for kw in self._sql_keywords:
                if kw in upper:
                    raise ValueError(f"SQL keyword rejected: {kw}")
        return text


INPUT_SANITIZER = InputSanitizer()
