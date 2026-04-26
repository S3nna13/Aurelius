"""Data masking utility for PII/secret redaction in logs and output.

Trail of Bits: mask sensitive data before it reaches output channels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_MASK_PATTERNS: list[tuple[str, str]] = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "***-**-****"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "***@***.***"),
    (r"\b\d{16}\b", "****-****-****-****"),
    (r"\b(?:\+1)?\d{10,11}\b", "***-***-****"),
    (r"\b(?:4[0-9]{3}|5[1-5][0-9]{2}|6011)\d{12,15}\b", "****-****-****-****"),
]


@dataclass
class DataMasker:
    """Mask PII and sensitive data in strings."""

    patterns: list[tuple[str, str]] = field(default_factory=lambda: _MASK_PATTERNS)

    def mask(self, text: str, replacement: str = "[REDACTED]") -> str:
        result = text
        for pattern, _ in self.patterns:
            result = re.sub(pattern, replacement, result)
        return result

    def mask_with_type(self, text: str) -> str:
        result = text
        for pattern, placeholder in self.patterns:
            result = re.sub(pattern, placeholder, result)
        return result

    def add_pattern(self, regex: str, placeholder: str) -> None:
        self.patterns.append((regex, placeholder))


DATA_MASKER = DataMasker()
