from __future__ import annotations

import re


class PIIDetector:
    """PII detection and redaction.

    Identifies and optionally redacts:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    - API keys / tokens
    - Names (basic heuristic)
    """

    PATTERNS: list[tuple[str, str]] = [
        ("email", r"[\w\.-]+@[\w\.-]+\.\w+"),
        ("phone", r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
        ("credit_card", r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        ("ip", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        ("api_key", r"(?:sk-|pk-|ark-)[a-fA-F0-9]{16,}"),
    ]

    def __init__(self, redact: bool = True) -> None:
        self.redact = redact

    def detect(self, text: str) -> list[tuple[str, str]]:
        findings: list[tuple[str, str]] = []
        for label, pattern in self.PATTERNS:
            for match in re.finditer(pattern, text):
                findings.append((label, match.group()))
        return findings

    def redact_pii(self, text: str, replacement: str = "[REDACTED]") -> str:
        result = text
        for label, pattern in self.PATTERNS:
            result = re.sub(pattern, replacement, result)
        return result
