"""Output sanitizer: PII scrubbing, URL filtering, secret pattern removal."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class SanitizationRule:
    name: str
    pattern: str
    replacement: str
    enabled: bool = True


@dataclass
class SanitizationResult:
    original_length: int
    sanitized_text: str
    rules_applied: list[str]
    redaction_count: int


class OutputSanitizer:
    DEFAULT_RULES: ClassVar[list[SanitizationRule]] = [
        SanitizationRule(
            name="email",
            pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            replacement="[EMAIL]",
        ),
        SanitizationRule(
            name="phone_us",
            pattern=r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
            replacement="[PHONE]",
        ),
        SanitizationRule(
            name="ssn",
            pattern=r"\b\d{3}-\d{2}-\d{4}\b",
            replacement="[SSN]",
        ),
        SanitizationRule(
            name="api_key",
            pattern=r"\b(sk|pk|api|key)[-_][A-Za-z0-9]{20,}\b",
            replacement="[API_KEY]",
        ),
        SanitizationRule(
            name="ipv4",
            pattern=r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            replacement="[IP]",
        ),
    ]

    def __init__(self, rules: list[SanitizationRule] | None = None) -> None:
        if rules is None:
            # Copy defaults so mutations don't affect class-level list
            self._rules: list[SanitizationRule] = [
                SanitizationRule(
                    name=r.name,
                    pattern=r.pattern,
                    replacement=r.replacement,
                    enabled=r.enabled,
                )
                for r in self.DEFAULT_RULES
            ]
        else:
            self._rules = list(rules)

    def sanitize(self, text: str) -> SanitizationResult:
        original_length = len(text)
        sanitized = text
        rules_applied: list[str] = []
        redaction_count = 0

        for rule in self._rules:
            if not rule.enabled:
                continue
            new_text, n = re.subn(rule.pattern, rule.replacement, sanitized)
            if n > 0:
                rules_applied.append(rule.name)
                redaction_count += n
                sanitized = new_text

        return SanitizationResult(
            original_length=original_length,
            sanitized_text=sanitized,
            rules_applied=rules_applied,
            redaction_count=redaction_count,
        )

    def add_rule(self, rule: SanitizationRule) -> None:
        self._rules.append(rule)

    def disable_rule(self, name: str) -> bool:
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = False
                return True
        return False
