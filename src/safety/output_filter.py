"""Output filter — Aurelius safety module.

Filters model outputs for safety policy violations using configurable rules.
Stdlib-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FilterAction(Enum):
    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"
    WARN = "warn"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterRule:
    """A single output filter rule."""
    name: str
    pattern: str
    action: FilterAction
    replacement: str = "[FILTERED]"


@dataclass(frozen=True)
class FilterResult:
    """Result of applying an OutputFilter to a text string."""
    original: str
    filtered: str
    action: FilterAction
    triggered_rules: List[str]


# ---------------------------------------------------------------------------
# Default rules
# ---------------------------------------------------------------------------

_DEFAULT_RULES: List[FilterRule] = [
    FilterRule(
        name="pii_email",
        pattern=r'\b[\w.-]+@[\w.-]+\.\w{2,}\b',
        action=FilterAction.REDACT,
        replacement="[EMAIL]",
    ),
    FilterRule(
        name="pii_phone",
        pattern=r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        action=FilterAction.REDACT,
        replacement="[PHONE]",
    ),
    FilterRule(
        name="pii_ssn",
        pattern=r'\b\d{3}-\d{2}-\d{4}\b',
        action=FilterAction.REDACT,
        replacement="[SSN]",
    ),
]


# ---------------------------------------------------------------------------
# OutputFilter
# ---------------------------------------------------------------------------

class OutputFilter:
    """Applies ordered filter rules to model output text."""

    def __init__(self, rules: Optional[List[FilterRule]] = None) -> None:
        if rules is None:
            self._rules: List[FilterRule] = list(_DEFAULT_RULES)
        else:
            self._rules = list(rules)
        # Cache compiled patterns; invalidated on add/remove
        self._compiled: dict[str, re.Pattern] = {}
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        self._compiled = {
            rule.name: re.compile(rule.pattern) for rule in self._rules
        }

    def filter(self, text: str) -> FilterResult:
        """Apply all rules in order and return a FilterResult.

        REDACT:  substitute matched text with replacement string.
        BLOCK:   immediately return "[BLOCKED]" with the triggering rule name.
        WARN:    add to triggered_rules but do not alter text.
        ALLOW:   pass through without modification.
        """
        current = text
        triggered: List[str] = []
        highest_action = FilterAction.ALLOW

        for rule in self._rules:
            rx = self._compiled.get(rule.name)
            if rx is None:
                continue
            m = rx.search(current)
            if m is None:
                continue

            triggered.append(rule.name)

            if rule.action == FilterAction.BLOCK:
                return FilterResult(
                    original=text,
                    filtered="[BLOCKED]",
                    action=FilterAction.BLOCK,
                    triggered_rules=triggered,
                )
            elif rule.action == FilterAction.REDACT:
                current = rx.sub(rule.replacement, current)
                if highest_action == FilterAction.ALLOW:
                    highest_action = FilterAction.REDACT
            elif rule.action == FilterAction.WARN:
                if highest_action == FilterAction.ALLOW:
                    highest_action = FilterAction.WARN
            # ALLOW: no change

        final_action = highest_action if triggered else FilterAction.ALLOW
        return FilterResult(
            original=text,
            filtered=current,
            action=final_action,
            triggered_rules=triggered,
        )

    def add_rule(self, rule: FilterRule) -> None:
        """Append a new rule to the filter."""
        self._rules.append(rule)
        self._compiled[rule.name] = re.compile(rule.pattern)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name. Returns True if found and removed."""
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                del self._rules[i]
                self._compiled.pop(name, None)
                return True
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OUTPUT_FILTER_REGISTRY: dict[str, type] = {
    "default": OutputFilter,
}
