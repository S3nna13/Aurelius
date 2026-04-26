"""Aurelius SARIF v2.1.0 report generator.

Converts Aurelius internal security findings into SARIF v2.1.0 JSON so
downstream code-scanning tooling (GitHub code scanning, Azure DevOps,
etc.) can consume them.

The implementation targets a deliberately *minimal* but spec-compliant
subset of SARIF 2.1.0 -- enough for the ``runs[*].tool.driver.rules``
and ``runs[*].results`` shape that consumers require. 100% original
Aurelius code; no code imported from any external repository.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SarifLevel",
    "SarifRule",
    "SarifResult",
    "SarifReport",
    "SARIF_REPORTER_REGISTRY",
]


SARIF_SCHEMA_URI = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
)
SARIF_VERSION = "2.1.0"
AURELIUS_TOOL_NAME = "Aurelius"


class SarifLevel(enum.StrEnum):
    """SARIF result severity levels (subset used by Aurelius)."""

    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class SarifRule:
    """Describes a rule emitted by the Aurelius tool driver."""

    rule_id: str
    short_description: str
    full_description: str = ""
    help_text: str = ""
    tags: Sequence[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise ValueError("SarifRule.rule_id must be a non-empty string")
        if not isinstance(self.short_description, str):
            raise TypeError("SarifRule.short_description must be a string")
        # Normalize tags to a concrete list of strings.
        normalized: list[str] = []
        for tag in self.tags:
            if not isinstance(tag, str) or not tag:
                raise ValueError("SarifRule.tags entries must be non-empty strings")
            normalized.append(tag)
        self.tags = normalized

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.rule_id,
            "shortDescription": {"text": self.short_description},
        }
        if self.full_description:
            payload["fullDescription"] = {"text": self.full_description}
        if self.help_text:
            payload["help"] = {"text": self.help_text}
        if self.tags:
            payload["properties"] = {"tags": list(self.tags)}
        return payload


@dataclass
class SarifResult:
    """A single SARIF result (i.e. a finding for one rule at one location)."""

    rule_id: str
    message: str
    level: SarifLevel = SarifLevel.WARNING
    uri: str | None = None
    start_line: int | None = None
    end_line: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise ValueError("SarifResult.rule_id must be a non-empty string")
        if not isinstance(self.message, str):
            raise TypeError("SarifResult.message must be a string")
        if not isinstance(self.level, SarifLevel):
            # Allow plain strings ("error", "warning", "note") for ergonomics.
            try:
                self.level = SarifLevel(self.level)
            except ValueError as exc:
                raise ValueError(
                    f"SarifResult.level must be a SarifLevel; got {self.level!r}"
                ) from exc
        for name, value in (("start_line", self.start_line), ("end_line", self.end_line)):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"SarifResult.{name} must be a positive int or None")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ruleId": self.rule_id,
            "level": self.level.value,
            "message": {"text": self.message},
        }
        if self.uri is not None:
            physical: dict[str, Any] = {
                "artifactLocation": {"uri": self.uri},
            }
            region: dict[str, Any] = {}
            if self.start_line is not None:
                region["startLine"] = self.start_line
            if self.end_line is not None:
                region["endLine"] = self.end_line
            if region:
                physical["region"] = region
            payload["locations"] = [{"physicalLocation": physical}]
        return payload


class SarifReport:
    """Accumulates rules + results and serializes them to SARIF 2.1.0."""

    def __init__(self, tool_name: str = AURELIUS_TOOL_NAME) -> None:
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("tool_name must be a non-empty string")
        self._tool_name = tool_name
        self._rules: list[SarifRule] = []
        self._rule_ids: set[str] = set()
        self._results: list[SarifResult] = []

    # --- mutation ---------------------------------------------------------

    def add_rule(self, rule: SarifRule) -> None:
        if not isinstance(rule, SarifRule):
            raise TypeError("add_rule requires a SarifRule instance")
        if rule.rule_id in self._rule_ids:
            return  # idempotent
        self._rule_ids.add(rule.rule_id)
        self._rules.append(rule)

    def add_result(self, result: SarifResult) -> None:
        if not isinstance(result, SarifResult):
            raise TypeError("add_result requires a SarifResult instance")
        self._results.append(result)

    # --- accessors --------------------------------------------------------

    @property
    def rules(self) -> list[SarifRule]:
        return list(self._rules)

    @property
    def results(self) -> list[SarifResult]:
        return list(self._results)

    # --- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "$schema": SARIF_SCHEMA_URI,
            "version": SARIF_VERSION,
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": self._tool_name,
                            "rules": [rule.to_dict() for rule in self._rules],
                        }
                    },
                    "results": [result.to_dict() for result in self._results],
                }
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


# Registry for named reports (e.g. "default", "nightly", "cycle-150").
SARIF_REPORTER_REGISTRY: dict[str, SarifReport] = {}
SARIF_REPORTER_REGISTRY["default"] = SarifReport()
