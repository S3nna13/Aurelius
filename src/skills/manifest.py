"""Aurelius v2 Skill Manifest — typed schema for built-in skill definitions."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class RiskLevel(enum.StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SkillStatus(enum.StrEnum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"
    UNSAFE = "unsafe"


class SkillExecutionMode(enum.StrEnum):
    DRY_RUN = "dry_run"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    ROLLBACK = "rollback"


@dataclass
class SkillPermission:
    """A single permission declaration for a skill."""
    name: str  # file_read, file_write, terminal, network, browser, cua, etc.
    scope: str = "limited"  # "full", "limited", "none"


@dataclass
class SkillManifest:
    """Typed manifest for an Aurelius native skill.

    Every built-in skill MUST have a valid manifest.
    External skills are validated against this schema before loading.
    """
    id: str  # e.g. "coding.python_test_repair"
    name: str  # e.g. "Python Test Repair"
    version: str  # e.g. "1.0.0"
    category: str  # e.g. "coding"
    summary: str  # One-line description
    description: str = ""  # Full description
    permissions: list[SkillPermission] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    required_tools: list[str] = field(default_factory=list)
    entrypoint: str = ""  # "skills.builtin.coding.python_test_repair:run"
    inputs_schema: str = ""  # Path to JSON schema for inputs
    outputs_schema: str = ""  # Path to JSON schema for outputs
    daies_tests: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    status: SkillStatus = SkillStatus.STABLE
    supported_modes: list[SkillExecutionMode] = field(default_factory=lambda: [
        SkillExecutionMode.DRY_RUN,
        SkillExecutionMode.EXECUTE,
        SkillExecutionMode.VERIFY,
    ])
    max_runtime_seconds: int = 300
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Validate manifest and return list of errors (empty = valid)."""
        errors: list[str] = []
        if not self.id or "." not in self.id:
            errors.append("Skill id must be in 'category.name' format")
        if not self.name:
            errors.append("Skill name is required")
        if not self.version:
            errors.append("Skill version is required")
        if not self.category:
            errors.append("Skill category is required")
        if not self.summary:
            errors.append("Skill summary is required")
        if not self.entrypoint and self.status == SkillStatus.STABLE:
            errors.append("Stable skills must have an entrypoint")
        if self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) and                 SkillExecutionMode.DRY_RUN not in self.supported_modes:
            errors.append("High-risk skills must support dry_run mode")
        if required_tools := self.required_tools:
            # Basic tool validation
            known_tools = {
                "file.search", "file.read", "file.write", "file.patch",
                "terminal.run", "web.search", "web.extract",
                "browser.click", "browser.type", "browser.navigate",
                "memory.search", "memory.write",
                "cron.create", "cron.list",
            }
            for t in required_tools:
                if t not in known_tools and not t.startswith("custom:"):
                    errors.append(f"Unknown tool: {t}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to JSON-serialisable dict."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "category": self.category,
            "summary": self.summary,
            "description": self.description,
            "permissions": [{"name": p.name, "scope": p.scope} for p in self.permissions],
            "risk_level": self.risk_level.value,
            "required_tools": self.required_tools,
            "entrypoint": self.entrypoint,
            "inputs_schema": self.inputs_schema,
            "outputs_schema": self.outputs_schema,
            "daies_tests": self.daies_tests,
            "tags": self.tags,
            "status": self.status.value,
            "supported_modes": [m.value for m in self.supported_modes],
            "max_runtime_seconds": self.max_runtime_seconds,
            "metadata": self.metadata,
        }
