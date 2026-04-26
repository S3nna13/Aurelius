"""Unified persona data model for Aurelius.

A single UnifiedPersona dataclass replaces all prior persona systems:
PersonaConfig, PersonaDefinition, SecurityPersona, AgentMode, PersonalityType.

Facets are composable capability attachments that let any persona gain
features from any other system (security guardrails, threat intel
classification, constitution scoring, harm filtering, tool gating).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PersonaDomain(StrEnum):
    GENERAL = "general"
    SECURITY = "security"
    THREAT_INTEL = "threat_intel"
    AGENT = "agent"
    CODING = "coding"


class PersonaTone(StrEnum):
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    EMPATHETIC = "empathetic"
    CONCISE = "concise"


class ResponseStyle(StrEnum):
    CONCISE = "concise"
    STRUCTURED = "structured"
    VERBOSE = "verbose"
    METHODICAL = "methodical"
    ADAPTIVE = "adaptive"


class GuardrailSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardrailScope(StrEnum):
    ALL = "all"
    OFFENSIVE = "offensive"
    DEFENSIVE = "defensive"
    GENERAL = "general"


@dataclass(frozen=True)
class WorkflowStage:
    name: str
    description: str = ""


@dataclass(frozen=True)
class OutputContract:
    name: str
    schema: dict[str, Any] = field(default_factory=dict)
    required_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class Guardrail:
    id: str
    text: str
    severity: GuardrailSeverity = GuardrailSeverity.HIGH
    scope: GuardrailScope = GuardrailScope.ALL


@dataclass(frozen=True)
class IntentMapping:
    intent: str
    behavior: str
    output_contract_name: str | None = None


@dataclass(frozen=True)
class PersonaFacet:
    facet_type: str
    config: dict[str, Any] = field(default_factory=dict)


_PERSONA_NAME_RE = __import__("re").compile(r"^[A-Za-z0-9_-]{1,128}$")
_MAX_SYSTEM_PROMPT_LENGTH = 32_768


@dataclass
class UnifiedPersona:
    """Single persona definition that subsumes all prior persona systems.

    Replaces: PersonaConfig, PersonaDefinition, SecurityPersona,
    AgentMode, PersonalityType.
    """

    id: str
    name: str
    domain: PersonaDomain
    description: str
    system_prompt: str
    tone: PersonaTone = PersonaTone.FORMAL
    response_style: ResponseStyle = ResponseStyle.CONCISE
    temperature: float = 0.7
    workflow_stages: tuple[WorkflowStage, ...] = ()
    output_contracts: tuple[OutputContract, ...] = ()
    guardrails: tuple[Guardrail, ...] = ()
    intent_mappings: tuple[IntentMapping, ...] = ()
    facets: tuple[PersonaFacet, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    priority: int = 4
    immutable_prompt: bool = False

    def __post_init__(self) -> None:
        if not _PERSONA_NAME_RE.match(self.id):
            raise ValueError(
                f"Persona id must be 1-128 chars, alphanumeric/underscore/hyphen: {self.id!r}"
            )
        if not self.name.strip():
            raise ValueError(f"Persona name must be non-empty: {self.name!r}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be in [0.0, 2.0], got {self.temperature}")
        if not 0 <= self.priority <= 4:
            raise ValueError(f"Priority must be 0-4, got {self.priority}")

    def with_facets(self, *extra_facets: PersonaFacet) -> UnifiedPersona:
        """Return a copy of this persona with additional facets merged in."""
        existing_types = {f.facet_type for f in self.facets}
        merged = list(self.facets)
        for f in extra_facets:
            if f.facet_type in existing_types:
                idx = next(i for i, ef in enumerate(merged) if ef.facet_type == f.facet_type)
                merged[idx] = f
            else:
                merged.append(f)
        from dataclasses import replace

        return replace(self, facets=tuple(merged))

    def has_facet(self, facet_type: str) -> bool:
        return any(f.facet_type == facet_type for f in self.facets)

    def get_facet(self, facet_type: str) -> PersonaFacet | None:
        for f in self.facets:
            if f.facet_type == facet_type:
                return f
        return None

    def get_output_contract(self, name: str) -> OutputContract | None:
        for c in self.output_contracts:
            if c.name == name:
                return c
        return None

    def has_guardrail(self, guardrail_id: str) -> bool:
        return any(g.id == guardrail_id for g in self.guardrails)

    @property
    def truncated_system_prompt(self) -> str:
        if len(self.system_prompt) <= _MAX_SYSTEM_PROMPT_LENGTH:
            return self.system_prompt
        return self.system_prompt[:_MAX_SYSTEM_PROMPT_LENGTH]


__all__ = [
    "PersonaDomain",
    "PersonaTone",
    "ResponseStyle",
    "GuardrailSeverity",
    "GuardrailScope",
    "WorkflowStage",
    "OutputContract",
    "Guardrail",
    "IntentMapping",
    "PersonaFacet",
    "UnifiedPersona",
]