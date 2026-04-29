"""Persona registry — define and apply chat personas.

BACKWARD-COMPATIBLE WRAPPER: delegates to UnifiedPersonaRegistry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src.persona import (
    AURELIUS_ANALYST,
    AURELIUS_CODING,
    AURELIUS_CREATIVE,
    AURELIUS_GENERAL,
    AURELIUS_TEACHER,
    BUILTIN_PERSONAS,
    UnifiedPersonaRegistry,
)

_LEGACY_TO_UNIFIED: dict[str, str] = {
    "assistant": "aurelius-general",
    "coding": "aurelius-coding",
    "teacher": "aurelius-teacher",
    "analyst": "aurelius-analyst",
    "creative": "aurelius-creative",
}


class PersonaTone(StrEnum):
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    EMPATHETIC = "empathetic"
    CONCISE = "concise"


@dataclass
class PersonaConfig:
    persona_id: str
    name: str
    description: str
    tone: PersonaTone
    system_prompt: str
    temperature: float = 0.7


class PersonaRegistry:
    """Backward-compatible wrapper delegating to UnifiedPersonaRegistry."""

    DEFAULT_PERSONAS: dict[str, PersonaConfig] = {
        "assistant": PersonaConfig(
            persona_id="assistant",
            name="Assistant",
            description="Helpful general-purpose assistant",
            tone=PersonaTone.FORMAL,
            system_prompt=AURELIUS_GENERAL.system_prompt,
            temperature=AURELIUS_GENERAL.temperature,
        ),
        "coding": PersonaConfig(
            persona_id="coding",
            name="Coding Expert",
            description="Expert software engineer focused on correctness",
            tone=PersonaTone.TECHNICAL,
            system_prompt=AURELIUS_CODING.system_prompt,
            temperature=AURELIUS_CODING.temperature,
        ),
        "teacher": PersonaConfig(
            persona_id="teacher",
            name="Teacher",
            description="Patient educator who meets the learner where they are",
            tone=PersonaTone.EMPATHETIC,
            system_prompt=AURELIUS_TEACHER.system_prompt,
            temperature=AURELIUS_TEACHER.temperature,
        ),
        "analyst": PersonaConfig(
            persona_id="analyst",
            name="Analyst",
            description="Data and research analyst focused on evidence",
            tone=PersonaTone.FORMAL,
            system_prompt=AURELIUS_ANALYST.system_prompt,
            temperature=AURELIUS_ANALYST.temperature,
        ),
        "creative": PersonaConfig(
            persona_id="creative",
            name="Creative Writer",
            description="Creative writing helper with an expressive voice",
            tone=PersonaTone.CASUAL,
            system_prompt=AURELIUS_CREATIVE.system_prompt,
            temperature=AURELIUS_CREATIVE.temperature,
        ),
    }

    def __init__(self) -> None:
        self._personas: dict[str, PersonaConfig] = dict(self.DEFAULT_PERSONAS)
        self._unified = UnifiedPersonaRegistry()
        for p in BUILTIN_PERSONAS:
            if p.domain.value in ("general", "coding"):
                self._unified.register(p)

    def _to_legacy(self, unified: "UnifiedPersona") -> PersonaConfig:
        return PersonaConfig(
            persona_id=unified.id,
            name=unified.name,
            description=unified.description,
            tone=PersonaTone(unified.tone.value),
            system_prompt=unified.system_prompt,
            temperature=unified.temperature,
        )

    def register(self, config: PersonaConfig) -> None:
        self._personas[config.persona_id] = config

    def get(self, persona_id: str) -> PersonaConfig | None:
        if persona_id in self._personas:
            return self._personas[persona_id]
        unified_id = _LEGACY_TO_UNIFIED.get(persona_id, persona_id)
        try:
            unified = self._unified.get(unified_id)
        except Exception:
            return None
        return self._to_legacy(unified)

    def list_personas(self) -> list[PersonaConfig]:
        return [self._to_legacy(p) for p in self._unified.list_personas()]

    def apply_persona(
        self,
        messages: list[dict],
        persona_id: str,
    ) -> list[dict]:
        from src.persona import PromptComposer

        config = self.get(persona_id)
        if config is None:
            raise KeyError(f"Unknown persona: {persona_id!r}")

        # If resolved from unified registry, use PromptComposer for facet composition
        unified_id = _LEGACY_TO_UNIFIED.get(persona_id, persona_id)
        try:
            unified = self._unified.get(unified_id)
        except Exception:
            unified = None

        if unified is not None:
            composer = PromptComposer()
            system_prompt = composer.compose(unified)
        else:
            system_prompt = config.system_prompt

        system_msg: dict = {"role": "system", "content": system_prompt}
        if messages and messages[0].get("role") == "system":
            return [system_msg, *messages[1:]]
        return [system_msg, *messages]
