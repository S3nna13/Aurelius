"""Persona registry — define and apply chat personas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PersonaTone(str, Enum):
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
    """Registry of named chat personas."""

    DEFAULT_PERSONAS: dict[str, PersonaConfig] = {
        "assistant": PersonaConfig(
            persona_id="assistant",
            name="Assistant",
            description="Helpful general-purpose assistant",
            tone=PersonaTone.FORMAL,
            system_prompt=(
                "You are a helpful, accurate, and professional assistant. "
                "Provide clear, well-structured answers."
            ),
            temperature=0.7,
        ),
        "coding": PersonaConfig(
            persona_id="coding",
            name="Coding Expert",
            description="Expert software engineer focused on correctness",
            tone=PersonaTone.TECHNICAL,
            system_prompt=(
                "You are an expert software engineer. Produce correct, efficient, "
                "and well-documented code. Explain your reasoning concisely."
            ),
            temperature=0.3,
        ),
        "teacher": PersonaConfig(
            persona_id="teacher",
            name="Teacher",
            description="Patient educator who meets the learner where they are",
            tone=PersonaTone.EMPATHETIC,
            system_prompt=(
                "You are a patient, encouraging teacher. Break concepts down into "
                "accessible steps and check understanding along the way."
            ),
            temperature=0.8,
        ),
        "analyst": PersonaConfig(
            persona_id="analyst",
            name="Analyst",
            description="Data and research analyst focused on evidence",
            tone=PersonaTone.FORMAL,
            system_prompt=(
                "You are a rigorous data and research analyst. Cite evidence, "
                "quantify uncertainty, and present findings objectively."
            ),
            temperature=0.2,
        ),
        "creative": PersonaConfig(
            persona_id="creative",
            name="Creative Writer",
            description="Creative writing helper with an expressive voice",
            tone=PersonaTone.CASUAL,
            system_prompt=(
                "You are an imaginative creative writing companion. Embrace "
                "vivid language, unexpected angles, and playful experimentation."
            ),
            temperature=1.0,
        ),
    }

    def __init__(self) -> None:
        # Shallow copy so mutations don't affect the class-level dict
        self._personas: dict[str, PersonaConfig] = dict(self.DEFAULT_PERSONAS)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, config: PersonaConfig) -> None:
        """Add or overwrite a persona."""
        self._personas[config.persona_id] = config

    def get(self, persona_id: str) -> Optional[PersonaConfig]:
        """Return the PersonaConfig for *persona_id*, or None if not found."""
        return self._personas.get(persona_id)

    def list_personas(self) -> list[PersonaConfig]:
        """Return all registered personas."""
        return list(self._personas.values())

    # ------------------------------------------------------------------
    # Conversation integration
    # ------------------------------------------------------------------

    def apply_persona(
        self,
        messages: list[dict],
        persona_id: str,
    ) -> list[dict]:
        """Prepend the persona's system prompt to *messages*.

        If a system message already exists at position 0, it is replaced.
        Returns a new list (the original is not mutated).
        """
        config = self.get(persona_id)
        if config is None:
            raise KeyError(f"Unknown persona: {persona_id!r}")

        system_msg: dict = {"role": "system", "content": config.system_prompt}

        if messages and messages[0].get("role") == "system":
            return [system_msg, *messages[1:]]

        return [system_msg, *messages]
