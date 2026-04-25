"""Persona switcher — manage and switch between chat personas / system prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Valid persona names: alphanumeric, underscore, hyphen; 1–64 characters.
_PERSONA_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_PROMPT_LENGTH = 4096


@dataclass
class PersonaDefinition:
    name: str
    system_prompt: str
    metadata: dict = field(default_factory=dict)


class PersonaSwitcher:
    """Manage multiple chat personas and switch between them."""

    def __init__(self) -> None:
        self._personas: dict[str, PersonaDefinition] = {}
        self._current: str = "default"
        self.register_persona("default", "")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Validate and return a persona name.

        Raises:
            ValueError: If the name contains invalid characters or exceeds 64 chars.
        """
        if not isinstance(name, str):
            raise ValueError(f"Persona name must be a string, got {type(name).__name__}")
        if not _PERSONA_NAME_RE.match(name):
            raise ValueError(
                f"Invalid persona name {name!r}: must be 1–64 chars, alphanumeric/underscore/hyphen only."
            )
        return name

    @staticmethod
    def _truncate_prompt(prompt: str) -> str:
        """Truncate a system prompt to the maximum allowed length."""
        if not isinstance(prompt, str):
            raise ValueError(f"System prompt must be a string, got {type(prompt).__name__}")
        if len(prompt) > _MAX_PROMPT_LENGTH:
            return prompt[:_MAX_PROMPT_LENGTH]
        return prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_persona(
        self,
        name: str,
        system_prompt: str,
        metadata: dict | None = None,
    ) -> None:
        """Register (or overwrite) a persona."""
        sanitized = self._sanitize_name(name)
        truncated = self._truncate_prompt(system_prompt)
        self._personas[sanitized] = PersonaDefinition(
            name=sanitized,
            system_prompt=truncated,
            metadata=dict(metadata) if metadata is not None else {},
        )

    def switch_to(self, name: str) -> str:
        """Switch to *name* and return its system prompt.

        Raises:
            KeyError: If the persona is not registered.
        """
        sanitized = self._sanitize_name(name)
        if sanitized not in self._personas:
            raise KeyError(f"Persona {name!r} is not registered.")
        self._current = sanitized
        return self._personas[sanitized].system_prompt

    def current(self) -> str:
        """Return the name of the currently active persona."""
        return self._current

    def list_personas(self) -> list[str]:
        """Return a list of all registered persona names."""
        return list(self._personas.keys())

    def get_prompt(self, name: str) -> str:
        """Return the system prompt for *name*.

        Raises:
            KeyError: If the persona is not registered.
        """
        sanitized = self._sanitize_name(name)
        if sanitized not in self._personas:
            raise KeyError(f"Persona {name!r} is not registered.")
        return self._personas[sanitized].system_prompt

    def remove_persona(self, name: str) -> None:
        """Remove a persona.

        Raises:
            KeyError: If the persona is not registered.
            ValueError: If attempting to remove the default persona.
        """
        sanitized = self._sanitize_name(name)
        if sanitized == "default":
            raise ValueError("Cannot remove the default persona.")
        if sanitized not in self._personas:
            raise KeyError(f"Persona {name!r} is not registered.")
        del self._personas[sanitized]
        if self._current == sanitized:
            self._current = "default"
