"""Unified persona registry — single source of truth for all personas.

Replaces: PersonaRegistry, PersonaSwitcher, SecurityPersonaRegistry,
AgentModeRegistry. All persona lookups, registration, and routing
go through this one registry.
"""

from __future__ import annotations

from collections import defaultdict

from .unified_persona import PersonaDomain, UnifiedPersona


class PersonaNotFoundError(KeyError):
    pass


class PersonaAlreadyRegisteredError(KeyError):
    pass


class UnifiedPersonaRegistry:
    """Single registry that replaces all 5 existing persona registries.

    Merges: PersonaRegistry, PersonaSwitcher, SecurityPersonaRegistry,
            PersonalityRouter, AgentModeRegistry into one.
    """

    def __init__(self) -> None:
        self._personas: dict[str, UnifiedPersona] = {}
        self._domain_index: dict[PersonaDomain, list[str]] = defaultdict(list)
        self._current: str | None = None

    def register(self, persona: UnifiedPersona) -> None:
        if persona.id in self._personas:
            raise PersonaAlreadyRegisteredError(
                f"Persona {persona.id!r} is already registered"
            )
        self._personas[persona.id] = persona
        self._domain_index[persona.domain].append(persona.id)

    def unregister(self, persona_id: str) -> None:
        if persona_id not in self._personas:
            raise PersonaNotFoundError(f"Persona {persona_id!r} not found")
        persona = self._personas.pop(persona_id)
        self._domain_index[persona.domain].remove(persona_id)
        if self._current == persona_id:
            self._current = None

    def get(self, persona_id: str) -> UnifiedPersona:
        if persona_id not in self._personas:
            raise PersonaNotFoundError(
                f"Persona {persona_id!r} not found. "
                f"Known: {sorted(self._personas)}"
            )
        return self._personas[persona_id]

    def get_or_default(self, persona_id: str) -> UnifiedPersona:
        try:
            return self.get(persona_id)
        except PersonaNotFoundError:
            return self.default()

    def list_personas(self, domain: PersonaDomain | None = None) -> list[UnifiedPersona]:
        if domain is not None:
            ids = self._domain_index.get(domain, [])
            return [self._personas[i] for i in ids]
        return list(self._personas.values())

    def list_persona_ids(self, domain: PersonaDomain | None = None) -> list[str]:
        if domain is not None:
            return list(self._domain_index.get(domain, []))
        return sorted(self._personas.keys())

    def default(self) -> UnifiedPersona:
        if "aurelius-general" in self._personas:
            return self._personas["aurelius-general"]
        if self._personas:
            first_id = next(iter(self._personas))
            return self._personas[first_id]
        raise PersonaNotFoundError("No personas registered")

    def switch_to(self, persona_id: str) -> UnifiedPersona:
        persona = self.get(persona_id)
        self._current = persona.id
        return persona

    def current(self) -> UnifiedPersona | None:
        if self._current is None:
            return None
        return self._personas.get(self._current)

    def find_by_tool(self, tool_name: str) -> list[UnifiedPersona]:
        return [
            p
            for p in self._personas.values()
            if not p.allowed_tools or tool_name in p.allowed_tools
        ]

    def find_by_facet(self, facet_type: str) -> list[UnifiedPersona]:
        return [
            p for p in self._personas.values() if p.has_facet(facet_type)
        ]

    def find_by_domain(self, domain: PersonaDomain) -> list[UnifiedPersona]:
        ids = self._domain_index.get(domain, [])
        return [self._personas[i] for i in ids]

    def build_messages(
        self,
        persona_id: str,
        user_message: str,
        history: list[dict] | None = None,
    ) -> list[dict]:
        persona = self.get(persona_id)
        messages: list[dict] = [
            {"role": "system", "content": persona.truncated_system_prompt},
        ]
        if history:
            for turn in history:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages

    def validate_response(
        self,
        persona_id: str,
        response_obj: dict | str,
        query_type: str | None = None,
    ) -> tuple[bool, list[str]]:
        persona = self.get(persona_id)
        if not persona.output_contracts:
            return True, []

        if query_type is not None:
            contract = persona.get_output_contract(query_type)
            if contract is None:
                return False, [f"unknown output contract: {query_type!r}"]
            return self._validate_against_contract(response_obj, contract)

        contract = persona.output_contracts[0]
        return self._validate_against_contract(response_obj, contract)

    @staticmethod
    def _validate_against_contract(
        response_obj: dict | str,
        contract: "OutputContract",
    ) -> tuple[bool, list[str]]:
        from .unified_persona import OutputContract

        if isinstance(response_obj, str):
            return True, []

        errors: list[str] = []
        for field in contract.required_fields:
            if field not in response_obj:
                errors.append(f"missing required field: {field}")
        return (not errors), errors

    def __len__(self) -> int:
        return len(self._personas)

    def __contains__(self, persona_id: str) -> bool:
        return persona_id in self._personas

    def __repr__(self) -> str:
        domains = {d.value: len(ids) for d, ids in self._domain_index.items() if ids}
        return f"UnifiedPersonaRegistry({len(self._personas)} personas, domains={domains})"


__all__ = [
    "PersonaNotFoundError",
    "PersonaAlreadyRegisteredError",
    "UnifiedPersonaRegistry",
]