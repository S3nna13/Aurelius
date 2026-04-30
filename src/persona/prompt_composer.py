"""Prompt composer — assembles system prompts from persona + facets + priority encoding.

Replaces: PersonaRegistry.apply_persona(),
          SecurityPersonaRegistry.build_messages(),
          ThreatIntelPersona.build_messages(),
          SystemPromptPriorityEncoder.merge().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum

from .unified_persona import GuardrailSeverity, PersonaFacet, UnifiedPersona


class SystemPromptPriority(IntEnum):
    DEVELOPER = 0
    OPERATOR = 1
    USER = 2
    TOOL = 3
    MODEL = 4


@dataclass(frozen=True)
class SystemPromptFragment:
    priority: SystemPromptPriority
    content: str
    source_id: str
    immutable: bool = False


_ALWAYS_RE = re.compile(r"always\s+(\w+)", re.IGNORECASE)
_NEVER_RE = re.compile(r"never\s+(\w+)", re.IGNORECASE)


class PromptComposer:
    """Composes system prompts from persona + facets + priority encoding.

    Pipeline:
    1. Start with persona.system_prompt at persona.priority
    2. Activate facets — render additional prompts from facet configs
    3. Apply guardrails — prepend critical guardrails as high-priority fragments
    4. Add intent hints if persona has intent_mappings
    5. Merge with any external context_fragments using priority sorting
    6. Truncate if exceeds max_total_chars, respecting immutable fragments
    """

    def __init__(self, max_total_chars: int = 16_000, separator: str = "\n---\n") -> None:
        if max_total_chars <= 0:
            raise ValueError("max_total_chars must be positive")
        self.max_total_chars = max_total_chars
        self.separator = separator

    def compose(
        self,
        persona: UnifiedPersona,
        user_input: str | None = None,
        context_fragments: list[SystemPromptFragment] | None = None,
    ) -> str:
        fragments = self._build_fragments(persona, user_input)

        if context_fragments:
            fragments.extend(context_fragments)

        ordered = self._sort_fragments(fragments)
        return self._merge(ordered)

    def build_messages(
        self,
        persona: UnifiedPersona,
        user_message: str,
        history: list[dict] | None = None,
        context_fragments: list[SystemPromptFragment] | None = None,
    ) -> list[dict]:
        system_prompt = self.compose(
            persona, user_input=user_message, context_fragments=context_fragments
        )
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if history:
            for turn in history:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _build_fragments(
        self,
        persona: UnifiedPersona,
        user_input: str | None,
    ) -> list[SystemPromptFragment]:
        fragments: list[SystemPromptFragment] = []

        fragments.append(
            SystemPromptFragment(
                priority=SystemPromptPriority(persona.priority),
                content=persona.system_prompt,
                source_id=persona.id,
                immutable=persona.immutable_prompt,
            )
        )

        for facet in persona.facets:
            facet_prompt = self._render_facet_prompt(facet)
            if facet_prompt:
                facet_priority = self._facet_priority(facet)
                fragments.append(
                    SystemPromptFragment(
                        priority=facet_priority,
                        content=facet_prompt,
                        source_id=f"{persona.id}.{facet.facet_type}",
                        immutable=facet.facet_type in ("security", "constitution"),
                    )
                )

        for guardrail in persona.guardrails:
            if guardrail.severity in (GuardrailSeverity.CRITICAL, GuardrailSeverity.HIGH):
                fragments.append(
                    SystemPromptFragment(
                        priority=SystemPromptPriority.DEVELOPER,
                        content=f"MUST: {guardrail.text}",
                        source_id=f"guardrail.{guardrail.id}",
                        immutable=True,
                    )
                )
            else:
                fragments.append(
                    SystemPromptFragment(
                        priority=SystemPromptPriority.OPERATOR,
                        content=f"SHOULD: {guardrail.text}",
                        source_id=f"guardrail.{guardrail.id}",
                        immutable=False,
                    )
                )

        if user_input and persona.intent_mappings:
            intent_hint = self._resolve_intent_hint(persona, user_input)
            if intent_hint:
                fragments.append(
                    SystemPromptFragment(
                        priority=SystemPromptPriority.TOOL,
                        content=intent_hint,
                        source_id=f"intent.{persona.id}",
                        immutable=False,
                    )
                )

        return fragments

    def _render_facet_prompt(self, facet: PersonaFacet) -> str:
        ft = facet.facet_type
        cfg = facet.config

        if ft == "security":
            mode = cfg.get("mode", "defensive")
            scope = cfg.get("scope", "all")
            parts = [f"Security mode: {mode}"]
            if scope != "all":
                parts.append(f"Scope: {scope} assets only")
            return "; ".join(parts) + "."

        if ft == "threat_intel":
            classifiers = cfg.get("query_classifiers", [])
            if classifiers:
                return f"Classify queries as: {', '.join(classifiers)}. Respond with structured output per the relevant contract."
            return "Classify threat intelligence queries and respond with structured output."

        if ft == "agent_mode":
            mode = cfg.get("mode", "general")
            return f"Agent mode: {mode}."

        if ft == "constitution":
            dims = cfg.get("dimensions", [])
            if dims:
                return f"Constitutional dimensions: {', '.join(dims)}."
            return "Apply constitutional scoring to all responses."

        if ft == "harm_filter":
            categories = cfg.get("categories", "all")
            action = cfg.get("action", "block")
            if isinstance(categories, (list, tuple)):
                cats_str = ", ".join(categories)
            else:
                cats_str = str(categories)
            return f"Harm filter: {cats_str} — action: {action}."

        if ft == "personality":
            traits = cfg.get("traits", [])
            if traits:
                return f"Personality traits: {', '.join(traits)}."
            return ""

        if ft == "dialogue":
            return "Follow the dialogue state machine for multi-turn conversation management."

        return ""

    @staticmethod
    def _facet_priority(facet: PersonaFacet) -> SystemPromptPriority:
        priorities = {
            "security": SystemPromptPriority.OPERATOR,
            "threat_intel": SystemPromptPriority.OPERATOR,
            "constitution": SystemPromptPriority.OPERATOR,
            "harm_filter": SystemPromptPriority.DEVELOPER,
            "agent_mode": SystemPromptPriority.TOOL,
            "personality": SystemPromptPriority.MODEL,
            "dialogue": SystemPromptPriority.MODEL,
        }
        return priorities.get(facet.facet_type, SystemPromptPriority.MODEL)

    def _resolve_intent_hint(self, persona: UnifiedPersona, user_input: str) -> str | None:
        text = user_input.lower()

        for mapping in persona.intent_mappings:
            if mapping.intent in text:
                contract_name = mapping.output_contract_name
                if contract_name:
                    return (
                        f"[intent={mapping.intent}] "
                        f"Respond using the {contract_name} structured-output contract "
                        f"defined in the system prompt."
                    )
                return f"[intent={mapping.intent}] {mapping.behavior}"

        if persona.intent_mappings:
            default = persona.intent_mappings[0]
            contract_name = default.output_contract_name
            if contract_name:
                return (
                    "[intent=detect] "
                    "Classify the query and respond using the appropriate "
                    "structured-output contract."
                )

        return None

    @staticmethod
    def _sort_fragments(fragments: list[SystemPromptFragment]) -> list[SystemPromptFragment]:
        indexed = [(i, f) for i, f in enumerate(fragments) if f.content.strip()]
        indexed.sort(key=lambda pair: (int(pair[1].priority), pair[0]))
        return [f for _, f in indexed]

    def _merge(self, ordered: list[SystemPromptFragment]) -> str:
        if not ordered:
            return ""

        pieces = [
            f"[SOURCE:{f.source_id} PRIORITY:{f.priority.name}]\n{f.content}" for f in ordered
        ]
        total = sum(len(p) for p in pieces) + len(self.separator) * (len(pieces) - 1)

        if total <= self.max_total_chars:
            return self.separator.join(pieces)

        evict_order = sorted(
            range(len(ordered)),
            key=lambda i: (-int(ordered[i].priority), -i),
        )
        remaining = list(pieces)
        for idx in evict_order:
            if ordered[idx].immutable:
                continue
            remaining[idx] = ""
            total = sum(len(p) for p in remaining if p) + len(self.separator) * max(
                0, sum(1 for p in remaining if p) - 1
            )
            if total <= self.max_total_chars:
                break

        return self.separator.join(p for p in remaining if p)


__all__ = ["SystemPromptPriority", "SystemPromptFragment", "PromptComposer"]
