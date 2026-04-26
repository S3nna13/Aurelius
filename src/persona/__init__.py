"""Aurelius Unified Persona System.

A single UnifiedPersona data model with composable PersonaFacets replaces
all prior persona systems: PersonaRegistry, PersonaSwitcher, SecurityPersonaRegistry,
PersonalityRouter, AgentModeRegistry.

Usage:
    from src.persona import UnifiedPersona, UnifiedPersonaRegistry, PersonaRouter
    from src.persona import builtins

    registry = UnifiedPersonaRegistry()
    for persona in builtins.ALL_BUILTINS:
        registry.register(persona)

    router = PersonaRouter(registry)
    persona = router.route("What is CVE-2024-3094?")
    # -> AURELIUS_THREATEL

    composer = PromptComposer()
    messages = composer.build_messages(persona, "Explain CVE-2024-3094")
"""

from .unified_persona import (
    Guardrail,
    GuardrailScope,
    GuardrailSeverity,
    IntentMapping,
    OutputContract,
    PersonaDomain,
    PersonaFacet,
    PersonaTone,
    ResponseStyle,
    UnifiedPersona,
    WorkflowStage,
)
from .persona_registry import (
    PersonaAlreadyRegisteredError,
    PersonaNotFoundError,
    UnifiedPersonaRegistry,
)
from .persona_router import PersonaRouter, RoutingResult
from .prompt_composer import PromptComposer, SystemPromptFragment, SystemPromptPriority

__all__ = [
    "Guardrail",
    "GuardrailScope",
    "GuardrailSeverity",
    "IntentMapping",
    "OutputContract",
    "PersonaDomain",
    "PersonaFacet",
    "PersonaTone",
    "ResponseStyle",
    "UnifiedPersona",
    "WorkflowStage",
    "UnifiedPersonaRegistry",
    "PersonaNotFoundError",
    "PersonaAlreadyRegisteredError",
    "PersonaRouter",
    "RoutingResult",
    "PromptComposer",
    "SystemPromptFragment",
    "SystemPromptPriority",
]