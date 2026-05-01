from __future__ import annotations

from .persona_registry import (
    Persona,
    PersonaAlreadyRegisteredError,
    PersonaNotFoundError,
    PersonaRegistry,
    UnifiedPersonaRegistry,
)
from .persona_router import PersonaRouter
from .prompt_composer import PromptComposer
from .unified_persona import (
    Guardrail,
    IntentMapping,
    OutputContract,
    PersonaDomain,
    PersonaFacet,
    PersonaTone,
    ResponseStyle,
    UnifiedPersona,
    WorkflowStage,
)

__all__ = [
    "UnifiedPersona",
    "Persona",
    "PersonaFacet",
    "PersonaDomain",
    "PersonaTone",
    "ResponseStyle",
    "WorkflowStage",
    "OutputContract",
    "Guardrail",
    "IntentMapping",
    "PersonaRegistry",
    "PersonaNotFoundError",
    "PersonaAlreadyRegisteredError",
    "UnifiedPersonaRegistry",
    "PersonaRouter",
    "PromptComposer",
    "AURELIUS_GENERAL",
    "AURELIUS_CODING",
    "AURELIUS_TEACHER",
    "AURELIUS_ANALYST",
    "AURELIUS_CREATIVE",
    "AURELIUS_REDTEAM",
    "AURELIUS_BLUETEAM",
    "AURELIUS_PURPLETEAM",
    "AURELIUS_THREATINTEL",
    "AURELIUS_CODE_MODE",
    "AURELIUS_ARCHITECT_MODE",
    "AURELIUS_ASK_MODE",
    "AURELIUS_DEBUG_MODE",
    "BUILTIN_PERSONAS",
]

_BUILTIN_EXPORTS = {
    "AURELIUS_GENERAL",
    "AURELIUS_CODING",
    "AURELIUS_TEACHER",
    "AURELIUS_ANALYST",
    "AURELIUS_CREATIVE",
    "AURELIUS_REDTEAM",
    "AURELIUS_BLUETEAM",
    "AURELIUS_PURPLETEAM",
    "AURELIUS_THREATINTEL",
    "AURELIUS_CODE_MODE",
    "AURELIUS_ARCHITECT_MODE",
    "AURELIUS_ASK_MODE",
    "AURELIUS_DEBUG_MODE",
    "BUILTIN_PERSONAS",
}


def __getattr__(name: str):
    if name in _BUILTIN_EXPORTS:
        from . import builtins as _builtins

        value = getattr(_builtins, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
