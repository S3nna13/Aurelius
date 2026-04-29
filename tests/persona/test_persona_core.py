"""Tests for persona management system — adding coverage."""

import pytest
from src.persona.builtins import BUILTIN_PERSONAS
from src.persona.persona_registry import UnifiedPersonaRegistry
from src.persona.unified_persona import UnifiedPersona


def test_builtin_personas_exist():
    assert len(BUILTIN_PERSONAS) > 0


def test_builtin_persona_creation():
    for persona in BUILTIN_PERSONAS:
        assert isinstance(persona, UnifiedPersona)
        assert persona.name
        assert persona.system_prompt


def test_unified_registry():
    reg = UnifiedPersonaRegistry()
    for p in BUILTIN_PERSONAS:
        reg.register(p)
    assert len(reg.list_persona_ids()) >= len(BUILTIN_PERSONAS)


def test_persona_lookup():
    reg = UnifiedPersonaRegistry()
    ids = []
    for p in BUILTIN_PERSONAS:
        reg.register(p)
        ids.append(p.id)
    if ids:
        found = reg.get(ids[0])
        assert found is not None


def test_persona_id_lookup():
    reg = UnifiedPersonaRegistry()
    for p in BUILTIN_PERSONAS:
        reg.register(p)
    ids = reg.list_persona_ids()
    assert len(ids) >= len(BUILTIN_PERSONAS)


def test_persona_not_found():
    reg = UnifiedPersonaRegistry()
    with pytest.raises(KeyError):
        reg.get("nonexistent_persona")
