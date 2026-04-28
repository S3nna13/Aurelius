"""Tests for persona management system."""

from src.persona.persona_registry import Persona, PersonaRegistry
from src.persona.unified_persona import PersonaDomain, UnifiedPersona


def test_persona_creation():
    p = Persona(id="test", name="Tester", system_prompt="You are a test assistant.")
    assert p.id == "test"
    assert p.name == "Tester"


def test_registry_register():
    r = PersonaRegistry()
    p = Persona(id="coding", name="Coder", system_prompt="You are a coding assistant.")
    r.register(p)
    assert r.count == 1


def test_registry_get():
    r = PersonaRegistry()
    p = Persona(id="math", name="Math Tutor", system_prompt="You are a math tutor.")
    r.register(p)
    assert r.get("math") is p
    assert r.get("nonexistent") is None


def test_registry_list():
    r = PersonaRegistry()
    r.register(Persona(id="a", name="A", system_prompt=""))
    r.register(Persona(id="b", name="B", system_prompt=""))
    assert len(r.list()) == 2


def test_registry_delete():
    r = PersonaRegistry()
    r.register(Persona(id="x", name="X", system_prompt=""))
    assert r.delete("x") is True
    assert r.delete("x") is False
    assert r.count == 0


def test_default_personas():
    r = PersonaRegistry()
    r.register(
        Persona(
            id="general",
            name="General Assistant",
            system_prompt="You are a helpful assistant.",
        )
    )
    r.register(
        Persona(
            id="coding",
            name="Coding Assistant",
            system_prompt="You are an expert programmer.",
        )
    )
    r.register(
        Persona(
            id="security",
            name="Security Assistant",
            system_prompt="You are a security expert.",
        )
    )
    assert r.count == 3


def test_registry_accepts_unified_persona():
    r = PersonaRegistry()
    unified = UnifiedPersona(
        id="unified",
        name="Unified Assistant",
        domain=PersonaDomain.GENERAL,
        description="Unified compatibility persona",
        system_prompt="You are a unified assistant.",
    )

    r.register(unified)

    persona = r.get("unified")
    assert persona is not None
    assert persona.id == "unified"
    assert persona.name == "Unified Assistant"
