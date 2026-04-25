"""Tests for src/chat/persona_registry.py."""

import pytest

from src.chat.persona_registry import (
    PersonaConfig,
    PersonaRegistry,
    PersonaTone,
)


# ---------------------------------------------------------------------------
# Enum smoke tests
# ---------------------------------------------------------------------------

def test_persona_tone_values():
    assert PersonaTone.FORMAL == "formal"
    assert PersonaTone.TECHNICAL == "technical"
    assert len(PersonaTone) == 5


# ---------------------------------------------------------------------------
# Default personas
# ---------------------------------------------------------------------------

def test_default_personas_count():
    registry = PersonaRegistry()
    assert len(registry.list_personas()) == 5


def test_default_assistant_persona():
    registry = PersonaRegistry()
    persona = registry.get("assistant")
    assert persona is not None
    assert persona.tone == PersonaTone.FORMAL
    assert persona.temperature == 0.7


def test_default_coding_persona():
    registry = PersonaRegistry()
    persona = registry.get("coding")
    assert persona is not None
    assert persona.tone == PersonaTone.TECHNICAL
    assert persona.temperature == 0.3


def test_default_creative_persona():
    registry = PersonaRegistry()
    persona = registry.get("creative")
    assert persona is not None
    assert persona.tone == PersonaTone.CASUAL
    assert persona.temperature == 1.0


def test_default_analyst_persona():
    registry = PersonaRegistry()
    persona = registry.get("analyst")
    assert persona is not None
    assert persona.temperature == 0.2


def test_default_teacher_persona():
    registry = PersonaRegistry()
    persona = registry.get("teacher")
    assert persona is not None
    assert persona.tone == PersonaTone.EMPATHETIC


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------

def test_register_new_persona():
    registry = PersonaRegistry()
    config = PersonaConfig(
        persona_id="custom",
        name="Custom",
        description="A custom persona",
        tone=PersonaTone.CONCISE,
        system_prompt="Be brief.",
        temperature=0.5,
    )
    registry.register(config)
    assert registry.get("custom") is config


def test_register_overwrites_existing():
    registry = PersonaRegistry()
    new_cfg = PersonaConfig(
        persona_id="assistant",
        name="New Assistant",
        description="Override",
        tone=PersonaTone.CASUAL,
        system_prompt="Casual override.",
        temperature=0.9,
    )
    registry.register(new_cfg)
    assert registry.get("assistant").name == "New Assistant"


def test_get_unknown_persona_returns_none():
    registry = PersonaRegistry()
    assert registry.get("nonexistent") is None


# ---------------------------------------------------------------------------
# list_personas
# ---------------------------------------------------------------------------

def test_list_personas_returns_list():
    registry = PersonaRegistry()
    personas = registry.list_personas()
    assert isinstance(personas, list)
    ids = {p.persona_id for p in personas}
    assert "assistant" in ids
    assert "coding" in ids


# ---------------------------------------------------------------------------
# apply_persona
# ---------------------------------------------------------------------------

def test_apply_persona_prepends_system_message():
    registry = PersonaRegistry()
    messages = [{"role": "user", "content": "Hello"}]
    result = registry.apply_persona(messages, "assistant")
    assert result[0]["role"] == "system"
    assert "assistant" in result[0]["content"].lower() or len(result[0]["content"]) > 0
    assert result[1] == messages[0]


def test_apply_persona_replaces_existing_system_message():
    registry = PersonaRegistry()
    messages = [
        {"role": "system", "content": "Old system prompt"},
        {"role": "user", "content": "Hi"},
    ]
    result = registry.apply_persona(messages, "coding")
    assert result[0]["role"] == "system"
    assert result[0]["content"] != "Old system prompt"
    assert len(result) == 2


def test_apply_persona_unknown_raises_key_error():
    registry = PersonaRegistry()
    with pytest.raises(KeyError):
        registry.apply_persona([], "ghost")


def test_apply_persona_does_not_mutate_original():
    registry = PersonaRegistry()
    original = [{"role": "user", "content": "test"}]
    registry.apply_persona(original, "teacher")
    assert len(original) == 1


def test_instance_mutation_does_not_affect_class_defaults():
    registry = PersonaRegistry()
    registry.register(
        PersonaConfig(
            persona_id="assistant",
            name="Modified",
            description="Modified",
            tone=PersonaTone.CASUAL,
            system_prompt="Modified.",
        )
    )
    # Class-level default should be untouched
    assert PersonaRegistry.DEFAULT_PERSONAS["assistant"].name == "Assistant"
