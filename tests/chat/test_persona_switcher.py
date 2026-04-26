"""Tests for src/chat/persona_switcher.py."""

from __future__ import annotations

import pytest

from src.chat.persona_switcher import PersonaDefinition, PersonaSwitcher


# ---------------------------------------------------------------------------
# PersonaDefinition dataclass
# ---------------------------------------------------------------------------

def test_persona_definition_defaults():
    pd = PersonaDefinition(name="test", system_prompt="hello")
    assert pd.name == "test"
    assert pd.system_prompt == "hello"
    assert pd.metadata == {}


def test_persona_definition_with_metadata():
    pd = PersonaDefinition(name="test", system_prompt="hello", metadata={"tone": "formal"})
    assert pd.metadata == {"tone": "formal"}


# ---------------------------------------------------------------------------
# PersonaSwitcher — class-based tests
# ---------------------------------------------------------------------------

class TestPersonaSwitcher:
    def test_default_persona(self):
        ps = PersonaSwitcher()
        assert ps.current() == "default"
        assert ps.get_prompt("default") == ""
        assert "default" in ps.list_personas()

    def test_register_and_switch(self):
        ps = PersonaSwitcher()
        ps.register_persona("coder", "You are a coding assistant.")
        prompt = ps.switch_to("coder")
        assert prompt == "You are a coding assistant."
        assert ps.current() == "coder"

    def test_register_overwrites_existing(self):
        ps = PersonaSwitcher()
        ps.register_persona("coder", "Original.")
        ps.register_persona("coder", "Updated.")
        assert ps.get_prompt("coder") == "Updated."

    def test_switch_returns_prompt(self):
        ps = PersonaSwitcher()
        ps.register_persona("analyst", "Analyze data carefully.")
        assert ps.switch_to("analyst") == "Analyze data carefully."

    def test_list_personas(self):
        ps = PersonaSwitcher()
        ps.register_persona("a", "A")
        ps.register_persona("b", "B")
        personas = ps.list_personas()
        assert isinstance(personas, list)
        assert "default" in personas
        assert "a" in personas
        assert "b" in personas

    def test_remove_persona(self):
        ps = PersonaSwitcher()
        ps.register_persona("temp", "Temp")
        ps.remove_persona("temp")
        assert "temp" not in ps.list_personas()

    def test_remove_reverts_current_to_default(self):
        ps = PersonaSwitcher()
        ps.register_persona("temp", "Temp")
        ps.switch_to("temp")
        ps.remove_persona("temp")
        assert ps.current() == "default"

    def test_remove_default_raises(self):
        ps = PersonaSwitcher()
        with pytest.raises(ValueError, match="Cannot remove the default persona"):
            ps.remove_persona("default")

    def test_remove_unknown_raises(self):
        ps = PersonaSwitcher()
        with pytest.raises(KeyError, match="not registered"):
            ps.remove_persona("ghost")

    def test_switch_to_unknown_raises(self):
        ps = PersonaSwitcher()
        with pytest.raises(KeyError, match="not registered"):
            ps.switch_to("ghost")

    def test_get_prompt_unknown_raises(self):
        ps = PersonaSwitcher()
        with pytest.raises(KeyError, match="not registered"):
            ps.get_prompt("ghost")

    def test_name_sanitization_rejects_invalid_chars(self):
        ps = PersonaSwitcher()
        invalid_names = [
            "hello world",
            "foo/bar",
            "foo\\bar",
            "foo.bar",
            "foo@bar",
            "foo$bar",
            "foo!bar",
            "foo<bar>",
            "",
            "a" * 65,
        ]
        for name in invalid_names:
            with pytest.raises(ValueError):
                ps.register_persona(name, "prompt")

    def test_name_sanitization_accepts_valid_chars(self):
        ps = PersonaSwitcher()
        valid_names = [
            "hello_world",
            "hello-world",
            "HelloWorld",
            "abc123",
            "a",
            "a" * 64,
        ]
        for name in valid_names:
            ps.register_persona(name, "prompt")
            assert name in ps.list_personas()

    def test_prompt_truncation(self):
        ps = PersonaSwitcher()
        long_prompt = "x" * 5000
        ps.register_persona("long", long_prompt)
        assert len(ps.get_prompt("long")) == 4096

    def test_register_with_metadata(self):
        ps = PersonaSwitcher()
        ps.register_persona("meta", "prompt", metadata={"key": "value"})
        definition = ps._personas["meta"]
        assert definition.metadata == {"key": "value"}

    def test_register_without_metadata(self):
        ps = PersonaSwitcher()
        ps.register_persona("no_meta", "prompt")
        definition = ps._personas["no_meta"]
        assert definition.metadata == {}

    def test_sanitize_name_non_string_raises(self):
        ps = PersonaSwitcher()
        with pytest.raises(ValueError, match="must be a string"):
            ps.register_persona(123, "prompt")  # type: ignore[arg-type]

    def test_truncate_prompt_non_string_raises(self):
        ps = PersonaSwitcher()
        with pytest.raises(ValueError, match="must be a string"):
            ps.register_persona("name", 123)  # type: ignore[arg-type]
