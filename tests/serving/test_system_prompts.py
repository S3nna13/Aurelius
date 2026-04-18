"""Tests for src/serving/system_prompts.py."""

import pytest
from src.serving.system_prompts import SYSTEM_PROMPTS, SystemPromptLibrary

REQUIRED_KEYS = {"default", "coding", "security", "researcher", "concise", "creative"}


def test_system_prompts_has_all_required_keys():
    assert REQUIRED_KEYS.issubset(SYSTEM_PROMPTS.keys())


def test_system_prompt_library_instantiates():
    lib = SystemPromptLibrary()
    assert lib is not None


def test_get_default_returns_non_empty_string():
    lib = SystemPromptLibrary()
    result = lib.get("default")
    assert isinstance(result, str) and len(result) > 0


def test_get_nonexistent_returns_default_prompt():
    lib = SystemPromptLibrary()
    result = lib.get("totally_nonexistent_persona_xyz")
    assert result == lib.get("default")


def test_list_personas_returns_sorted_list_including_default():
    lib = SystemPromptLibrary()
    personas = lib.list_personas()
    assert isinstance(personas, list)
    assert "default" in personas
    assert personas == sorted(personas)


def test_add_increases_list_personas_length():
    lib = SystemPromptLibrary()
    before = len(lib.list_personas())
    lib.add("brand_new_persona", "You are brand new.")
    assert len(lib.list_personas()) == before + 1


def test_render_substitutes_variables():
    lib = SystemPromptLibrary()
    lib.add("greeting", "Hello, {user_name}! How can I help?")
    result = lib.render("greeting", user_name="Alice")
    assert "Alice" in result
    assert "{user_name}" not in result


def test_build_messages_returns_list_of_two_dicts():
    lib = SystemPromptLibrary()
    messages = lib.build_messages("default", "Hello!")
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert all(isinstance(m, dict) for m in messages)


def test_build_messages_first_message_has_role_system():
    lib = SystemPromptLibrary()
    messages = lib.build_messages("default", "Hello!")
    assert messages[0]["role"] == "system"


def test_build_messages_second_message_has_role_user_with_correct_content():
    lib = SystemPromptLibrary()
    user_msg = "What is the capital of France?"
    messages = lib.build_messages("default", user_msg)
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == user_msg
