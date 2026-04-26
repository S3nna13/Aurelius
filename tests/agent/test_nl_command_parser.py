"""Unit tests for src/agent/nl_command_parser.py."""

from __future__ import annotations

import pytest

from src.agent.nl_command_parser import (
    DEFAULT_NL_PARSER,
    NL_PARSER_REGISTRY,
    NLCommandParseError,
    NLCommandParser,
    ParsedCommand,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh() -> NLCommandParser:
    """Return a fresh parser to avoid test pollution."""
    return NLCommandParser()


# ---------------------------------------------------------------------------
# 1. test_run_skill_with_on_arg
# ---------------------------------------------------------------------------


def test_run_skill_with_on_arg():
    parser = _fresh()
    result = parser.parse("run skill code-review on src/main.py")
    assert result.action == "run_skill"
    assert result.target == "code-review"
    assert result.args == {"on": "src/main.py"}
    assert result.raw_text == "run skill code-review on src/main.py"


# ---------------------------------------------------------------------------
# 2. test_run_skill_without_on_arg
# ---------------------------------------------------------------------------


def test_run_skill_without_on_arg():
    parser = _fresh()
    result = parser.parse("execute skill linter")
    assert result.action == "run_skill"
    assert result.target == "linter"
    assert result.args == {}
    assert result.raw_text == "execute skill linter"


# ---------------------------------------------------------------------------
# 3. test_use_skill_alias
# ---------------------------------------------------------------------------


def test_use_skill_alias():
    parser = _fresh()
    result = parser.parse("use skill formatter")
    assert result.action == "run_skill"
    assert result.target == "formatter"
    assert result.args == {}


# ---------------------------------------------------------------------------
# 4. test_list_skills
# ---------------------------------------------------------------------------


def test_list_skills():
    parser = _fresh()
    for text in ("list skills", "show skills", "what skills are available"):
        result = parser.parse(text)
        assert result.action == "list_skills"
        assert result.target is None
        assert result.args == {}


# ---------------------------------------------------------------------------
# 5. test_load_plugin
# ---------------------------------------------------------------------------


def test_load_plugin():
    parser = _fresh()
    result = parser.parse("load plugin my_plugin")
    assert result.action == "load_plugin"
    assert result.target == "my_plugin"
    assert result.args == {}


# ---------------------------------------------------------------------------
# 6. test_install_plugin_alias
# ---------------------------------------------------------------------------


def test_install_plugin_alias():
    parser = _fresh()
    result = parser.parse("install plugin another-plugin")
    assert result.action == "load_plugin"
    assert result.target == "another-plugin"
    assert result.args == {}


# ---------------------------------------------------------------------------
# 7. test_list_plugins
# ---------------------------------------------------------------------------


def test_list_plugins():
    parser = _fresh()
    for text in ("list plugins", "show plugins"):
        result = parser.parse(text)
        assert result.action == "list_plugins"
        assert result.target is None
        assert result.args == {}


# ---------------------------------------------------------------------------
# 8. test_activate_skill
# ---------------------------------------------------------------------------


def test_activate_skill():
    parser = _fresh()
    result = parser.parse("activate skill test-gen")
    assert result.action == "activate_skill"
    assert result.target == "test-gen"
    assert result.args == {}


# ---------------------------------------------------------------------------
# 9. test_enable_skill_alias
# ---------------------------------------------------------------------------


def test_enable_skill_alias():
    parser = _fresh()
    result = parser.parse("enable skill test-gen")
    assert result.action == "activate_skill"
    assert result.target == "test-gen"
    assert result.args == {}


# ---------------------------------------------------------------------------
# 10. test_deactivate_skill
# ---------------------------------------------------------------------------


def test_deactivate_skill():
    parser = _fresh()
    result = parser.parse("deactivate skill doc-writer")
    assert result.action == "deactivate_skill"
    assert result.target == "doc-writer"
    assert result.args == {}


# ---------------------------------------------------------------------------
# 11. test_disable_skill_alias
# ---------------------------------------------------------------------------


def test_disable_skill_alias():
    parser = _fresh()
    result = parser.parse("disable skill doc-writer")
    assert result.action == "deactivate_skill"
    assert result.target == "doc-writer"
    assert result.args == {}


# ---------------------------------------------------------------------------
# 12. test_agent_status_variations
# ---------------------------------------------------------------------------


def test_agent_status_variations():
    parser = _fresh()
    for text in ("show agent status", "what is the agent doing"):
        result = parser.parse(text)
        assert result.action == "agent_status"
        assert result.target is None
        assert result.args == {}


# ---------------------------------------------------------------------------
# 13. test_list_agents
# ---------------------------------------------------------------------------


def test_list_agents():
    parser = _fresh()
    for text in ("list agents", "show agents"):
        result = parser.parse(text)
        assert result.action == "list_agents"
        assert result.target is None
        assert result.args == {}


# ---------------------------------------------------------------------------
# 14. test_run_task
# ---------------------------------------------------------------------------


def test_run_task():
    parser = _fresh()
    for text in ("run task cleanup", "execute task cleanup"):
        result = parser.parse(text)
        assert result.action == "run_task"
        assert result.target == "cleanup"
        assert result.args == {}


# ---------------------------------------------------------------------------
# 15. test_show_board_variations
# ---------------------------------------------------------------------------


def test_show_board_variations():
    parser = _fresh()
    for text in ("show board", "show tasks", "show work"):
        result = parser.parse(text)
        assert result.action == "show_board"
        assert result.target is None
        assert result.args == {}


# ---------------------------------------------------------------------------
# 16. test_unknown_input_fallback_to_chat
# ---------------------------------------------------------------------------


def test_unknown_input_fallback_to_chat():
    parser = _fresh()
    result = parser.parse("tell me a joke")
    assert result.action == "chat"
    assert result.target is None
    assert result.args == {}
    assert result.raw_text == "tell me a joke"


# ---------------------------------------------------------------------------
# 17. test_empty_string_raises
# ---------------------------------------------------------------------------


def test_empty_string_raises():
    parser = _fresh()
    with pytest.raises(
        NLCommandParseError, match="Input text must be a non-empty string"
    ):
        parser.parse("")


# ---------------------------------------------------------------------------
# 18. test_whitespace_only_raises
# ---------------------------------------------------------------------------


def test_whitespace_only_raises():
    parser = _fresh()
    with pytest.raises(
        NLCommandParseError, match="Input text must be a non-empty string"
    ):
        parser.parse("   \t\n  ")


# ---------------------------------------------------------------------------
# 19. test_case_insensitivity
# ---------------------------------------------------------------------------


def test_case_insensitivity():
    parser = _fresh()
    result = parser.parse("RUN SKILL Code-Review ON src/main.py")
    assert result.action == "run_skill"
    assert result.target == "code-review"
    assert result.args == {"on": "src/main.py"}


# ---------------------------------------------------------------------------
# 20. test_default_singleton_exists
# ---------------------------------------------------------------------------


def test_default_singleton_exists():
    assert isinstance(DEFAULT_NL_PARSER, NLCommandParser)


# ---------------------------------------------------------------------------
# 21. test_registry_singleton_exists
# ---------------------------------------------------------------------------


def test_registry_singleton_exists():
    assert "default" in NL_PARSER_REGISTRY
    assert NL_PARSER_REGISTRY["default"] is DEFAULT_NL_PARSER


# ---------------------------------------------------------------------------
# 22. test_custom_parser_in_registry
# ---------------------------------------------------------------------------


def test_custom_parser_in_registry():
    custom = NLCommandParser()
    NL_PARSER_REGISTRY["custom"] = custom
    assert NL_PARSER_REGISTRY["custom"] is custom
    # Clean up to avoid pollution
    del NL_PARSER_REGISTRY["custom"]


# ---------------------------------------------------------------------------
# 23. test_parsed_command_defaults
# ---------------------------------------------------------------------------


def test_parsed_command_defaults():
    cmd = ParsedCommand(action="chat")
    assert cmd.action == "chat"
    assert cmd.target is None
    assert cmd.args == {}
    assert cmd.raw_text == ""
