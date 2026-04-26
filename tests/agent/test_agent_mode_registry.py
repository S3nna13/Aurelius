"""Unit tests for src/agent/agent_mode_registry.py."""

from __future__ import annotations

import pytest

from src.agent.agent_mode_registry import (
    AGENT_MODE_REGISTRY,
    DEFAULT_MODE_REGISTRY,
    AgentMode,
    AgentModeError,
    AgentModeRegistry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_registry() -> AgentModeRegistry:
    """Return a fresh registry to avoid test pollution."""
    return AgentModeRegistry()


def _sample_mode(mode_id: str = "test") -> AgentMode:
    """Return a minimally valid AgentMode for testing."""
    return AgentMode(
        mode_id=mode_id,
        name="Test Mode",
        description="A mode for testing.",
        system_prompt_prefix="You are in test mode.",
        allowed_tools=["read", "write"],
        response_style="concise",
    )


# ---------------------------------------------------------------------------
# 1. test_register_and_retrieve_mode
# ---------------------------------------------------------------------------


def test_register_and_retrieve_mode():
    reg = _fresh_registry()
    mode = _sample_mode("retrieve")
    reg.register(mode)
    retrieved = reg.get("retrieve")
    assert retrieved is mode


# ---------------------------------------------------------------------------
# 2. test_unregister_works
# ---------------------------------------------------------------------------


def test_unregister_works():
    reg = _fresh_registry()
    mode = _sample_mode("remove_me")
    reg.register(mode)
    reg.unregister("remove_me")
    with pytest.raises(AgentModeError, match="mode_id 'remove_me' not found"):
        reg.get("remove_me")


# ---------------------------------------------------------------------------
# 3. test_unregister_unknown_raises
# ---------------------------------------------------------------------------


def test_unregister_unknown_raises():
    reg = _fresh_registry()
    with pytest.raises(AgentModeError, match="mode_id 'missing' not found"):
        reg.unregister("missing")


# ---------------------------------------------------------------------------
# 4. test_list_modes_returns_all
# ---------------------------------------------------------------------------


def test_list_modes_returns_all():
    reg = _fresh_registry()
    reg.register(_sample_mode("a"))
    reg.register(_sample_mode("b"))
    modes = reg.list_modes()
    assert len(modes) == 2
    ids = {m.mode_id for m in modes}
    assert ids == {"a", "b"}


# ---------------------------------------------------------------------------
# 5. test_find_by_tool_filters_correctly
# ---------------------------------------------------------------------------


def test_find_by_tool_filters_correctly():
    reg = _fresh_registry()
    reg.register(
        AgentMode(
            mode_id="all_tools",
            name="All",
            description="",
            system_prompt_prefix="",
            allowed_tools=[],
            response_style="",
        )
    )
    reg.register(
        AgentMode(
            mode_id="restricted",
            name="Restricted",
            description="",
            system_prompt_prefix="",
            allowed_tools=["read", "search"],
            response_style="",
        )
    )
    results = reg.find_by_tool("read")
    ids = {m.mode_id for m in results}
    assert "all_tools" in ids
    assert "restricted" in ids

    results_search = reg.find_by_tool("search")
    ids_search = {m.mode_id for m in results_search}
    assert "all_tools" in ids_search
    assert "restricted" in ids_search

    results_write = reg.find_by_tool("write")
    ids_write = {m.mode_id for m in results_write}
    assert "all_tools" in ids_write
    assert "restricted" not in ids_write


# ---------------------------------------------------------------------------
# 6. test_default_mode_returns_code
# ---------------------------------------------------------------------------


def test_default_mode_returns_code():
    reg = AgentModeRegistry()
    reg.register(
        AgentMode(
            mode_id="code",
            name="Code",
            description="",
            system_prompt_prefix="",
            allowed_tools=[],
            response_style="",
        )
    )
    reg.register(
        AgentMode(
            mode_id="other",
            name="Other",
            description="",
            system_prompt_prefix="",
            allowed_tools=[],
            response_style="",
        )
    )
    assert reg.default_mode().mode_id == "code"


# ---------------------------------------------------------------------------
# 7. test_default_mode_falls_back_to_first
# ---------------------------------------------------------------------------


def test_default_mode_falls_back_to_first():
    reg = _fresh_registry()
    reg.register(_sample_mode("first"))
    assert reg.default_mode().mode_id == "first"


# ---------------------------------------------------------------------------
# 8. test_default_mode_empty_registry_raises
# ---------------------------------------------------------------------------


def test_default_mode_empty_registry_raises():
    reg = _fresh_registry()
    with pytest.raises(AgentModeError, match="no modes registered"):
        reg.default_mode()


# ---------------------------------------------------------------------------
# 9. test_switch_context_merges_prefix
# ---------------------------------------------------------------------------


def test_switch_context_merges_prefix():
    reg = _fresh_registry()
    reg.register(_sample_mode("ctx"))
    ctx = {"system_prompt": "Original prompt."}
    new_ctx = reg.switch_context("ctx", ctx)
    assert new_ctx["system_prompt"] == "You are in test mode.\nOriginal prompt."


# ---------------------------------------------------------------------------
# 10. test_switch_context_creates_prompt_when_missing
# ---------------------------------------------------------------------------


def test_switch_context_creates_prompt_when_missing():
    reg = _fresh_registry()
    reg.register(_sample_mode("ctx"))
    new_ctx = reg.switch_context("ctx", {})
    assert new_ctx["system_prompt"] == "You are in test mode."


# ---------------------------------------------------------------------------
# 11. test_is_tool_allowed_empty_list_means_all
# ---------------------------------------------------------------------------


def test_is_tool_allowed_empty_list_means_all():
    reg = _fresh_registry()
    reg.register(
        AgentMode(
            mode_id="open",
            name="Open",
            description="",
            system_prompt_prefix="",
            allowed_tools=[],
            response_style="",
        )
    )
    assert reg.is_tool_allowed("open", "anything") is True
    assert reg.is_tool_allowed("open", "read") is True


# ---------------------------------------------------------------------------
# 12. test_is_tool_allowed_restricted
# ---------------------------------------------------------------------------


def test_is_tool_allowed_restricted():
    reg = _fresh_registry()
    reg.register(
        AgentMode(
            mode_id="limited",
            name="Limited",
            description="",
            system_prompt_prefix="",
            allowed_tools=["read", "search"],
            response_style="",
        )
    )
    assert reg.is_tool_allowed("limited", "read") is True
    assert reg.is_tool_allowed("limited", "search") is True
    assert reg.is_tool_allowed("limited", "write") is False


# ---------------------------------------------------------------------------
# 13. test_pre_registered_defaults_exist
# ---------------------------------------------------------------------------


def test_pre_registered_defaults_exist():
    defaults = DEFAULT_MODE_REGISTRY.list_modes()
    ids = {m.mode_id for m in defaults}
    assert ids == {"code", "architect", "ask", "debug", "custom"}


# ---------------------------------------------------------------------------
# 14. test_duplicate_registration_raises
# ---------------------------------------------------------------------------


def test_duplicate_registration_raises():
    reg = _fresh_registry()
    mode = _sample_mode("dup")
    reg.register(mode)
    with pytest.raises(AgentModeError, match="mode_id 'dup' is already registered"):
        reg.register(mode)


# ---------------------------------------------------------------------------
# 15. test_registry_singleton
# ---------------------------------------------------------------------------


def test_registry_singleton():
    assert isinstance(DEFAULT_MODE_REGISTRY, AgentModeRegistry)
    assert "default" in AGENT_MODE_REGISTRY
    assert AGENT_MODE_REGISTRY["default"] is DEFAULT_MODE_REGISTRY


# ---------------------------------------------------------------------------
# 16. test_custom_registry_in_agent_mode_registry
# ---------------------------------------------------------------------------


def test_custom_registry_in_agent_mode_registry():
    custom = AgentModeRegistry()
    custom.register(_sample_mode("custom_mode"))
    AGENT_MODE_REGISTRY["test_custom"] = custom
    assert AGENT_MODE_REGISTRY["test_custom"] is custom
    assert (
        AGENT_MODE_REGISTRY["test_custom"].get("custom_mode").mode_id == "custom_mode"
    )
    del AGENT_MODE_REGISTRY["test_custom"]


# ---------------------------------------------------------------------------
# 17. test_mode_validation_empty_mode_id_raises
# ---------------------------------------------------------------------------


def test_mode_validation_empty_mode_id_raises():
    reg = _fresh_registry()
    bad = _sample_mode("")
    with pytest.raises(AgentModeError, match="mode_id must be a non-empty string"):
        reg.register(bad)


# ---------------------------------------------------------------------------
# 18. test_mode_validation_whitespace_mode_id_raises
# ---------------------------------------------------------------------------


def test_mode_validation_whitespace_mode_id_raises():
    reg = _fresh_registry()
    bad = _sample_mode("   ")
    with pytest.raises(AgentModeError, match="mode_id must be a non-empty string"):
        reg.register(bad)


# ---------------------------------------------------------------------------
# 19. test_get_unknown_mode_raises
# ---------------------------------------------------------------------------


def test_get_unknown_mode_raises():
    reg = _fresh_registry()
    with pytest.raises(AgentModeError, match="mode_id 'ghost' not found"):
        reg.get("ghost")


# ---------------------------------------------------------------------------
# 20. test_switch_context_preserves_other_keys
# ---------------------------------------------------------------------------


def test_switch_context_preserves_other_keys():
    reg = _fresh_registry()
    reg.register(_sample_mode("ctx"))
    ctx = {"temperature": 0.7, "system_prompt": "Base."}
    new_ctx = reg.switch_context("ctx", ctx)
    assert new_ctx["temperature"] == 0.7
    assert new_ctx["system_prompt"] == "You are in test mode.\nBase."
