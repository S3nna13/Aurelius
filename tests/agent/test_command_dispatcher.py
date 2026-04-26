"""Unit tests for src/agent/command_dispatcher.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent.command_dispatcher import (
    COMMAND_DISPATCHER_REGISTRY,
    DEFAULT_COMMAND_DISPATCHER,
    CommandDispatchError,
    CommandDispatcher,
    DispatchResult,
)
from src.agent.nl_command_parser import ParsedCommand
from src.agent.skill_executor import SkillContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cmd(
    action: str,
    target: str | None = None,
    args: dict[str, str] | None = None,
    raw_text: str = "",
) -> ParsedCommand:
    return ParsedCommand(
        action=action,
        target=target,
        args=args or {},
        raw_text=raw_text,
    )


# ---------------------------------------------------------------------------
# 1. test_dispatch_chat_returns_raw_text
# ---------------------------------------------------------------------------


def test_dispatch_chat_returns_raw_text():
    dispatcher = CommandDispatcher()
    cmd = _make_cmd("chat", raw_text="hello there")
    result = dispatcher.dispatch(cmd)
    assert result.success is True
    assert result.output == "hello there"


# ---------------------------------------------------------------------------
# 2. test_dispatch_run_skill_with_mocks
# ---------------------------------------------------------------------------


def test_dispatch_run_skill_with_mocks():
    catalog = MagicMock()
    skill = MagicMock()
    skill.instructions = "do something"
    catalog.get.return_value = skill

    executor = MagicMock()
    exec_result = MagicMock()
    exec_result.output = "done"
    executor.execute.return_value = exec_result

    dispatcher = CommandDispatcher(skill_catalog=catalog, skill_executor=executor)
    cmd = _make_cmd("run_skill", target="my_skill", args={"param": "value"})
    result = dispatcher.dispatch(cmd)

    assert result.success is True
    assert result.output == "done"
    catalog.get.assert_called_once_with("my_skill")
    executor.execute.assert_called_once_with(
        "my_skill",
        "do something",
        SkillContext(variables={"param": "value"}),
    )


# ---------------------------------------------------------------------------
# 3. test_dispatch_list_skills_with_mock_catalog
# ---------------------------------------------------------------------------


def test_dispatch_list_skills_with_mock_catalog():
    catalog = MagicMock()
    entry1 = MagicMock()
    entry1.skill_id = "skill_a"
    entry2 = MagicMock()
    entry2.skill_id = "skill_b"
    catalog.list.return_value = [entry1, entry2]

    dispatcher = CommandDispatcher(skill_catalog=catalog)
    cmd = _make_cmd("list_skills")
    result = dispatcher.dispatch(cmd)

    assert result.success is True
    assert "skill_a" in result.output
    assert "skill_b" in result.output


# ---------------------------------------------------------------------------
# 4. test_dispatch_activate_skill_with_mock_catalog
# ---------------------------------------------------------------------------


def test_dispatch_activate_skill_with_mock_catalog():
    catalog = MagicMock()
    dispatcher = CommandDispatcher(skill_catalog=catalog)
    cmd = _make_cmd("activate_skill", target="skill_x")
    result = dispatcher.dispatch(cmd)

    assert result.success is True
    catalog.activate.assert_called_once_with("skill_x")


# ---------------------------------------------------------------------------
# 5. test_dispatch_deactivate_skill_with_mock_catalog
# ---------------------------------------------------------------------------


def test_dispatch_deactivate_skill_with_mock_catalog():
    catalog = MagicMock()
    dispatcher = CommandDispatcher(skill_catalog=catalog)
    cmd = _make_cmd("deactivate_skill", target="skill_y")
    result = dispatcher.dispatch(cmd)

    assert result.success is True
    catalog.deactivate.assert_called_once_with("skill_y")


# ---------------------------------------------------------------------------
# 6. test_dispatch_load_plugin_with_mock_loader
# ---------------------------------------------------------------------------


def test_dispatch_load_plugin_with_mock_loader():
    loader = MagicMock()
    dispatcher = CommandDispatcher(plugin_loader=loader)
    cmd = _make_cmd(
        "load_plugin", target="plugin_z", args={"entry_point": "plugin_z.main"}
    )
    result = dispatcher.dispatch(cmd)

    assert result.success is True
    loader.load.assert_called_once_with("plugin_z", "plugin_z.main")


# ---------------------------------------------------------------------------
# 7. test_dispatch_load_plugin_uses_target_when_no_entry_point
# ---------------------------------------------------------------------------


def test_dispatch_load_plugin_uses_target_when_no_entry_point():
    loader = MagicMock()
    dispatcher = CommandDispatcher(plugin_loader=loader)
    cmd = _make_cmd("load_plugin", target="plugin_z")
    result = dispatcher.dispatch(cmd)

    assert result.success is True
    loader.load.assert_called_once_with("plugin_z", "plugin_z")


# ---------------------------------------------------------------------------
# 8. test_dispatch_list_plugins_with_mock_loader
# ---------------------------------------------------------------------------


def test_dispatch_list_plugins_with_mock_loader():
    loader = MagicMock()
    loader.list_loaded.return_value = ["p1", "p2"]
    dispatcher = CommandDispatcher(plugin_loader=loader)
    cmd = _make_cmd("list_plugins")
    result = dispatcher.dispatch(cmd)

    assert result.success is True
    assert "p1" in result.output
    assert "p2" in result.output


# ---------------------------------------------------------------------------
# 9. test_dispatch_agent_status_returns_generic
# ---------------------------------------------------------------------------


def test_dispatch_agent_status_returns_generic():
    dispatcher = CommandDispatcher()
    cmd = _make_cmd("agent_status")
    result = dispatcher.dispatch(cmd)

    assert result.success is True
    assert "status" in result.output.lower()


# ---------------------------------------------------------------------------
# 10. test_dispatch_show_board_returns_generic
# ---------------------------------------------------------------------------


def test_dispatch_show_board_returns_generic():
    dispatcher = CommandDispatcher()
    cmd = _make_cmd("show_board")
    result = dispatcher.dispatch(cmd)

    assert result.success is True
    assert "board" in result.output.lower()


# ---------------------------------------------------------------------------
# 11. test_dispatch_run_task_returns_generic
# ---------------------------------------------------------------------------


def test_dispatch_run_task_returns_generic():
    dispatcher = CommandDispatcher()
    cmd = _make_cmd("run_task", target="task_1")
    result = dispatcher.dispatch(cmd)

    assert result.success is True
    assert "task_1" in result.output


# ---------------------------------------------------------------------------
# 12. test_dispatch_list_agents_returns_generic
# ---------------------------------------------------------------------------


def test_dispatch_list_agents_returns_generic():
    dispatcher = CommandDispatcher()
    cmd = _make_cmd("list_agents")
    result = dispatcher.dispatch(cmd)

    assert result.success is True
    assert "agents" in result.output.lower()


# ---------------------------------------------------------------------------
# 13. test_dispatch_missing_skill_catalog_for_list_skills
# ---------------------------------------------------------------------------


def test_dispatch_missing_skill_catalog_for_list_skills():
    dispatcher = CommandDispatcher()
    cmd = _make_cmd("list_skills")
    result = dispatcher.dispatch(cmd)

    assert result.success is False
    assert "skill_catalog" in result.output


# ---------------------------------------------------------------------------
# 14. test_dispatch_missing_skill_executor_for_run_skill
# ---------------------------------------------------------------------------


def test_dispatch_missing_skill_executor_for_run_skill():
    catalog = MagicMock()
    dispatcher = CommandDispatcher(skill_catalog=catalog)
    cmd = _make_cmd("run_skill", target="s")
    result = dispatcher.dispatch(cmd)

    assert result.success is False
    assert "skill_executor" in result.output


# ---------------------------------------------------------------------------
# 15. test_dispatch_missing_plugin_loader_for_load_plugin
# ---------------------------------------------------------------------------


def test_dispatch_missing_plugin_loader_for_load_plugin():
    dispatcher = CommandDispatcher()
    cmd = _make_cmd("load_plugin", target="p")
    result = dispatcher.dispatch(cmd)

    assert result.success is False
    assert "plugin_loader" in result.output


# ---------------------------------------------------------------------------
# 16. test_dispatch_missing_plugin_loader_for_list_plugins
# ---------------------------------------------------------------------------


def test_dispatch_missing_plugin_loader_for_list_plugins():
    dispatcher = CommandDispatcher()
    cmd = _make_cmd("list_plugins")
    result = dispatcher.dispatch(cmd)

    assert result.success is False
    assert "plugin_loader" in result.output


# ---------------------------------------------------------------------------
# 17. test_dispatch_unknown_action_raises
# ---------------------------------------------------------------------------


def test_dispatch_unknown_action_raises():
    dispatcher = CommandDispatcher()
    cmd = _make_cmd("unknown_action")
    with pytest.raises(CommandDispatchError, match="Unknown action"):
        dispatcher.dispatch(cmd)


# ---------------------------------------------------------------------------
# 18. test_custom_handler_registration
# ---------------------------------------------------------------------------


def test_custom_handler_registration():
    dispatcher = CommandDispatcher()

    def custom_handler(cmd: ParsedCommand) -> DispatchResult:
        return DispatchResult(success=True, output="custom")

    dispatcher.register_handler("custom_action", custom_handler)
    cmd = _make_cmd("custom_action")
    result = dispatcher.dispatch(cmd)

    assert result.output == "custom"


# ---------------------------------------------------------------------------
# 19. test_custom_handler_override_default
# ---------------------------------------------------------------------------


def test_custom_handler_override_default():
    dispatcher = CommandDispatcher()

    def chat_handler(cmd: ParsedCommand) -> DispatchResult:
        return DispatchResult(success=True, output="overridden")

    dispatcher.register_handler("chat", chat_handler)
    cmd = _make_cmd("chat", raw_text="hi")
    result = dispatcher.dispatch(cmd)

    assert result.output == "overridden"


# ---------------------------------------------------------------------------
# 20. test_exception_during_handler_caught
# ---------------------------------------------------------------------------


def test_exception_during_handler_caught():
    dispatcher = CommandDispatcher()

    def bad_handler(cmd: ParsedCommand) -> DispatchResult:
        raise RuntimeError("boom")

    dispatcher.register_handler("bad", bad_handler)
    cmd = _make_cmd("bad")
    result = dispatcher.dispatch(cmd)

    assert result.success is False
    assert "boom" in result.output


# ---------------------------------------------------------------------------
# 21. test_registry_singleton
# ---------------------------------------------------------------------------


def test_registry_singleton():
    assert "default" in COMMAND_DISPATCHER_REGISTRY
    assert COMMAND_DISPATCHER_REGISTRY["default"] is DEFAULT_COMMAND_DISPATCHER


# ---------------------------------------------------------------------------
# 22. test_empty_target_handled_gracefully
# ---------------------------------------------------------------------------


def test_empty_target_handled_gracefully():
    catalog = MagicMock()
    catalog.activate.side_effect = ValueError("skill_id must be non-empty")
    dispatcher = CommandDispatcher(skill_catalog=catalog)
    cmd = _make_cmd("activate_skill", target="")
    result = dispatcher.dispatch(cmd)

    assert result.success is False
