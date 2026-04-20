"""Integration: tool sandbox denylist registry + config flag."""

from __future__ import annotations


def test_tool_guard_registry_registered():
    import src.agent as agent

    assert "sandbox_denylist" in agent.TOOL_GUARD_REGISTRY
    assert agent.TOOL_GUARD_REGISTRY["sandbox_denylist"] is agent.ToolSandboxDenylist


def test_prior_agent_registries_intact():
    import src.agent as agent

    assert "react" in agent.AGENT_LOOP_REGISTRY
    assert "dispatch_task" in agent.AGENT_LOOP_REGISTRY
    assert "budget_bounded" in agent.AGENT_LOOP_REGISTRY


def test_default_denylist_re_exported():
    import src.agent as agent

    assert len(agent.TOOL_SANDBOX_DEFAULT_DENYLIST) >= 25


def test_config_flag_defaults_off():
    from src.model.config import AureliusConfig

    cfg = AureliusConfig()
    assert cfg.agent_tool_sandbox_denylist_enabled is False


def test_smoke_guard_blocks_dangerous_call():
    from src.agent import ToolSandboxDenylist

    guard = ToolSandboxDenylist()
    v = guard.evaluate("shell", {"cmd": "rm -rf /"})
    assert v.allowed is False
    assert v.violated_rules
