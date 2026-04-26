"""Integration tests for PluginHookRegistry wired into AGENT_LOOP_REGISTRY."""

from __future__ import annotations

from src.agent import AGENT_LOOP_REGISTRY
from src.agent.plugin_hook import HOOK_REGISTRY, PluginHookRegistry

# ---------------------------------------------------------------------------
# 1. "plugin_hook" key present in AGENT_LOOP_REGISTRY
# ---------------------------------------------------------------------------


def test_plugin_hook_in_agent_loop_registry():
    assert "plugin_hook" in AGENT_LOOP_REGISTRY, (
        "'plugin_hook' must be registered in AGENT_LOOP_REGISTRY"
    )


# ---------------------------------------------------------------------------
# 2. Class from registry is constructable; register + fire works
# ---------------------------------------------------------------------------


def test_registry_class_construct_and_fire():
    cls = AGENT_LOOP_REGISTRY["plugin_hook"]
    reg = cls()
    fired = []
    reg.register("pre_tool_call", lambda **kw: fired.append(kw.get("tool_name")))
    reg.fire("pre_tool_call", tool_name="grep")
    assert fired == ["grep"]


# ---------------------------------------------------------------------------
# 3. Module-level HOOK_REGISTRY importable and is a PluginHookRegistry
# ---------------------------------------------------------------------------


def test_hook_registry_singleton_importable():
    assert isinstance(HOOK_REGISTRY, PluginHookRegistry)


# ---------------------------------------------------------------------------
# 4. Regression guard — existing AGENT_LOOP_REGISTRY keys still present
# ---------------------------------------------------------------------------


def test_existing_agent_loop_registry_keys_intact():
    expected_keys = {
        "react",
        "safe_dispatch",
        "beam_plan",
        "task_decompose",
        "dispatch_task",
        "budget_bounded",
        "agent_swarm",
    }
    for key in expected_keys:
        assert key in AGENT_LOOP_REGISTRY, (
            f"Regression: existing key {key!r} missing from AGENT_LOOP_REGISTRY"
        )
