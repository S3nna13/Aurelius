"""Integration tests for the public ``agent`` namespace."""

from __future__ import annotations

import importlib

import agent as agent_pkg



import pytest

pytestmark = pytest.mark.integration

EXPECTED_EXPORTS = {
    "AgentRuntime": ("agent.agent_runtime", "AgentRuntime"),
    "AgentSpec": ("agent.agent_runtime", "AgentSpec"),
    "AgentMessage": ("agent.agent_runtime", "AgentMessage"),
    "AgentStatus": ("agent.agent_runtime", "AgentStatus"),
    "AgentMemory": ("agent.agent_memory", "AgentMemory"),
    "MemoryEntry": ("agent.agent_memory", "MemoryEntry"),
    "AgentPersistence": ("agent.agent_persistence", "AgentPersistence"),
    "SkillRegistry": ("agent.skill_registry", "SkillRegistry"),
    "SkillSpec": ("agent.skill_registry", "SkillSpec"),
    "Skill": ("agent.skill_library", "Skill"),
    "VoyagerSkillLibrary": ("agent.skill_library", "VoyagerSkillLibrary"),
    "ToolRegistry": ("agent.tool_registry", "ToolRegistry"),
    "SupervisorCoordinator": ("agent.multi_agent", "SupervisorCoordinator"),
    "SwarmCoordinator": ("agent.multi_agent", "SwarmCoordinator"),
    "DebateCoordinator": ("agent.multi_agent", "DebateCoordinator"),
}


def test_src_agent_package_exposes_new_surface_exports() -> None:
    for export_name, (module_name, attr_name) in EXPECTED_EXPORTS.items():
        module = importlib.import_module(module_name)
        assert getattr(agent_pkg, export_name) is getattr(module, attr_name)

    for export_name in EXPECTED_EXPORTS:
        assert export_name in agent_pkg.__all__



