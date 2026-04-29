"""Integration tests for the public ``src.agent`` namespace."""

from __future__ import annotations

import importlib

import src.agent as agent_pkg

EXPECTED_EXPORTS = {
    "AgentRuntime": ("src.agent.agent_runtime", "AgentRuntime"),
    "AgentSpec": ("src.agent.agent_runtime", "AgentSpec"),
    "AgentMessage": ("src.agent.agent_runtime", "AgentMessage"),
    "AgentStatus": ("src.agent.agent_runtime", "AgentStatus"),
    "AgentMemory": ("src.agent.agent_memory", "AgentMemory"),
    "MemoryEntry": ("src.agent.agent_memory", "MemoryEntry"),
    "AgentPersistence": ("src.agent.agent_persistence", "AgentPersistence"),
    "SkillRegistry": ("src.agent.skill_registry", "SkillRegistry"),
    "SkillSpec": ("src.agent.skill_registry", "SkillSpec"),
    "Skill": ("src.agent.skill_library", "Skill"),
    "VoyagerSkillLibrary": ("src.agent.skill_library", "VoyagerSkillLibrary"),
    "ToolRegistry": ("src.agent.tool_registry", "ToolRegistry"),
    "SupervisorCoordinator": ("src.agent.multi_agent", "SupervisorCoordinator"),
    "SwarmCoordinator": ("src.agent.multi_agent", "SwarmCoordinator"),
    "DebateCoordinator": ("src.agent.multi_agent", "DebateCoordinator"),
}


def test_src_agent_package_exposes_new_surface_exports() -> None:
    for export_name, (module_name, attr_name) in EXPECTED_EXPORTS.items():
        module = importlib.import_module(module_name)
        assert getattr(agent_pkg, export_name) is getattr(module, attr_name)

    for export_name in EXPECTED_EXPORTS:
        assert export_name in agent_pkg.__all__
