"""Legacy Aurelius agent registry backed by the canonical registry snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aurelius._registry_snapshot import (
    AGENT_CATALOG,
)
from aurelius._registry_snapshot import (
    AGENT_CATEGORIES as SNAPSHOT_AGENT_CATEGORIES,
)


@dataclass(frozen=True)
class AgentRecord:
    id: str
    name: str
    description: str
    category: str
    capabilities: list[str] = field(default_factory=list)
    enabled: bool = False
    default_tools: list[str] = field(default_factory=list)
    icon: str = "Bot"
    color: str = "#4fc3f7"
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "capabilities": list(self.capabilities),
            "enabled": self.enabled,
            "default_tools": list(self.default_tools),
            "icon": self.icon,
            "color": self.color,
            "parameters": dict(self.parameters),
        }


AgentType = AgentRecord


def _build_agent(record: dict[str, Any]) -> AgentRecord:
    return AgentRecord(
        id=str(record["id"]),
        name=str(record["name"]),
        description=str(record["description"]),
        category=str(record["category"]),
        capabilities=list(record.get("capabilities", [])),
        enabled=bool(record.get("enabled", False)),
        default_tools=list(record.get("default_tools", [])),
        icon=str(record.get("icon", "Bot")),
        color=str(record.get("color", "#4fc3f7")),
        parameters=dict(record.get("parameters", {})),
    )


AGENT_CATEGORIES = list(SNAPSHOT_AGENT_CATEGORIES)

ALL_AGENTS: list[AgentRecord] = [_build_agent(agent) for agent in AGENT_CATALOG]
AGENT_REGISTRY: dict[str, AgentRecord] = {agent.id: agent for agent in ALL_AGENTS}
AGENTS_BY_CATEGORY: dict[str, list[AgentRecord]] = {}
for agent in ALL_AGENTS:
    AGENTS_BY_CATEGORY.setdefault(agent.category, []).append(agent)

RESEARCH_AGENT: str = "research"
DEVOPS_AGENT: str = "devops"
CREATIVE_AGENT: str = "creative"
TUTOR_AGENT: str = "tutor"


def agent_to_dict(agent: AgentRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(agent, AgentRecord):
        return agent.to_dict()
    return {
        "id": str(agent["id"]),
        "name": str(agent["name"]),
        "description": str(agent["description"]),
        "category": str(agent["category"]),
        "capabilities": list(agent.get("capabilities", [])),
        "enabled": bool(agent.get("enabled", False)),
        "default_tools": list(agent.get("default_tools", [])),
        "icon": str(agent.get("icon", "Bot")),
        "color": str(agent.get("color", "#4fc3f7")),
        "parameters": dict(agent.get("parameters", {})),
    }


__all__ = [
    "AgentRecord",
    "AgentType",
    "AGENT_REGISTRY",
    "AGENTS_BY_CATEGORY",
    "AGENT_CATEGORIES",
    "ALL_AGENTS",
    "RESEARCH_AGENT",
    "DEVOPS_AGENT",
    "CREATIVE_AGENT",
    "TUTOR_AGENT",
    "agent_to_dict",
]
