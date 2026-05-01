"""Aurelius API — agent registry, skills registry, and plugin endpoints.

Serves agent types, skills, and plugin management from the Python layer.
Connects to the Node.js BFF which proxies to the frontend.
"""

from __future__ import annotations

import json
from typing import Any

try:
    from aurelius.agent_registry import (
        ALL_AGENTS,
        AGENT_REGISTRY,
        AGENTS_BY_CATEGORY,
        agent_to_dict,
    )
    from aurelius.skills_registry import (
        ALL_SKILLS,
        SKILL_REGISTRY,
        SKILLS_BY_CATEGORY,
        skill_to_dict,
    )
    _HAS_REGISTRIES = True
except ImportError:
    _HAS_REGISTRIES = False
    ALL_AGENTS = []
    AGENT_REGISTRY = {}
    AGENTS_BY_CATEGORY = {}
    ALL_SKILLS = []
    SKILL_REGISTRY = {}
    SKILLS_BY_CATEGORY = {}


def get_registry_snapshot(category: str | None = None) -> dict[str, Any]:
    agents = AGENTS_BY_CATEGORY.get(category, []) if category else ALL_AGENTS
    skills = SKILLS_BY_CATEGORY.get(category, []) if category else ALL_SKILLS
    return {
        "agents": [agent_to_dict(agent) for agent in agents],
        "agent_categories": sorted(AGENTS_BY_CATEGORY.keys()),
        "skills": [skill_to_dict(skill) for skill in skills],
        "skill_categories": sorted(SKILLS_BY_CATEGORY.keys()),
    }


def register_api_routes(app: Any) -> None:
    """Register all Python API routes on the given Flask/FastAPI app."""

    @app.route("/api/registry/agents", methods=["GET"])
    def list_agents():
        category = app.request.args.get("category")
        if category:
            agents = AGENTS_BY_CATEGORY.get(category, [])
        else:
            agents = ALL_AGENTS
        return {
            "agents": [
                {
                    "id": a.id, "name": a.name, "description": a.description,
                    "category": a.category, "capabilities": a.capabilities,
                    "default_tools": a.default_tools, "icon": a.icon,
                }
                for a in agents
            ],
            "total": len(agents),
            "categories": list(AGENTS_BY_CATEGORY.keys()),
        }

    @app.route("/api/registry/agents/<agent_id>", methods=["GET"])
    def get_agent(agent_id: str):
        agent = AGENT_REGISTRY.get(agent_id)
        if not agent:
            return {"error": f"Agent '{agent_id}' not found"}, 404
        return {"agent": {
            "id": agent.id, "name": agent.name, "description": agent.description,
            "category": agent.category, "capabilities": agent.capabilities,
            "default_tools": agent.default_tools, "icon": agent.icon,
            "parameters": agent.parameters,
        }}

    @app.route("/api/registry/skills", methods=["GET"])
    def list_skills():
        category = app.request.args.get("category")
        if category:
            skills = SKILLS_BY_CATEGORY.get(category, [])
        else:
            skills = ALL_SKILLS
        return {
            "skills": [
                {
                    "id": s.id, "name": s.name, "description": s.description,
                    "category": s.category, "agent_types": s.agent_types, "tags": s.tags,
                }
                for s in skills
            ],
            "total": len(skills),
            "categories": list(SKILLS_BY_CATEGORY.keys()),
        }

    @app.route("/api/registry/skills/<skill_id>", methods=["GET"])
    def get_skill(skill_id: str):
        skill = SKILL_REGISTRY.get(skill_id)
        if not skill:
            return {"error": f"Skill '{skill_id}' not found"}, 404
        return {"skill": {
            "id": skill.id, "name": skill.name, "description": skill.description,
            "category": skill.category, "agent_types": skill.agent_types, "tags": skill.tags,
        }}


def get_agent_for_task(task: str) -> dict[str, Any]:
    """Route a task description to the best matching agent type."""
    task_lower = task.lower()
    scores: list[tuple[float, Any]] = []

    for agent in ALL_AGENTS:
        score = sum(2 for cap in agent.capabilities if cap.lower() in task_lower)
        score += sum(1 for tool in agent.default_tools if tool.lower() in task_lower)
        if score > 0:
            scores.append((score, agent))

    scores.sort(key=lambda x: x[0], reverse=True)

    if scores:
        best = scores[0][1]
        return {
            "agent_id": best.id,
            "agent_name": best.name,
            "category": best.category,
            "capabilities": best.capabilities,
            "confidence": scores[0][0] / max(len(best.capabilities), 1),
        }

    return {
        "agent_id": "general",
        "agent_name": "General Assistant",
        "category": "general",
        "capabilities": ["chat", "help", "answer"],
        "confidence": 1.0,
    }


def get_skills_for_agent(agent_id: str) -> list[dict[str, Any]]:
    """Get all skills that match an agent type."""
    skills = []
    for skill in ALL_SKILLS:
        if agent_id in skill.agent_types or not skill.agent_types:
            skills.append({
                "id": skill.id, "name": skill.name,
                "description": skill.description, "category": skill.category,
            })
    return skills
