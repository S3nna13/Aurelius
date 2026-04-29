"""Aurelius — Agent Registry, Skills Registry, Plugin System, and API.

Complete agent and skills management system.
"""

import sys
from importlib import import_module

_src_model = import_module("src.model")
sys.modules.setdefault("aurelius.model", _src_model)
if "src.model.transformer" in sys.modules:
    sys.modules.setdefault("aurelius.model.transformer", sys.modules["src.model.transformer"])

_src_training = import_module("src.training")
sys.modules.setdefault("aurelius.training", _src_training)
if "src.training.gradient_surgery" in sys.modules:
    sys.modules.setdefault(
        "aurelius.training.gradient_surgery",
        sys.modules["src.training.gradient_surgery"],
    )

from .agent_registry import (
    AGENT_REGISTRY,
    AGENTS_BY_CATEGORY,
    ALL_AGENTS,
    CODING_AGENT,
    COMMUNICATION_AGENT,
    CREATIVE_AGENT,
    DEVOPS_AGENT,
    RESEARCH_AGENT,
    SCHEDULING_AGENT,
    SQL_AGENT,
    TUTOR_AGENT,
)
from .api_registry import get_agent_for_task, get_skills_for_agent, register_api_routes
from .plugin_system import BUILTIN_PLUGINS, PLUGIN_MANAGER, PluginManager
from .skills_registry import ALL_SKILLS, SKILL_REGISTRY, SKILLS_BY_CATEGORY

__all__ = [
    "ALL_AGENTS",
    "AGENT_REGISTRY",
    "AGENTS_BY_CATEGORY",
    "CODING_AGENT",
    "COMMUNICATION_AGENT",
    "CREATIVE_AGENT",
    "DEVOPS_AGENT",
    "RESEARCH_AGENT",
    "SCHEDULING_AGENT",
    "SQL_AGENT",
    "TUTOR_AGENT",
    "ALL_SKILLS",
    "SKILL_REGISTRY",
    "SKILLS_BY_CATEGORY",
    "BUILTIN_PLUGINS",
    "PLUGIN_MANAGER",
    "PluginManager",
    "get_agent_for_task",
    "get_skills_for_agent",
    "register_api_routes",
]
