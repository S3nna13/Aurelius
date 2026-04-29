"""Complete plugin system for Aurelius.

Plugins extend agent capabilities with new tools, skills, and integrations.
Every plugin has a manifest, can be enabled/disabled, and provides
zero or more tools and skills to the agent runtime.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PluginManifest:
    id: str
    name: str
    version: str
    description: str
    author: str = "unknown"
    requires: list[str] = field(default_factory=list)
    provides_tools: list[str] = field(default_factory=list)
    provides_skills: list[str] = field(default_factory=list)
    entry_point: str = ""


@dataclass
class Plugin:
    manifest: PluginManifest
    enabled: bool = True
    tools: dict[str, Callable] = field(default_factory=dict)
    loaded: bool = False


# ── Built-in Plugin Definitions ────────────────────────────────────────

BUILTIN_PLUGINS: list[PluginManifest] = [
    PluginManifest(
        id="filesystem", name="Filesystem Tools", version="1.0.0",
        description="Read, write, and manage files on the local filesystem.",
        author="Aurelius", provides_tools=["read_file", "write_file", "list_dir", "search_files"],
    ),
    PluginManifest(
        id="web", name="Web Tools", version="1.0.0",
        description="Search the web, fetch URLs, and extract content.",
        author="Aurelius", provides_tools=["search_web", "fetch_url", "extract_content"],
    ),
    PluginManifest(
        id="database", name="Database Tools", version="1.0.0",
        description="Query databases, explore schemas, and manage data.",
        author="Aurelius", provides_tools=["query_db", "describe_schema", "list_tables"],
    ),
    PluginManifest(
        id="communication", name="Communication Tools", version="1.0.0",
        description="Send emails, messages, and manage notifications.",
        author="Aurelius", provides_tools=["send_email", "send_message", "create_notification"],
    ),
    PluginManifest(
        id="system", name="System Tools", version="1.0.0",
        description="Run commands, monitor system, and manage processes.",
        author="Aurelius", requires=["security"],
        provides_tools=["run_command", "system_info", "process_list"],
    ),
    PluginManifest(
        id="code", name="Code Tools", version="1.0.0",
        description="Analyze, compile, and run code in multiple languages.",
        author="Aurelius", provides_tools=["run_code", "lint_code", "format_code"],
    ),
    PluginManifest(
        id="ai", name="AI Tools", version="1.0.0",
        description="Generate text, embeddings, and interact with LLMs.",
        author="Aurelius",
        provides_tools=["generate_text", "get_embeddings", "classify_text"],
        provides_skills=["prompt_engineering", "tool_creation"],
    ),
    PluginManifest(
        id="analytics", name="Analytics Tools", version="1.0.0",
        description="Analyze data, create charts, and generate reports.",
        author="Aurelius",
        provides_tools=["analyze_data", "create_chart", "generate_report"],
        provides_skills=["data_analysis", "data_visualization"],
    ),
    PluginManifest(
        id="security", name="Security Tools", version="1.0.0",
        description="Scan for vulnerabilities, audit logs, check compliance.",
        author="Aurelius",
        provides_tools=["security_scan", "audit_log", "compliance_check"],
        provides_skills=["security_scanning", "incident_response"],
    ),
    PluginManifest(
        id="devops", name="DevOps Tools", version="1.0.0",
        description="Deploy services, monitor infrastructure, manage containers.",
        author="Aurelius", requires=["system"],
        provides_tools=["deploy", "monitor", "container_exec"],
        provides_skills=["deployment", "infrastructure_monitoring"],
    ),
    PluginManifest(
        id="productivity", name="Productivity Tools", version="1.0.0",
        description="Manage tasks, calendar, notes, and projects.",
        author="Aurelius",
        provides_tools=["create_task", "schedule_event", "create_note"],
        provides_skills=["scheduling", "task_management", "note_taking"],
    ),
    PluginManifest(
        id="education", name="Education Tools", version="1.0.0",
        description="Generate quizzes, explain concepts, assess understanding.",
        author="Aurelius",
        provides_tools=["generate_quiz", "explain_concept", "assess_knowledge"],
        provides_skills=["teaching", "quiz_generation", "language_learning"],
    ),
]

PLUGIN_MANIFESTS: dict[str, PluginManifest] = {p.id: p for p in BUILTIN_PLUGINS}


class PluginManager:
    """Manages plugin lifecycle — load, enable, disable, and provide tools."""

    def __init__(self, plugins_dir: str | Path | None = None):
        self.plugins: dict[str, Plugin] = {}
        self.plugins_dir = Path(plugins_dir) if plugins_dir else Path("plugins")
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self._load_builtins()

    def _load_builtins(self) -> None:
        for manifest in BUILTIN_PLUGINS:
            plugin = Plugin(manifest=manifest)
            self.plugins[manifest.id] = plugin
        logger.info(f"Loaded {len(BUILTIN_PLUGINS)} built-in plugins")

    def get_plugin(self, plugin_id: str) -> Plugin | None:
        return self.plugins.get(plugin_id)

    def enable_plugin(self, plugin_id: str) -> bool:
        plugin = self.plugins.get(plugin_id)
        if plugin and not plugin.enabled:
            plugin.enabled = True
            logger.info(f"Plugin enabled: {plugin_id}")
            return True
        return False

    def disable_plugin(self, plugin_id: str) -> bool:
        plugin = self.plugins.get(plugin_id)
        if plugin and plugin.enabled:
            plugin.enabled = False
            logger.info(f"Plugin disabled: {plugin_id}")
            return True
        return False

    def list_plugins(self, enabled_only: bool = False) -> list[Plugin]:
        if enabled_only:
            return [p for p in self.plugins.values() if p.enabled]
        return list(self.plugins.values())

    def get_tools_for_agent(self, agent_id: str) -> list[str]:
        from aurelius.agent_registry import AGENT_REGISTRY
        agent = AGENT_REGISTRY.get(agent_id)
        if not agent:
            return []
        return agent.default_tools

    def get_all_tools(self) -> list[dict[str, Any]]:
        tools = []
        for plugin in self.plugins.values():
            if plugin.enabled:
                for tool_id in plugin.manifest.provides_tools:
                    tools.append({
                        "id": tool_id,
                        "plugin": plugin.manifest.id,
                        "plugin_name": plugin.manifest.name,
                    })
        return tools

    def get_skills_from_plugins(self) -> list[str]:
        skills = set()
        for plugin in self.plugins.values():
            if plugin.enabled:
                for skill in plugin.manifest.provides_skills:
                    skills.add(skill)
        return sorted(skills)

    def get_plugin_summary(self) -> dict[str, Any]:
        enabled = sum(1 for p in self.plugins.values() if p.enabled)
        total_tools = sum(len(p.manifest.provides_tools) for p in self.plugins.values() if p.enabled)
        total_skills = sum(len(p.manifest.provides_skills) for p in self.plugins.values() if p.enabled)
        return {
            "total_plugins": len(self.plugins),
            "enabled_plugins": enabled,
            "total_tools": total_tools,
            "total_skills": total_skills,
            "plugins": [{
                "id": p.manifest.id, "name": p.manifest.name,
                "version": p.manifest.version, "enabled": p.enabled,
                "tools": p.manifest.provides_tools,
                "skills": p.manifest.provides_skills,
            } for p in self.plugins.values()],
        }


# Global plugin manager singleton
PLUGIN_MANAGER = PluginManager()
