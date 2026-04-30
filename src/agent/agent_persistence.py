"""Agent persistence — save/load agent state, skills, tools, memory."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .agent_memory import AgentMemory, MemoryEntry
from .agent_runtime import AgentRuntime, AgentSpec
from .skill_registry import SkillRegistry


class AgentPersistence:
    """Persists and restores agent state to/from disk."""

    def __init__(self, base_path: str = ".aurelius/agents"):
        self.path = Path(base_path)
        self.path.mkdir(parents=True, exist_ok=True)

    def save_agents(self, runtime: AgentRuntime, name: str = "default") -> str:
        data = {
            "agents": [
                {
                    "id": a.id,
                    "name": a.name,
                    "role": a.role,
                    "system_prompt": a.system_prompt,
                    "tools": a.tools,
                    "max_iterations": a.max_iterations,
                    "status": a.status.value,
                }
                for a in runtime.agents.values()
            ],
            "messages": [
                {
                    "sender": m.sender,
                    "recipient": m.recipient,
                    "content": m.content[:200],
                    "msg_type": m.msg_type,
                }
                for m in runtime.mailbox
            ],
        }
        fp = self.path / f"{name}.json"
        fp.write_text(json.dumps(data, indent=2))
        return str(fp)

    def load_agents(self, runtime: AgentRuntime, name: str = "default") -> bool:
        fp = self.path / f"{name}.json"
        if not fp.exists():
            return False
        data = json.loads(fp.read_text())
        for a in data.get("agents", []):
            spec = AgentSpec(
                id=a["id"],
                name=a["name"],
                role=a["role"],
                system_prompt=a["system_prompt"],
                tools=a["tools"],
                max_iterations=a["max_iterations"],
            )
            runtime.register_agent(spec)
        return True

    def save_skills(self, registry: SkillRegistry, name: str = "skills") -> str:
        data = registry.to_dict()
        fp = self.path / f"{name}.json"
        fp.write_text(json.dumps(data, indent=2))
        return str(fp)

    def load_skills(self, registry: SkillRegistry, name: str = "skills") -> bool:
        fp = self.path / f"{name}.json"
        if not fp.exists():
            return False
        data = json.loads(fp.read_text())
        registry.update_from_dict(data)
        return True

    def save_memory(self, memory: AgentMemory, name: str = "memory") -> str:
        data = {
            "episodic": [
                {
                    "content": e.content,
                    "memory_type": e.memory_type,
                    "timestamp": e.timestamp,
                    "importance": e.importance,
                    "tags": e.tags,
                }
                for e in memory.episodic[-100:]
            ],
            "semantic": {
                k: {
                    "content": v.content,
                    "memory_type": v.memory_type,
                    "timestamp": v.timestamp,
                    "importance": v.importance,
                    "tags": v.tags,
                }
                for k, v in list(memory.semantic.items())[:50]
            },
            "procedural": memory.procedural,
        }
        fp = self.path / f"{name}.json"
        fp.write_text(json.dumps(data, indent=2))
        return str(fp)

    def load_memory(self, memory: AgentMemory, name: str = "memory") -> bool:
        fp = self.path / f"{name}.json"
        if not fp.exists():
            return False

        data = json.loads(fp.read_text())
        memory.episodic.clear()
        memory.semantic.clear()
        memory.procedural.clear()

        for raw_entry in data.get("episodic", []):
            if not isinstance(raw_entry, dict):
                continue
            memory.episodic.append(
                MemoryEntry(
                    content=str(raw_entry.get("content", "")),
                    memory_type=str(raw_entry.get("memory_type", "episodic")),
                    timestamp=float(raw_entry.get("timestamp", time.time())),
                    importance=float(raw_entry.get("importance", 1.0)),
                    tags=list(raw_entry.get("tags", [])),
                )
            )

        semantic_data = data.get("semantic", {})
        if isinstance(semantic_data, dict):
            for key, raw_entry in semantic_data.items():
                if isinstance(raw_entry, dict):
                    memory.semantic[str(key)] = MemoryEntry(
                        content=str(raw_entry.get("content", "")),
                        memory_type=str(raw_entry.get("memory_type", "semantic")),
                        timestamp=float(raw_entry.get("timestamp", time.time())),
                        importance=float(raw_entry.get("importance", 1.0)),
                        tags=list(raw_entry.get("tags", [])),
                    )
                else:
                    memory.semantic[str(key)] = MemoryEntry(
                        content=str(raw_entry),
                        memory_type="semantic",
                    )

        procedural = data.get("procedural", {})
        if isinstance(procedural, dict):
            memory.procedural.update({str(k): str(v) for k, v in procedural.items()})

        return True
