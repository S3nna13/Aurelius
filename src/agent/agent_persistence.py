"""Agent persistence — save/load agent state, skills, tools, memory."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .agent_memory import AgentMemory, MemoryEntry
from .agent_runtime import AgentMessage, AgentRuntime, AgentSpec, AgentStatus
from .react_loop import AgentTrace
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
                    "content": m.content,
                    "msg_type": m.msg_type,
                    "metadata": m.metadata,
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
        runtime.agents.clear()
        runtime.mailbox.clear()
        for a in data.get("agents", []):
            status_value = str(a.get("status", AgentStatus.IDLE.value))
            try:
                status = AgentStatus(status_value)
            except ValueError:
                status = AgentStatus.IDLE
            spec = AgentSpec(
                id=str(a.get("id", "")),
                name=str(a.get("name", "")),
                role=str(a.get("role", "general")),
                system_prompt=str(a.get("system_prompt", "You are a helpful AI agent.")),
                tools=list(a.get("tools", [])),
                max_iterations=int(a.get("max_iterations", 20)),
                status=status,
            )
            runtime.agents[spec.id] = spec
        for m in data.get("messages", []):
            if not isinstance(m, dict):
                continue
            metadata = m.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            runtime.mailbox.append(
                AgentMessage(
                    sender=str(m.get("sender", "")),
                    recipient=str(m.get("recipient", "")),
                    content=str(m.get("content", "")),
                    msg_type=str(m.get("msg_type", "text")),
                    metadata=metadata,
                )
            )
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

    def save_trace(self, trace: AgentTrace, agent_id: str = "default") -> str:
        steps_path = self.path / "traces" / agent_id
        steps_path.mkdir(parents=True, exist_ok=True)
        ts = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{time.time_ns()}"
        fp = steps_path / f"{ts}.json"
        fp.write_text(
            json.dumps(
                {
                    "agent_id": agent_id,
                    "timestamp": ts,
                    "task": trace.task,
                    "system_prompt": trace.system_prompt,
                    "status": trace.status,
                    "steps_used": trace.steps_used,
                    "final_answer": trace.final_answer,
                    "steps": [
                        {
                            "role": s.role,
                            "content": s.content[:500],
                            "tool_name": s.tool_name,
                            "tool_input": s.tool_input,
                            "tool_output": s.tool_output[:500] if s.tool_output else None,
                            "error": s.error,
                        }
                        for s in trace.steps
                    ],
                },
                indent=2,
            )
        )
        return str(fp)

    def list_traces(self, agent_id: str = "default", limit: int = 10) -> list[dict]:
        steps_path = self.path / "traces" / agent_id
        if not steps_path.exists():
            return []
        files = sorted(steps_path.iterdir(), reverse=True)[:limit]
        traces = []
        for f in files:
            try:
                traces.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return traces

    def load_latest_trace(self, agent_id: str = "default") -> AgentTrace | None:
        steps_path = self.path / "traces" / agent_id
        if not steps_path.exists():
            return None
        files = sorted(steps_path.iterdir(), reverse=True)
        if not files:
            return None
        try:
            data = json.loads(files[0].read_text())
            from .react_loop import AgentStep

            trace = AgentTrace(
                task=str(data.get("task") or ""),
                system_prompt=str(data.get("system_prompt") or ""),
                status=data.get("status", "no_answer"),
                steps_used=data.get("steps_used", 0),
                final_answer=data.get("final_answer"),
            )
            for s_data in data.get("steps", []):
                trace.steps.append(
                    AgentStep(
                        role=s_data.get("role", ""),
                        content=s_data.get("content", ""),
                        tool_name=s_data.get("tool_name"),
                        tool_input=s_data.get("tool_input"),
                        tool_output=s_data.get("tool_output"),
                        error=s_data.get("error"),
                    )
                )
            return trace
        except (json.JSONDecodeError, OSError, KeyError):
            return None
