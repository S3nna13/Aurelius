"""Persistent agent — ReAct loop + memory + trace persistence."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .agent_memory import AgentMemory
from .agent_persistence import AgentPersistence
from .react_loop import AgentTrace, ReActLoop


class PersistentAgent:
    """ReAct loop with memory recall and trace persistence.

    Wraps ReActLoop with:
    * Prior memory loaded and injected into the system prompt
    * Full trace saved to disk after each run
    * AgentMemory updated with outcomes for future recall
    """

    def __init__(
        self,
        agent_id: str = "default",
        generate_fn: Callable[[list[dict]], str] | None = None,
        tool_registry: dict[str, Callable[..., Any]] | None = None,
        memory: AgentMemory | None = None,
        persist_path: str = ".aurelius/agents",
        max_steps: int = 8,
        max_tool_seconds: float = 5.0,
    ) -> None:
        self.agent_id = agent_id
        self.memory = memory or AgentMemory()
        self.persistence = AgentPersistence(persist_path)
        self._generate_fn = generate_fn
        self._tool_registry = tool_registry or {}
        self._max_steps = max_steps
        self._max_tool_seconds = max_tool_seconds
        self._loop: ReActLoop | None = None
        self.last_trace: AgentTrace | None = None

        self.persistence.load_memory(self.memory, f"memory_{agent_id}")

    @property
    def loop(self) -> ReActLoop:
        if self._loop is None:
            if self._generate_fn is None:
                raise RuntimeError("generate_fn must be set before running")
            self._loop = ReActLoop(
                generate_fn=self._generate_fn,
                tool_registry=self._tool_registry,
                max_steps=self._max_steps,
                max_tool_seconds=self._max_tool_seconds,
            )
        return self._loop

    def _build_memory_context(self) -> str:
        recent = self.memory.recall("", top_k=5, memory_type="episodic")
        semantic = self.memory.recall("", top_k=3, memory_type="semantic")
        parts: list[str] = []
        if recent:
            parts.append("Recent context:")
            for e in recent:
                parts.append(f"  - {e.content[:200]}")
        if semantic:
            parts.append("Learned knowledge:")
            for e in semantic:
                parts.append(f"  - {e.content[:200]}")
        return "\n".join(parts)

    def run(self, task: str, system_prompt: str = "") -> AgentTrace:
        ctx = self._build_memory_context()
        if ctx:
            full_prompt = (
                f"{system_prompt}\n\nMemory context:\n{ctx}"
                if system_prompt
                else f"Memory context:\n{ctx}"
            )
        else:
            full_prompt = system_prompt

        trace = self.loop.run(task=task, system_prompt=full_prompt)
        self.last_trace = trace

        self.persistence.save_trace(trace, self.agent_id)

        if trace.final_answer:
            self.memory.remember(
                content=f"[{self.agent_id}] Task: {task[:100]} -> {trace.final_answer[:200]}",
                memory_type="episodic",
                importance=0.7,
                tags=[self.agent_id, "task"],
            )
        if trace.status == "error":
            errors = [s.error for s in trace.steps if s.error]
            if errors:
                self.memory.remember(
                    content=f"[{self.agent_id}] Error pattern: {'; '.join(errors[:3])}",
                    memory_type="semantic",
                    importance=0.5,
                    tags=[self.agent_id, "error"],
                )

        self.persistence.save_memory(self.memory, f"memory_{self.agent_id}")
        return trace

    def update_generate_fn(self, fn: Callable[[list[dict]], str]) -> None:
        self._generate_fn = fn
        self._loop = None
