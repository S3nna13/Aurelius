"""Neural Brain — core thinking/planning/acting/reasoning loop.

Architecture:
  Input → Encode → Retrieve → Plan → Reason → Act → Verify → Reflect → Update Memory → Repeat

Modules:
  - InputEncoder: encode user input into internal representation
  - MemoryRetriever: fetch relevant context from all memory types
  - WorkingMemory: active scratchpad for current reasoning
  - Planner: decompose task into subtasks
  - Reasoner: chain-of-thought, step-by-step reasoning
  - ToolController: choose and call tools
  - AgentRouter: route subtasks to specialized agents
  - CodeMaker: generate and verify code
  - Verifier: check outputs for correctness
  - Reflector: learn from successes and failures
  - ExecutiveController: orchestrate the full loop
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class BrainState(StrEnum):
    IDLE = "idle"
    ENCODING = "encoding"
    RETRIEVING = "retrieving"
    PLANNING = "planning"
    REASONING = "reasoning"
    ACTING = "acting"
    VERIFYING = "verifying"
    REFLECTING = "reflecting"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Thought:
    content: str
    type: str = "thought"
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrainContext:
    input: str = ""
    encoded: dict[str, Any] = field(default_factory=dict)
    memories: list[dict[str, Any]] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    reasoning: list[Thought] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    verifications: list[dict[str, Any]] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    output: str = ""
    state: BrainState = BrainState.IDLE
    iteration: int = 0
    max_iterations: int = 10


class NeuralBrain:
    """Full neural brain loop — input to output with reasoning, tools, agents.

    The brain runs a loop:
      encode → retrieve → plan → reason → act → verify → reflect → update memory
    """

    def __init__(
        self,
        plan_fn: Callable[[str], list[str]] | None = None,
        reason_fn: Callable[[str], str] | None = None,
        tool_fn: Callable[[str, dict], str] | None = None,
        agent_fn: Callable[[str], str] | None = None,
        verify_fn: Callable[[str], dict[str, Any]] | None = None,
        reflect_fn: Callable[[str, str], str] | None = None,
        memory_fn: Callable[[str, dict], None] | None = None,
    ):
        self.plan_fn = plan_fn or self._default_plan
        self.reason_fn = reason_fn or self._default_reason
        self.tool_fn = tool_fn or self._default_tool
        self.agent_fn = agent_fn or self._default_agent
        self.verify_fn = verify_fn or self._default_verify
        self.reflect_fn = reflect_fn or self._default_reflect
        self.memory_fn = memory_fn or self._default_memory
        self._stats: dict[str, int] = {"runs": 0, "iterations": 0, "tools": 0, "agents": 0}

    def run(self, user_input: str, context: BrainContext | None = None) -> BrainContext:
        ctx = context or BrainContext(input=user_input)
        ctx.input = user_input
        ctx.state = BrainState.ENCODING
        self._stats["runs"] += 1

        # 1. Encode
        ctx.encoded = self._encode(user_input)

        # 2. Retrieve memories
        ctx.state = BrainState.RETRIEVING
        ctx.memories = self._retrieve(user_input)

        # 3-8. Main loop
        for i in range(ctx.max_iterations):
            ctx.iteration = i
            self._stats["iterations"] += 1

            # 3. Plan
            ctx.state = BrainState.PLANNING
            ctx.plan = self.plan_fn(user_input)

            # 4. Reason
            ctx.state = BrainState.REASONING
            for step in ctx.plan:
                thought = self.reason_fn(f"{user_input}\nStep: {step}")
                ctx.reasoning.append(Thought(content=thought, type="reasoning"))

            # 5. Act (tools and agents)
            ctx.state = BrainState.ACTING
            for step in ctx.plan:
                if self._needs_tool(step):
                    tool_result = self.tool_fn(step, {"input": user_input})
                    ctx.actions.append({"step": step, "tool": True, "result": tool_result})
                    self._stats["tools"] += 1
                if self._needs_agent(step):
                    agent_result = self.agent_fn(step)
                    ctx.actions.append({"step": step, "agent": True, "result": agent_result})
                    self._stats["agents"] += 1

            # 6. Verify
            ctx.state = BrainState.VERIFYING
            combined = "\n".join([t.content for t in ctx.reasoning] + [str(a.get("result", "")) for a in ctx.actions])
            verification = self.verify_fn(combined)
            ctx.verifications.append(verification)

            if verification.get("passed", False):
                break

            # 7. Reflect
            ctx.state = BrainState.REFLECTING
            reflection = self.reflect_fn(user_input, verification.get("feedback", ""))
            ctx.reflections.append(reflection)

            # 8. Update memory
            self.memory_fn(f"reflection_{i}", {"reflection": reflection, "verification": verification})

        ctx.state = BrainState.COMPLETE
        ctx.output = self._synthesize(ctx)
        return ctx

    def _encode(self, text: str) -> dict[str, Any]:
        return {"original": text, "length": len(text), "tokens_est": len(text.split())}

    def _retrieve(self, text: str) -> list[dict[str, Any]]:
        return []

    def _needs_tool(self, step: str) -> bool:
        keywords = ["search", "fetch", "compute", "calculate", "run", "execute", "read", "write"]
        return any(k in step.lower() for k in keywords)

    def _needs_agent(self, step: str) -> bool:
        keywords = ["agent", "delegate", "research", "code", "analyze"]
        return any(k in step.lower() for k in keywords)

    def _synthesize(self, ctx: BrainContext) -> str:
        parts = []
        for t in ctx.reasoning:
            parts.append(t.content)
        for a in ctx.actions:
            parts.append(str(a.get("result", "")))
        return "\n".join(parts)

    def get_stats(self) -> dict[str, int]:
        return dict(self._stats)

    @staticmethod
    def _default_plan(task: str) -> list[str]:
        return [f"Understand: {task[:50]}", f"Break down: {task[:50]}", f"Solve: {task[:50]}", "Verify solution"]

    @staticmethod
    def _default_reason(prompt: str) -> str:
        return f"Reasoning step: analyzing {prompt[:50]}..."

    @staticmethod
    def _default_tool(name: str, args: dict) -> str:
        return f"<tool_result>{name} executed</tool_result>"

    @staticmethod
    def _default_agent(task: str) -> str:
        return f"<agent_result>{task[:50]} processed</agent_result>"

    @staticmethod
    def _default_verify(output: str) -> dict[str, Any]:
        return {"passed": True, "feedback": "Output verified", "confidence": 0.9}

    @staticmethod
    def _default_reflect(input_str: str, feedback: str) -> str:
        return f"Reflection: {feedback[:100]}"

    @staticmethod
    def _default_memory(key: str, value: dict) -> None:
        pass
