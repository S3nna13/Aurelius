"""NeuralBrain — Aurelius's core thinking/planning/acting/reasoning loop.

Ported from Aurelius's aurelius/neural_brain.py.

The NeuralBrain is the central cognitive loop:
    encode → retrieve → plan → reason → act → verify → reflect → update memory

All functions are injectable via callable injection (Strategy Pattern).
Defaults are provided for every step, making the system immediately usable
while allowing full customization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.memory import MemoryManager


class BrainState(Enum):
    IDLE = "idle"
    ENCODING = "encoding"
    RETRIEVING = "retrieving"
    PLANNING = "planning"
    REASONING = "reasoning"
    ACTING = "acting"
    VERIFYING = "verifying"
    REFLECTING = "reflecting"
    UPDATING_MEMORY = "updating_memory"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class BrainContext:
    state: BrainState = BrainState.IDLE
    input_text: str = ""
    encoded_input: Any = None
    retrieved_context: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    verification_result: str = ""
    reflection: str = ""
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class NeuralBrain:
    """Core cognitive loop with injectable functions for each step.

    Each step can be overridden by passing a callable:
        brain = NeuralBrain(plan_fn=my_planner, reason_fn=my_reasoner)

    Default implementations are simple pass-through stubs that make the
    system immediately functional.
    """

    def __init__(
        self,
        encode_fn: Callable[[str], Any] | None = None,
        retrieve_fn: Callable[[Any], list[str]] | None = None,
        plan_fn: Callable[[str, list[str]], list[str]] | None = None,
        reason_fn: Callable[[str, list[str]], list[str]] | None = None,
        act_fn: Callable[[list[str], list[str]], list[dict[str, Any]]] | None = None,
        verify_fn: Callable[[str, str], str] | None = None,
        reflect_fn: Callable[[str, str, str], str] | None = None,
        memory_fn: Callable[[str, str], None] | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self.memory_manager = memory_manager
        self.encode_fn = encode_fn or self._default_encode
        self.retrieve_fn = retrieve_fn or self._build_memory_retrieve(memory_manager)
        self.plan_fn = plan_fn or self._default_plan
        self.reason_fn = reason_fn or self._default_reason
        self.act_fn = act_fn or self._default_act
        self.verify_fn = verify_fn or self._default_verify
        self.reflect_fn = reflect_fn or self._default_reflect
        self.memory_fn = memory_fn or self._build_memory_update(memory_manager)

    def run(self, input_text: str) -> BrainContext:
        """Execute one full cognitive cycle."""
        ctx = BrainContext(input_text=input_text)

        try:
            ctx.state = BrainState.ENCODING
            ctx.encoded_input = self.encode_fn(input_text)

            ctx.state = BrainState.RETRIEVING
            ctx.retrieved_context = self.retrieve_fn(ctx.encoded_input)

            ctx.state = BrainState.PLANNING
            ctx.plan = self.plan_fn(input_text, ctx.retrieved_context)

            ctx.state = BrainState.REASONING
            ctx.reasoning_steps = self.reason_fn(input_text, ctx.plan)

            ctx.state = BrainState.ACTING
            ctx.actions = self.act_fn(ctx.reasoning_steps, ctx.plan)

            ctx.state = BrainState.VERIFYING
            ctx.verification_result = self.verify_fn(input_text, str(ctx.actions))

            ctx.state = BrainState.REFLECTING
            ctx.reflection = self.reflect_fn(input_text, str(ctx.actions), ctx.verification_result)

            ctx.state = BrainState.UPDATING_MEMORY
            self.memory_fn(input_text, ctx.reflection)

            ctx.state = BrainState.COMPLETED
            ctx.output = ctx.actions[-1].get("result", "") if ctx.actions else ""

        except Exception as e:
            ctx.state = BrainState.ERROR
            ctx.error = str(e)

        return ctx

    # --- Default implementations ---

    @staticmethod
    def _default_encode(text: str) -> Any:
        return text

    @staticmethod
    def _default_retrieve(encoded: Any) -> list[str]:
        return []

    @staticmethod
    def _build_memory_retrieve(
        memory_manager: MemoryManager | None,
    ) -> Callable[[Any], list[str]]:
        if memory_manager is None:
            return NeuralBrain._default_retrieve

        def _retrieve(encoded: Any) -> list[str]:
            query = encoded if isinstance(encoded, str) else str(encoded)
            return memory_manager.contextualize(query, top_k=5)

        return _retrieve

    @staticmethod
    def _default_plan(task: str, context: list[str]) -> list[str]:
        return [f"Step 1: Process the request: {task[:50]}..."]

    @staticmethod
    def _default_reason(task: str, plan: list[str]) -> list[str]:
        return [f"Reasoning about: {task[:50]}..."]

    @staticmethod
    def _default_act(reasoning: list[str], plan: list[str]) -> list[dict[str, Any]]:
        return [{"action": "respond", "result": reasoning[-1] if reasoning else "No result"}]

    @staticmethod
    def _default_verify(task: str, actions: str) -> str:
        return "verification_passed"

    @staticmethod
    def _default_reflect(task: str, actions: str, verification: str) -> str:
        return "Reflection: completed task within expected parameters"

    @staticmethod
    def _default_memory(task: str, reflection: str) -> None:
        pass

    @staticmethod
    def _build_memory_update(
        memory_manager: MemoryManager | None,
    ) -> Callable[[str, str], None]:
        if memory_manager is None:
            return NeuralBrain._default_memory

        def _remember(task: str, reflection: str) -> None:
            memory_manager.remember(
                f"Task: {task}\nReflection: {reflection}",
                importance=0.7,
                tags=["neural_brain", "reflection"],
                context="brain_cycle",
                working_key="last_reflection",
            )

        return _remember
