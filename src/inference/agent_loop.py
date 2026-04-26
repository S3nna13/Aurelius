"""Agentic loop: ReAct-style reasoning-action-observation cycle with tool use and self-reflection."""  # noqa: E501

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Configuration for the ReAct agent loop."""

    max_steps: int = 10
    max_tokens_per_step: int = 64
    temperature: float = 0.7
    stop_token: str = "<|end|>"
    reflection_enabled: bool = True
    thought_prefix: str = "Thought:"
    action_prefix: str = "Action:"
    observation_prefix: str = "Observation:"


@dataclass
class AgentStep:
    """Record of one ReAct reasoning step."""

    step_num: int
    thought: str
    action: str | None
    observation: str | None
    is_final: bool


# ---------------------------------------------------------------------------
# Agent memory
# ---------------------------------------------------------------------------


class AgentMemory:
    """Rolling memory of agent steps."""

    def __init__(self, max_steps: int = 10) -> None:
        self.max_steps = max_steps
        self.steps: list[AgentStep] = []

    def add_step(self, step: AgentStep) -> None:
        """Append a step, trimming oldest if over max_steps."""
        self.steps.append(step)
        if len(self.steps) > self.max_steps:
            self.steps = self.steps[-self.max_steps :]

    def format_history(self) -> str:
        """Format all steps as human-readable ReAct history."""
        parts: list[str] = []
        for step in self.steps:
            parts.append(f"Thought: {step.thought}")
            if step.action is not None:
                parts.append(f"Action: {step.action}")
            if step.observation is not None:
                parts.append(f"Observation: {step.observation}")
        return "\n".join(parts)

    def get_last_n(self, n: int) -> list[AgentStep]:
        """Return the last n steps."""
        return self.steps[-n:] if n > 0 else []

    def clear(self) -> None:
        """Reset steps to empty list."""
        self.steps = []


# ---------------------------------------------------------------------------
# ReAct parsing and prompt formatting
# ---------------------------------------------------------------------------


def parse_react_output(text: str, config: AgentConfig) -> tuple[str, str | None]:
    """Parse model output into (thought, action_or_none).

    Extracts thought after thought_prefix and action after action_prefix.
    If no action found, returns (thought, None) — treat as final answer.
    """
    thought = ""
    action: str | None = None

    # Extract thought
    thought_pattern = (
        re.escape(config.thought_prefix) + r"\s*(.*?)(?=" + re.escape(config.action_prefix) + r"|$)"
    )
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
    else:
        # Fall back: everything before Action: or the whole text
        if config.action_prefix in text:
            thought = text[: text.index(config.action_prefix)].strip()
        else:
            thought = text.strip()

    # Extract action
    action_pattern = re.escape(config.action_prefix) + r"\s*(.*?)$"
    action_match = re.search(action_pattern, text, re.DOTALL)
    if action_match:
        action_text = action_match.group(1).strip()
        if action_text:
            action = action_text

    return thought, action


def format_react_prompt(
    task: str,
    history: str,
    available_tools: list[str],
    config: AgentConfig,
) -> str:
    """Build a ReAct-style prompt.

    Format::

        Task: {task}
        Available tools: {tools}
        {history}
        Thought:
    """
    tools_str = ", ".join(available_tools) if available_tools else "none"
    parts = [
        f"Task: {task}",
        f"Available tools: {tools_str}",
    ]
    if history:
        parts.append(history)
    parts.append("Thought:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


class SimpleToolExecutor:
    """Executes tool calls using simple deterministic registered callables."""

    def __init__(self) -> None:
        self.tools: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        """Register a tool by name."""
        self.tools[name] = fn

    def execute(self, action_str: str) -> str:
        """Parse and execute action string of form: tool_name(arg1, arg2, ...)

        Returns string result or error message (never raises).
        """
        action_str = action_str.strip()
        match = re.match(r"^(\w+)\((.*)\)$", action_str, re.DOTALL)
        if not match:
            return f"Error: could not parse action '{action_str}'"

        tool_name = match.group(1)
        args_str = match.group(2).strip()

        if tool_name not in self.tools:
            return f"Error: tool '{tool_name}' not found"

        # Parse args: split by comma, strip whitespace and quotes
        if args_str:
            raw_args = [a.strip().strip("'\"") for a in args_str.split(",")]
        else:
            raw_args = []

        try:
            result = self.tools[tool_name](*raw_args)
            return str(result)
        except Exception as exc:  # noqa: BLE001
            return f"Error executing tool '{tool_name}': {exc}"

    def list_tools(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self.tools.keys())


# ---------------------------------------------------------------------------
# ReAct agent
# ---------------------------------------------------------------------------


class ReActAgent:
    """Full ReAct-style agent: Thought → Action → Observation loop."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encode: Callable,
        tokenizer_decode: Callable,
        tool_executor: SimpleToolExecutor,
        config: AgentConfig,
    ) -> None:
        self.model = model
        self.encode = tokenizer_encode
        self.decode = tokenizer_decode
        self.tool_executor = tool_executor
        self.config = config
        self.memory = AgentMemory(max_steps=config.max_steps)

    def _generate_step(self, prompt: str) -> str:
        """Greedy decode up to max_tokens_per_step tokens."""
        input_ids = self.encode(prompt)
        current_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        generated_ids: list[int] = []

        with torch.no_grad():
            for _ in range(self.config.max_tokens_per_step):
                output = self.model(current_ids)
                logits = output[1]  # (1, S, V)
                next_logits = logits[0, -1, :]  # (V,)

                if self.config.temperature > 0.0:
                    probs = torch.softmax(next_logits / self.config.temperature, dim=-1)
                    next_token = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_token = int(torch.argmax(next_logits).item())

                generated_ids.append(next_token)
                current_ids = torch.cat(
                    [current_ids, torch.tensor([[next_token]], dtype=torch.long)],
                    dim=1,
                )

                generated_text = self.decode(generated_ids)
                if self.config.stop_token in generated_text:
                    break

        return self.decode(generated_ids)

    def run(self, task: str) -> tuple[str, list[AgentStep]]:
        """Execute ReAct loop up to max_steps.

        Returns (final_answer, steps).
        """
        self.memory.clear()
        available_tools = self.tool_executor.list_tools()

        final_answer = ""
        steps: list[AgentStep] = []

        for step_idx in range(self.config.max_steps):
            history = self.memory.format_history()
            prompt = format_react_prompt(task, history, available_tools, self.config)

            generated = self._generate_step(prompt)
            thought, action = parse_react_output(generated, self.config)

            is_final = False
            observation: str | None = None

            if action is None or action.startswith("Final Answer"):
                is_final = True
                if action is not None and action.startswith("Final Answer"):
                    # Extract the answer after "Final Answer:"
                    final_answer = action[len("Final Answer") :].lstrip(":").strip()
                else:
                    final_answer = thought
            else:
                observation = self.tool_executor.execute(action)

            agent_step = AgentStep(
                step_num=step_idx,
                thought=thought,
                action=action,
                observation=observation,
                is_final=is_final,
            )
            self.memory.add_step(agent_step)
            steps.append(agent_step)

            if is_final:
                break

        # Mark last step as final if max_steps exhausted
        if steps and not steps[-1].is_final:
            last = steps[-1]
            steps[-1] = AgentStep(
                step_num=last.step_num,
                thought=last.thought,
                action=last.action,
                observation=last.observation,
                is_final=True,
            )
            if not final_answer:
                final_answer = last.thought

        return final_answer, steps

    def reset(self) -> None:
        """Clear agent memory."""
        self.memory.clear()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_agent_metrics(steps: list[AgentStep]) -> dict[str, float]:
    """Compute metrics over a list of agent steps.

    Returns:
        n_steps: total number of steps
        tool_use_rate: fraction of steps with action != None
        completion_rate: 1.0 if any step is_final else 0.0
    """
    n = len(steps)
    if n == 0:
        return {"n_steps": 0.0, "tool_use_rate": 0.0, "completion_rate": 0.0}

    tool_steps = sum(1 for s in steps if s.action is not None)
    is_complete = any(s.is_final for s in steps)

    return {
        "n_steps": float(n),
        "tool_use_rate": tool_steps / n,
        "completion_rate": 1.0 if is_complete else 0.0,
    }
