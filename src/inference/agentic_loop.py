"""Agentic reasoning loop: ReAct / Toolformer-style iterative tool use.

The model outputs structured text with special tokens marking tool calls:
    <tool>{"name": "calculator", "args": {"expr": "2+2"}}</tool>

Loop: parse -> execute -> append result -> continue generation until
'Final Answer:' is produced or max_steps is reached.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass

import torch

# Maximum length of a calculator expression accepted by the built-in tool.
_MAX_EXPR_LEN = 1_000

# Operator table for the AST-walker arithmetic evaluator. No names, no calls,
# no attribute access — strictly numeric literals and these operators.
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _calc_walk(node: ast.AST) -> float | int:
    """Recursively compute a numeric AST without invoking the dynamic code
    evaluator. This is strictly safer than feeding the compiled tree to a
    code-evaluation primitive because no Python runtime code path is taken."""
    if isinstance(node, ast.Expression):
        return _calc_walk(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_calc_walk(node.left), _calc_walk(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_calc_walk(node.operand))
    raise ValueError(f"Disallowed node type: {type(node).__name__}")


def _safe_arith_eval(expression: str) -> float | int:
    """Evaluate an arithmetic expression via a strict AST walker.

    This replaces a prior use of the built-in dynamic code evaluator on
    user-supplied input (finding AUR-SEC-2026-0022). Only numeric literals
    and a small set of arithmetic operators are accepted."""
    if not expression or len(expression) > _MAX_EXPR_LEN:
        raise ValueError(f"expression rejected by size guard: len={len(expression)}")
    tree = ast.parse(expression.strip(), mode="eval")
    return _calc_walk(tree)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    """A named callable tool the agent can invoke."""

    name: str
    description: str
    fn: object  # callable (args: dict) -> str


@dataclass
class AgentStep:
    """Record of one reasoning step in the agent loop."""

    thought: str  # model's reasoning text
    tool_call: dict | None  # {'name': str, 'args': dict} or None
    tool_result: str | None
    final_answer: str | None
    is_final: bool  # True when model produced Final Answer


@dataclass
class AgentConfig:
    """Configuration for the agent loop."""

    max_steps: int = 10
    tool_call_pattern: str = r"<tool>(.*?)</tool>"
    final_answer_pattern: str = r"Final Answer:\s*(.*?)(?:\n|$)"
    max_new_tokens_per_step: int = 128
    temperature: float = 0.0  # greedy by default


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry of available tools.

    Tools are invoked by name with a JSON args dict.
    """

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def register(self, tool: Tool) -> None:
        """Add or replace a tool in the registry."""
        self._tools[tool.name] = tool

    def execute(self, name: str, args: dict) -> str:
        """Execute tool by name. Returns result string or error message."""
        if name not in self._tools:
            return f"Error: tool '{name}' not found in registry."
        try:
            result = self._tools[name].fn(args)
            return str(result)
        except Exception as exc:  # noqa: BLE001
            return f"Error executing tool '{name}': {exc}"

    def describe(self) -> str:
        """Return formatted tool descriptions for inclusion in prompts."""
        lines: list[str] = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


class AgentLoop:
    """Iterative reasoning loop using the model + tools.

    Each step:
    1. Generate tokens until <tool>...</tool> or "Final Answer:" or max_new_tokens
    2. If tool call: execute tool, append result, continue
    3. If final answer: return
    4. If neither: continue generating (treat as thought)

    Args:
        model: AureliusTransformer
        tokenizer_encode: callable str -> list[int]
        tokenizer_decode: callable list[int] -> str
        tool_registry: ToolRegistry
        config: AgentConfig
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer_encode,  # str -> list[int]
        tokenizer_decode,  # list[int] -> str
        tool_registry: ToolRegistry,
        config: AgentConfig | None = None,
    ) -> None:
        self.model = model
        self.encode = tokenizer_encode
        self.decode = tokenizer_decode
        self.registry = tool_registry
        self.config = config or AgentConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_until_stop(
        self,
        input_ids: list[int],
        stop_patterns: list[str],
    ) -> tuple[str, str | None]:
        """Generate tokens until one of stop_patterns is found or max_new_tokens.

        Returns (generated_text, matched_pattern_or_None).
        Uses simple greedy decoding (temperature=0 -> argmax).
        """
        cfg = self.config
        generated_ids: list[int] = []
        current_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # (1, S)

        with torch.no_grad():
            for _ in range(cfg.max_new_tokens_per_step):
                output = self.model(current_ids)  # returns (loss, logits) tuple
                logits = output[1]  # (1, S, V)
                next_logits = logits[0, -1, :]  # (V,)

                if cfg.temperature > 0.0:
                    probs = torch.softmax(next_logits / cfg.temperature, dim=-1)
                    next_token = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_token = int(torch.argmax(next_logits).item())

                generated_ids.append(next_token)
                current_ids = torch.cat(
                    [current_ids, torch.tensor([[next_token]], dtype=torch.long)],
                    dim=1,
                )

                generated_text = self.decode(generated_ids)

                # Check stop patterns after each token
                for pattern in stop_patterns:
                    if re.search(pattern, generated_text, re.DOTALL):
                        return generated_text, pattern

        return self.decode(generated_ids), None

    def _parse_tool_call(self, text: str) -> dict | None:
        """Extract JSON from <tool>...</tool> pattern.

        Returns {'name': str, 'args': dict} or None if parse fails.
        """
        match = re.search(self.config.tool_call_pattern, text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(1))
            if not isinstance(data, dict):
                return None
            return data
        except (json.JSONDecodeError, ValueError):
            return None

    def _parse_final_answer(self, text: str) -> str | None:
        """Extract text after 'Final Answer:' pattern. Returns None if not found."""
        match = re.search(self.config.final_answer_pattern, text, re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, prompt: str) -> list[AgentStep]:
        """Execute agent loop on prompt.

        Returns list of AgentStep objects documenting each reasoning step.
        """
        cfg = self.config
        context = prompt
        steps: list[AgentStep] = []

        stop_patterns = [cfg.tool_call_pattern, cfg.final_answer_pattern]

        for _step_idx in range(cfg.max_steps):
            input_ids = self.encode(context)
            generated_text, matched_pattern = self._generate_until_stop(input_ids, stop_patterns)

            tool_call: dict | None = None
            tool_result: str | None = None
            final_answer: str | None = None
            is_final = False

            if matched_pattern == cfg.tool_call_pattern:
                # Tool call branch
                tool_call = self._parse_tool_call(generated_text)
                if tool_call is not None:
                    tool_result = self.registry.execute(
                        tool_call.get("name", ""),
                        tool_call.get("args", {}),
                    )
                    # Append result to context and continue
                    context = (
                        context + generated_text + f"\n<tool_result>{tool_result}</tool_result>\n"
                    )
                else:
                    # Could not parse tool call — treat as thought and stop
                    is_final = True

            elif matched_pattern == cfg.final_answer_pattern:
                # Final answer branch
                final_answer = self._parse_final_answer(generated_text)
                is_final = True

            else:
                # Neither pattern found — max tokens reached; treat as last thought
                is_final = True

            steps.append(
                AgentStep(
                    thought=generated_text,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    final_answer=final_answer,
                    is_final=is_final,
                )
            )

            if is_final:
                break

        # Ensure last step is marked final when max_steps is exhausted
        if steps and not steps[-1].is_final:
            last = steps[-1]
            steps[-1] = AgentStep(
                thought=last.thought,
                tool_call=last.tool_call,
                tool_result=last.tool_result,
                final_answer=last.final_answer,
                is_final=True,
            )

        return steps

    def format_result(self, steps: list[AgentStep]) -> str:
        """Format steps into human-readable string.

        Includes thoughts, tool calls, tool results, and final answer.
        """
        parts: list[str] = []
        for i, step in enumerate(steps, start=1):
            parts.append(f"Step {i}:")
            parts.append(f"  Thought: {step.thought.strip()}")
            if step.tool_call is not None:
                parts.append(f"  Tool Call: {step.tool_call}")
            if step.tool_result is not None:
                parts.append(f"  Tool Result: {step.tool_result}")
            if step.final_answer is not None:
                parts.append(f"  Final Answer: {step.final_answer}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_agent_prompt(
    system_prompt: str,
    tool_registry: ToolRegistry,
    user_query: str,
    history: list[AgentStep] | None = None,
) -> str:
    """Build the full prompt for the agent.

    Includes:
    - System instruction with available tools
    - Conversation history (thoughts + tool calls + results from prior steps)
    - Current user query

    Format::

        System: {system_prompt}

        Available Tools:
        {tool_registry.describe()}

        User: {user_query}

        [Prior steps if history provided]

        Thought:
    """
    sections: list[str] = [
        f"System: {system_prompt}",
        "",
        "Available Tools:",
        tool_registry.describe(),
        "",
        f"User: {user_query}",
    ]

    if history:
        sections.append("")
        for step in history:
            if step.thought:
                sections.append(f"Thought: {step.thought.strip()}")
            if step.tool_call is not None:
                tool_json = json.dumps(step.tool_call)
                sections.append(f"<tool>{tool_json}</tool>")
            if step.tool_result is not None:
                sections.append(f"<tool_result>{step.tool_result}</tool_result>")
            if step.final_answer is not None:
                sections.append(f"Final Answer: {step.final_answer}")

    sections.append("")
    sections.append("Thought:")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------


def _calc_fn(args: dict) -> str:
    expr = args.get("expr", "0")
    # Hardened: AST-walker arithmetic evaluator (no dynamic code path).
    # Replaces prior use of the dynamic code evaluator (AUR-SEC-2026-0022).
    return str(_safe_arith_eval(expr))


CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Evaluate a mathematical expression. Args: {'expr': str}",
    fn=_calc_fn,
)

WORD_COUNT_TOOL = Tool(
    name="word_count",
    description="Count words in a string. Args: {'text': str}",
    fn=lambda args: str(len(args.get("text", "").split())),
)
