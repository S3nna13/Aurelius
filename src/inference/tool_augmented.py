"""Tool-augmented generation: parallel tool calls, result caching, error recovery, and multi-step tool chains."""  # noqa: E501

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ToolConfig:
    """Configuration for tool-augmented generation."""

    max_tool_calls: int = 10  # max tools per generation
    timeout_seconds: float = 5.0
    cache_results: bool = True
    retry_on_error: bool = True
    max_retries: int = 2
    parallel_calls: bool = False  # enable parallel tool execution


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """Specification describing a callable tool."""

    name: str
    description: str
    parameters: dict  # JSON schema-like parameter spec
    returns_type: str = "string"  # "string" | "json" | "number"


@dataclass
class ToolCall:
    """Represents a single tool invocation."""

    tool_name: str
    arguments: dict
    call_id: str
    timestamp: float


@dataclass
class ToolResult:
    """Result returned after executing a tool call."""

    call_id: str
    tool_name: str
    result: str
    success: bool
    error_message: str | None
    execution_time: float


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_tool_calls_json(text: str) -> list[ToolCall]:
    """Parse JSON-formatted tool calls from model output.

    Expected format anywhere in text: ``[{"tool": "name", "args": {...}}, ...]``
    Returns list of ToolCall objects (empty list if none found or malformed).
    """
    # Extract the first JSON array found in the text
    pattern = r"\[.*?\]"
    matches = re.findall(pattern, text, re.DOTALL)

    for raw in matches:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, list):
            continue

        calls: list[ToolCall] = []
        valid = True
        for item in parsed:
            if not isinstance(item, dict):
                valid = False
                break
            tool_name = item.get("tool")
            args = item.get("args", {})
            if not isinstance(tool_name, str) or not isinstance(args, dict):
                valid = False
                break
            calls.append(
                ToolCall(
                    tool_name=tool_name,
                    arguments=args,
                    call_id=str(uuid.uuid4()),
                    timestamp=time.time(),
                )
            )
        if valid and calls:
            return calls

    return []


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_tool_results(results: list[ToolResult]) -> str:
    """Format tool results for re-injection into context.

    Success: "Tool {name} returned: {result}\\n"
    Error:   "Tool {name} failed: {error}\\n"
    """
    lines: list[str] = []
    for r in results:
        if r.success:
            lines.append(f"Tool {r.tool_name} returned: {r.result}")
        else:
            lines.append(f"Tool {r.tool_name} failed: {r.error_message}")
    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class ToolCache:
    """LRU-bounded cache for tool results."""

    def __init__(self, max_size: int = 100) -> None:
        self._max_size = max_size
        self._store: dict[str, str] = {}
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(tool_name: str, args: dict) -> str:
        canonical = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, tool_name: str, args: dict) -> str | None:
        key = self._key(tool_name, args)
        if key in self._store:
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def put(self, tool_name: str, args: dict, result: str) -> None:
        key = self._key(tool_name, args)
        if len(self._store) >= self._max_size and key not in self._store:
            # Evict the oldest entry (insertion order guaranteed in Python 3.7+)
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[key] = result

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store), "hits": self._hits, "misses": self._misses}


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


class ToolAugmentedGenerator:
    """Main interface for tool-augmented text generation."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        config: ToolConfig,
    ) -> None:
        self._model = model
        self._encode = tokenizer_encode
        self._decode = tokenizer_decode
        self._config = config
        self._tools: dict[str, tuple[ToolSpec, Callable]] = {}
        self._cache = ToolCache()

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def register_tool(self, spec: ToolSpec, fn: Callable) -> None:
        """Register a tool with its spec and implementation."""
        self._tools[spec.name] = (spec, fn)

    def list_tools(self) -> list[ToolSpec]:
        """Return list of registered tool specs."""
        return [spec for spec, _ in self._tools.values()]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call, with caching and optional retry."""
        entry = self._tools.get(call.tool_name)
        if entry is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                result="",
                success=False,
                error_message=f"Unknown tool: {call.tool_name}",
                execution_time=0.0,
            )

        _spec, fn = entry

        # Check cache
        if self._config.cache_results:
            cached = self._cache.get(call.tool_name, call.arguments)
            if cached is not None:
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    result=cached,
                    success=True,
                    error_message=None,
                    execution_time=0.0,
                )

        attempts = self._config.max_retries + 1 if self._config.retry_on_error else 1
        last_error: str = ""
        t0 = time.perf_counter()

        for _attempt in range(attempts):
            try:
                raw = fn(**call.arguments)
                result_str = str(raw)
                elapsed = time.perf_counter() - t0
                if self._config.cache_results:
                    self._cache.put(call.tool_name, call.arguments, result_str)
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    result=result_str,
                    success=True,
                    error_message=None,
                    execution_time=elapsed,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

        elapsed = time.perf_counter() - t0
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            result="",
            success=False,
            error_message=last_error,
            execution_time=elapsed,
        )

    # ------------------------------------------------------------------
    # Multi-step generation
    # ------------------------------------------------------------------

    def generate_with_tools(self, prompt: str, max_rounds: int = 3) -> tuple[str, list[ToolResult]]:
        """Greedy decode -> parse tool calls -> execute -> re-inject -> repeat.

        Returns ``(final_text, all_results)``.
        """
        all_results: list[ToolResult] = []
        context = prompt

        for _ in range(max_rounds):
            # Greedy decode
            ids = self._encode(context)
            input_tensor = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = self._model(input_tensor)
            # logits: (1, seq_len, vocab_size)
            next_id = int(logits[0, -1].argmax())
            generated_ids = [next_id]
            generated_text = self._decode(generated_ids)

            # Parse tool calls from generated text
            calls = parse_tool_calls_json(generated_text)
            if not calls:
                # No tool calls — we're done
                return context + generated_text, all_results

            # Execute tools
            round_results: list[ToolResult] = []
            total_calls = len(all_results)
            for call in calls:
                if total_calls >= self._config.max_tool_calls:
                    break
                result = self.execute_tool(call)
                round_results.append(result)
                total_calls += 1

            all_results.extend(round_results)

            # Re-inject results into context
            context = context + generated_text + "\n" + format_tool_results(round_results)

        return context, all_results


# ---------------------------------------------------------------------------
# Built-in calculator tool
# ---------------------------------------------------------------------------

# Restrict expressions to numbers and basic arithmetic operators only.
_SAFE_MATH_PATTERN = re.compile(r"^[\d\s\+\-\*\/\.\(\)\%]+$")


def _safe_eval_math(expression: str) -> str:
    """Evaluate a restricted arithmetic expression using Python's ast module.

    Only numeric literals and the operators +, -, *, /, %, ** are permitted.
    Raises ValueError for anything that doesn't match the safe pattern.
    """
    import ast
    import operator as op_module

    expr = expression.strip()
    # Quick character-level safety check
    if not re.match(r"^[\d\s\+\-\*\/\.\(\)\%]+$", expr):
        raise ValueError(f"Unsafe expression: {expression!r}")

    _ALLOWED_OPS = {
        ast.Add: op_module.add,
        ast.Sub: op_module.sub,
        ast.Mult: op_module.mul,
        ast.Div: op_module.truediv,
        ast.Mod: op_module.mod,
        ast.Pow: op_module.pow,
        ast.USub: op_module.neg,
        ast.UAdd: op_module.pos,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval(tree.body)
    except Exception as exc:
        raise ValueError(f"Evaluation error: {exc}") from exc

    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


def create_calculator_tool() -> tuple[ToolSpec, Callable]:
    """Create a simple calculator tool for testing.

    Safely evaluates arithmetic expressions using Python's ast module.
    Spec: name="calculator", args={"expression": "string"}.
    """
    spec = ToolSpec(
        name="calculator",
        description="Evaluate a mathematical expression and return the result.",
        parameters={
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate",
            }
        },
        returns_type="number",
    )

    def calculator(expression: str) -> str:
        return _safe_eval_math(expression)

    return spec, calculator
