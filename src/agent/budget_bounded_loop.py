"""Composite budget wrapper around :class:`ReActLoop`.

Implements an outer **tool-invocation budget** on top of the inner loop's
``max_steps`` (LLM turns).  This matches production agent deployments where
API cost and blast radius are bounded independently from reasoning depth
(Yao et al., ReAct, arXiv:2210.03629 — budgeted acting).

The wrapper never swallows failures: exhausted tool budget surfaces as a
normal tool error string so the trace remains auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .react_loop import AgentTrace, ReActLoop


@dataclass
class BudgetState:
    """Mutable counters shared by wrapped tool callables."""

    tool_invocations_used: int = 0


class ToolInvocationBudgetError(RuntimeError):
    """Raised when the configured tool-call cap is reached."""


def _wrap_tool_registry(
    tools: dict[str, Callable[..., Any]],
    state: BudgetState,
    max_tool_invocations: int,
) -> dict[str, Callable[..., Any]]:
    if not isinstance(tools, dict):
        raise TypeError("tool_registry must be a dict")
    if max_tool_invocations < 0:
        raise ValueError("max_tool_invocations must be >= 0")

    wrapped: dict[str, Callable[..., Any]] = {}

    for name, fn in tools.items():
        if not callable(fn):
            raise TypeError(f"tool {name!r} must be callable")

        def make_bound(
            tool_name: str,
            original: Callable[..., Any],
        ) -> Callable[..., Any]:
            def _wrapped(**kwargs: Any) -> Any:
                if state.tool_invocations_used >= max_tool_invocations:
                    raise ToolInvocationBudgetError(
                        f"tool_budget_exhausted: cap={max_tool_invocations} "
                        f"after {state.tool_invocations_used} invocations"
                    )
                state.tool_invocations_used += 1
                return original(**kwargs)

            _wrapped.__name__ = getattr(original, "__name__", "wrapped_tool")
            _wrapped.__doc__ = getattr(original, "__doc__", None)
            return _wrapped

        wrapped[name] = make_bound(name, fn)

    return wrapped


class BudgetBoundedLoop:
    """ReAct loop with an explicit cap on **successful tool dispatches**.

    Parameters
    ----------
    generate_fn, tool_registry
        Forwarded to :class:`ReActLoop` after tool registry wrapping.
    max_llm_turns
        Maps to ``ReActLoop(..., max_steps=...)``.
    max_tool_invocations
        Maximum number of tool bodies actually entered (each wrapped call
        increments before delegating).  ``0`` means every tool entry fails
        immediately with a budget error.
    max_tool_seconds
        Per-tool wall timeout (passed through to ``ReActLoop``).
    parser
        Optional parser override for ``ReActLoop``.
    """

    def __init__(
        self,
        generate_fn: Callable[[list[dict]], str],
        tool_registry: dict[str, Callable[..., Any]],
        *,
        max_llm_turns: int = 8,
        max_tool_invocations: int = 4,
        max_tool_seconds: float = 5.0,
        parser: Any = None,
    ) -> None:
        if max_llm_turns < 1:
            raise ValueError("max_llm_turns must be >= 1")
        self._state = BudgetState()
        wrapped = _wrap_tool_registry(
            tool_registry,
            self._state,
            max_tool_invocations,
        )
        self._inner = ReActLoop(
            generate_fn,
            wrapped,
            max_steps=max_llm_turns,
            max_tool_seconds=max_tool_seconds,
            parser=parser,
        )

    @property
    def tool_invocations_used(self) -> int:
        return self._state.tool_invocations_used

    def run(self, task: str, system_prompt: str = "") -> AgentTrace:
        """Delegate to the inner ReAct loop (same contract)."""
        return self._inner.run(task, system_prompt)


__all__ = [
    "BudgetBoundedLoop",
    "BudgetState",
    "ToolInvocationBudgetError",
    "_wrap_tool_registry",
]
