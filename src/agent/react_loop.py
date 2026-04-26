"""ReAct-style agent loop for Aurelius.

Implements the Reason+Act paradigm from Yao et al. 2022
("ReAct: Synergizing Reasoning and Acting", arXiv:2210.03629).

The loop is intentionally minimal and model-agnostic:

    plan -> act (emit tool_call) -> observe (tool result) -> reflect -> ...

until either the model emits a final answer, the step budget is
exhausted, or a fatal error occurs. Tool execution is sandboxed by the
caller; the loop's own responsibilities are:

    * prompt construction (system + task + step history)
    * model invocation via a pluggable ``generate_fn``
    * tool-call parsing (via :class:`UnifiedToolCallParser`)
    * argument validation against the registered tool signature
    * wall-clock timeout on tool invocation
    * capturing every failure on the producing :class:`AgentStep`

There are no silent fallbacks. Every failure mode is materialised as an
``AgentStep.error`` string so the caller can audit the full trace.

The module deliberately uses only the Python standard library: no torch,
no transformers, no langchain. This keeps the loop runnable in the
training harness, in evaluation sandboxes, and in serving workers.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .tool_call_parser import (
    ParsedToolCall,
    ToolCallParseError,
    UnifiedToolCallParser,
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """A single turn in the ReAct trace.

    Roles mirror the ChatML convention: ``assistant`` for model output,
    ``tool`` for observation turns synthesised from tool results.
    """

    role: str
    content: str
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_output: str | None = None
    error: str | None = None


@dataclass
class AgentTrace:
    """Full transcript of a :meth:`ReActLoop.run` invocation."""

    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    status: str = "no_answer"  # one of {success, budget, error, no_answer}
    steps_used: int = 0


# ---------------------------------------------------------------------------
# Final-answer extraction
# ---------------------------------------------------------------------------


_FINAL_XML_RE = re.compile(
    r"<final_answer>(?P<body>.*?)</final_answer>",
    re.DOTALL,
)
# "Final Answer:" (case-insensitive) at the start of a line.
_FINAL_PREFIX_RE = re.compile(
    r"(?im)^[ \t]*final answer[ \t]*:[ \t]*(?P<body>.*)$",
)


def _extract_final_answer(text: str) -> str | None:
    """Return the final-answer body if present, else ``None``.

    Two forms are recognised:

    * ``<final_answer>...</final_answer>`` (anywhere, multiline body)
    * ``Final Answer: ...`` at the start of a line (single-line body)

    The XML form is preferred when both appear. The extracted body is
    stripped of leading/trailing whitespace but otherwise verbatim.
    """
    if not text:
        return None
    m = _FINAL_XML_RE.search(text)
    if m is not None:
        return m.group("body").strip()
    m = _FINAL_PREFIX_RE.search(text)
    if m is not None:
        return m.group("body").strip()
    return None


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


class ReActLoop:
    """ReAct reasoning+action loop.

    Parameters
    ----------
    generate_fn:
        Callable taking the current message list and returning the raw
        model string for this step. Must be deterministic for a fixed
        input if the caller wants reproducible traces.
    tool_registry:
        Mapping of tool name to a Python callable. Keyword arguments
        are supplied from the parsed tool call; positional-only tools
        are rejected.
    max_steps:
        Upper bound on the number of model turns. Each assistant turn
        counts as one step regardless of whether it produced a tool
        call.
    max_tool_seconds:
        Wall-clock timeout per tool invocation. Timeouts surface as
        ``AgentStep.error`` containing the substring ``"timeout"``; the
        loop continues.
    parser:
        A parser exposing ``.parse(text) -> list[ParsedToolCall]``.
        Defaults to :class:`UnifiedToolCallParser` which auto-detects
        XML or JSON.
    """

    def __init__(
        self,
        generate_fn: Callable[[list[dict]], str],
        tool_registry: dict[str, Callable[..., Any]],
        max_steps: int = 8,
        max_tool_seconds: float = 5.0,
        parser: Any = None,
    ) -> None:
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")
        if not isinstance(tool_registry, dict):
            raise TypeError("tool_registry must be a dict")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if max_tool_seconds <= 0:
            raise ValueError("max_tool_seconds must be > 0")
        self._generate = generate_fn
        self._tools = tool_registry
        self._max_steps = max_steps
        self._tool_timeout = float(max_tool_seconds)
        self._parser = parser if parser is not None else UnifiedToolCallParser()

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(self, task: str, system_prompt: str = "") -> AgentTrace:
        """Drive the loop until terminal condition is reached."""
        if not isinstance(task, str):
            raise TypeError("task must be a str")
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a str")

        trace = AgentTrace()

        # Handle empty task upfront: we still make one call so callers
        # relying on system_prompt-only flows are supported, but if the
        # task is empty AND there is no system prompt we short-circuit.
        if task.strip() == "" and system_prompt.strip() == "":
            trace.status = "no_answer"
            return trace

        for step_idx in range(self._max_steps):
            messages = self._build_messages(
                task=task,
                system_prompt=system_prompt,
                steps=trace.steps,
            )
            try:
                raw = self._generate(messages)
            except Exception as exc:  # noqa: BLE001 - generate_fn is untrusted
                trace.steps.append(
                    AgentStep(
                        role="assistant",
                        content="",
                        error=f"generate_fn raised: {type(exc).__name__}: {exc}",
                    )
                )
                trace.steps_used = len([s for s in trace.steps if s.role == "assistant"])
                trace.status = "error"
                return trace

            if not isinstance(raw, str):
                trace.steps.append(
                    AgentStep(
                        role="assistant",
                        content="",
                        error=f"generate_fn returned non-str: {type(raw).__name__}",
                    )
                )
                trace.steps_used = len([s for s in trace.steps if s.role == "assistant"])
                trace.status = "error"
                return trace

            # Terminal: final answer.
            final = _extract_final_answer(raw)
            if final is not None:
                trace.steps.append(AgentStep(role="assistant", content=raw))
                trace.final_answer = final
                trace.status = "success"
                trace.steps_used = len([s for s in trace.steps if s.role == "assistant"])
                return trace

            # Parse any tool calls in the assistant output.
            assistant_step = AgentStep(role="assistant", content=raw)
            tool_calls: list[ParsedToolCall] = []
            try:
                tool_calls = list(self._parser.parse(raw))
            except ToolCallParseError as exc:
                assistant_step.error = f"tool_call_parse_error: {exc}"
            except Exception as exc:  # noqa: BLE001
                assistant_step.error = f"tool_call_parse_error: {type(exc).__name__}: {exc}"

            if not tool_calls:
                # The model reasoned but neither finalised nor called a
                # tool. Record and continue; next iteration will see this
                # turn as history and can self-correct. If we exhaust
                # budget in this state we return "budget".
                trace.steps.append(assistant_step)
                continue

            # Attach the first tool's bookkeeping to the assistant step
            # for convenience, then emit observation steps for each.
            first = tool_calls[0]
            assistant_step.tool_name = first.name
            assistant_step.tool_input = dict(first.arguments)
            trace.steps.append(assistant_step)

            for call in tool_calls:
                obs = self._dispatch_tool(call)
                trace.steps.append(obs)

        # Budget exhausted without final answer.
        trace.steps_used = len([s for s in trace.steps if s.role == "assistant"])
        trace.status = "budget"
        return trace

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        task: str,
        system_prompt: str,
        steps: list[AgentStep],
    ) -> list[dict]:
        """Render the full conversation as a plain list-of-dicts.

        We use dicts rather than :class:`Message` to keep this module
        import-light: the caller's ``generate_fn`` can transform this
        list to whatever wire format its model demands.
        """
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": task})
        for step in steps:
            if step.role == "assistant":
                messages.append({"role": "assistant", "content": step.content})
            elif step.role == "tool":
                # Surface the observation in a form the model can read.
                body = step.tool_output if step.tool_output is not None else (step.error or "")
                prefix = f"[{step.tool_name}] " if step.tool_name else ""
                messages.append({"role": "tool", "content": f"{prefix}{body}"})
            else:
                # Unknown role: pass through verbatim.
                messages.append({"role": step.role, "content": step.content})
        return messages

    def _dispatch_tool(self, call: ParsedToolCall) -> AgentStep:
        """Validate and execute one tool call, returning an observation.

        Every failure path (unknown tool, bad args, exception, timeout)
        is captured on ``AgentStep.error``; the loop never re-raises.
        """
        step = AgentStep(
            role="tool",
            content="",
            tool_name=call.name,
            tool_input=dict(call.arguments),
        )

        fn = self._tools.get(call.name)
        if fn is None:
            step.error = f"unknown_tool: {call.name!r}"
            return step

        # Best-effort argument validation against the callable signature.
        # This rejects argument names the target does not accept, which
        # blocks trivial "pollute kwargs" injection attempts.
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            # Builtin/C callable with no introspectable signature: skip
            # validation and let the call itself surface TypeErrors.
            sig = None

        kwargs = dict(call.arguments)
        if sig is not None:
            params = sig.parameters
            accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not accepts_var_kw:
                allowed = {
                    name
                    for name, p in params.items()
                    if p.kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }
                unexpected = set(kwargs) - allowed
                if unexpected:
                    step.error = (
                        "invalid_arguments: unexpected keys "
                        f"{sorted(unexpected)!r} for tool {call.name!r}"
                    )
                    return step

        # Execute with wall-clock timeout.
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(lambda: fn(**kwargs))
                try:
                    result = future.result(timeout=self._tool_timeout)
                except concurrent.futures.TimeoutError:
                    # The worker thread cannot be killed, but the loop
                    # moves on; the daemon-ish cleanup at executor exit
                    # lets the thread finish in the background.
                    future.cancel()
                    step.error = f"tool_error: timeout after {self._tool_timeout:.3f}s"
                    return step
        except Exception as exc:  # noqa: BLE001 - tool code is untrusted
            step.error = f"tool_error: {type(exc).__name__}: {exc}"
            return step

        # Normalise output to a string so observation rendering is trivial.
        if isinstance(result, str):
            step.tool_output = result
        else:
            try:
                step.tool_output = repr(result)
            except Exception as exc:  # noqa: BLE001
                step.error = f"tool_error: unrepr-able result: {exc}"
                return step
        step.content = step.tool_output
        return step


__all__ = ["AgentStep", "AgentTrace", "ReActLoop"]
