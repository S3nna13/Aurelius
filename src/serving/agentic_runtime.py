"""Shared runtime helpers for local chat and agentic ReAct generation."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.inference.agentic_loop import (
    CALCULATOR_TOOL,
    WORD_COUNT_TOOL,
    AgentConfig,
    AgentLoop,
    Tool,
    ToolRegistry,
    build_agent_prompt,
)

from .system_prompts import SYSTEM_PROMPTS
from .tool_executor import current_time, echo

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for annotations only
    from .api_server import ChatRequest

_TERMINAL_TURN_RE = re.compile(
    r"<\|(?P<role>system|user|assistant)\|>\n(?P<content>.*?)(?:<\|end\|>\n|<\|end\|>$)",
    re.DOTALL,
)

logger = logging.getLogger(__name__)


class ByteTokenizer:
    """Fallback byte-level tokenizer used when no real tokenizer is available."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: list[int]) -> str:
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_device(model: Any) -> torch.device:
    try:
        param = next(model.parameters())
        return param.device
    except Exception:  # noqa: BLE001
        return torch.device("cpu")


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    if callable(tokenizer):
        return list(tokenizer(text))
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must expose encode(str) -> list[int]")
    return list(encode(text))


def _decode_text(tokenizer: Any, ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer must expose decode(list[int]) -> str")
    return decode(ids)


def _truncate_prompt_ids(model: Any, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    config = getattr(model, "config", None)
    max_seq_len = getattr(config, "max_seq_len", None)
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        return prompt_ids
    max_prompt = max_seq_len - max_new_tokens - 1
    if max_prompt <= 0:
        return prompt_ids[-1:]
    if len(prompt_ids) <= max_prompt:
        return prompt_ids
    return prompt_ids[-max_prompt:]


def _make_default_tool_registry() -> ToolRegistry:
    def _current_time(_: dict) -> str:
        return current_time()

    def _echo(args: dict) -> str:
        message = args.get("message", args.get("text", ""))
        return echo(str(message))

    return ToolRegistry(
        [
            CALCULATOR_TOOL,
            WORD_COUNT_TOOL,
            Tool(
                name="current_time",
                description="Return the current ISO datetime.",
                fn=_current_time,
            ),
            Tool(
                name="echo",
                description="Return the message unchanged.",
                fn=_echo,
            ),
        ]
    )


def _render_conversation(messages: list[dict]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            continue
        if role == "user":
            parts.append(content)
            continue
        label = {
            "assistant": "Assistant",
            "tool": "Tool",
        }.get(role, role.capitalize() or "Message")
        parts.append(f"{label}: {content}")
    return "\n".join(parts).strip()


def _extract_terminal_prompt_context(prompt: str) -> tuple[str | None, str]:
    system_prompt: str | None = None
    messages: list[dict] = []
    for match in _TERMINAL_TURN_RE.finditer(prompt):
        role = match.group("role")
        content = match.group("content").strip()
        if role == "system" and system_prompt is None:
            system_prompt = content
            continue
        messages.append({"role": role, "content": content})
    if not messages and system_prompt is None:
        return None, prompt.strip()
    return system_prompt, _render_conversation(messages)


def _split_request_context(request: ChatRequest, fallback_system: str) -> tuple[str, str]:
    system_prompt = (request.system or "").strip() or fallback_system
    transcript: list[dict] = []
    for msg in request.messages:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system" and not request.system:
            system_prompt = content
            continue
        transcript.append({"role": role, "content": content})
    return system_prompt, _render_conversation(transcript)


def _run_agentic_prompt(
    *,
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    user_query: str,
    tool_registry: ToolRegistry,
    agent_config: AgentConfig,
) -> str:
    encode = lambda text: _truncate_prompt_ids(  # noqa: E731
        model,
        _encode_text(tokenizer, text),
        agent_config.max_new_tokens_per_step,
    )
    decode = lambda ids: _decode_text(tokenizer, ids)  # noqa: E731
    agent = AgentLoop(model, encode, decode, tool_registry, agent_config)
    prompt = build_agent_prompt(system_prompt, tool_registry, user_query)
    steps = agent.run(prompt)
    for step in reversed(steps):
        if step.final_answer:
            return step.final_answer.strip()
    if steps:
        last = steps[-1]
        if last.thought.strip():
            cleaned = re.sub(r"<tool>.*?</tool>", "", last.thought, flags=re.DOTALL).strip()
            return cleaned or last.thought.strip()
    return agent.format_result(steps)


def load_tokenizer_for_chat(checkpoint_dir: str | Path | None = None) -> Any:
    """Load a tokenizer for local chat, falling back to a byte tokenizer."""
    candidates: list[Path] = []
    repo_tokenizer = _repo_root() / "tokenizers" / "aurelius-128k"
    if checkpoint_dir is not None:
        base = Path(checkpoint_dir).expanduser()
        if base.is_file():
            base = base.parent
        candidates.extend(
            [
                base,
                base / "tokenizer",
                base.parent / "tokenizers" / "aurelius-128k",
            ]
        )
    candidates.append(repo_tokenizer)

    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve() if candidate.exists() else candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        tok_file = candidate / "tokenizer.json"
        if not tok_file.exists():
            continue
        try:
            from src.data.tokenizer import AureliusTokenizer
        except ImportError:
            break
        try:
            return AureliusTokenizer.load(candidate)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping tokenizer candidate %s: %s", candidate, exc)
            continue

    return ByteTokenizer()


def build_prompt_generate_fn(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Callable[[str], str]:
    """Return a simple local text-generation callable for a prompt string."""
    device = _model_device(model)

    def _generate(prompt: str) -> str:
        prompt_ids = _truncate_prompt_ids(
            model,
            _encode_text(tokenizer, prompt),
            max_new_tokens,
        )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        new_ids = output_ids[0, len(prompt_ids) :].tolist()
        return _decode_text(tokenizer, new_ids)

    return _generate


def build_agentic_prompt_generate_fn(
    model: Any,
    tokenizer: Any,
    *,
    system_prompt: str | None = None,
    agent_config: AgentConfig | None = None,
    tool_registry: ToolRegistry | None = None,
) -> Callable[[str], str]:
    """Return a ReAct-style generator for terminal prompt strings."""
    registry = tool_registry or _make_default_tool_registry()
    cfg = agent_config or AgentConfig()
    fallback_system = system_prompt or SYSTEM_PROMPTS["agentic"]

    def _generate(prompt: str) -> str:
        parsed_system, user_query = _extract_terminal_prompt_context(prompt)
        final_system = parsed_system or fallback_system
        return _run_agentic_prompt(
            model=model,
            tokenizer=tokenizer,
            system_prompt=final_system,
            user_query=user_query,
            tool_registry=registry,
            agent_config=cfg,
        )

    return _generate


def build_agentic_request_generate_fn(
    model: Any,
    tokenizer: Any,
    *,
    system_prompt: str | None = None,
    agent_config: AgentConfig | None = None,
    tool_registry: ToolRegistry | None = None,
) -> Callable[[ChatRequest], str]:
    """Return a ReAct-style generator for OpenAI-compatible chat requests."""
    registry = tool_registry or _make_default_tool_registry()
    cfg = agent_config or AgentConfig()
    fallback_system = system_prompt or SYSTEM_PROMPTS["agentic"]

    def _generate(request: ChatRequest) -> str:
        request_system, user_query = _split_request_context(request, fallback_system)
        request_cfg = replace(
            cfg,
            max_new_tokens_per_step=request.max_tokens,
            temperature=request.temperature,
        )
        return _run_agentic_prompt(
            model=model,
            tokenizer=tokenizer,
            system_prompt=request_system,
            user_query=user_query,
            tool_registry=registry,
            agent_config=request_cfg,
        )

    return _generate
