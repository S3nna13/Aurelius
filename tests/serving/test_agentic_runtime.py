"""Tests for the shared serving runtime helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.inference.agentic_loop import AgentConfig, AgentStep
from src.serving.agentic_runtime import (
    ByteTokenizer,
    build_agentic_prompt_generate_fn,
    build_agentic_request_generate_fn,
    build_prompt_generate_fn,
    load_tokenizer_for_chat,
)
from src.serving.api_server import ChatRequest
from src.serving.terminal_chat import build_chatml_prompt


class _DummyTokenizer:
    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encoded.append(text)
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(i)) for i in ids)


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_seq_len=64)
        self.calls: list[dict] = []

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        reasoning_budget: int | None = None,
        think_token_id: int = 50400,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "shape": tuple(input_ids.shape),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        extra = torch.tensor([[97, 98, 99]], dtype=torch.long, device=input_ids.device)
        extra = extra[:, :max_new_tokens]
        return torch.cat([input_ids, extra], dim=1)


class _FakeAgentLoop:
    instances: list[_FakeAgentLoop] = []

    def __init__(self, model, encode, decode, tool_registry, config):
        self.model = model
        self.encode = encode
        self.decode = decode
        self.tool_registry = tool_registry
        self.config = config
        self.prompts: list[str] = []
        _FakeAgentLoop.instances.append(self)

    def run(self, prompt: str):
        self.prompts.append(prompt)
        return [
            AgentStep(
                thought="thinking",
                tool_call=None,
                tool_result=None,
                final_answer="done",
                is_final=True,
            )
        ]

    def format_result(self, steps):
        return "formatted"


def test_build_prompt_generate_fn_returns_model_text():
    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    generate_fn = build_prompt_generate_fn(model, tokenizer, max_new_tokens=2)

    result = generate_fn("Hello")

    assert result == "ab"
    assert tokenizer.encoded[-1] == "Hello"
    assert model.calls[0]["max_new_tokens"] == 2


def test_build_agentic_request_generate_fn_renders_request_context(monkeypatch):
    monkeypatch.setattr("src.serving.agentic_runtime.AgentLoop", _FakeAgentLoop)
    _FakeAgentLoop.instances.clear()

    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    generate_fn = build_agentic_request_generate_fn(
        model,
        tokenizer,
        system_prompt="fallback system",
        agent_config=AgentConfig(max_steps=2, max_new_tokens_per_step=4),
    )

    request = ChatRequest(
        model="aurelius",
        messages=[
            {"role": "system", "content": "request system"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "prior"},
        ],
        temperature=0.2,
        max_tokens=7,
    )

    result = generate_fn(request)
    instance = _FakeAgentLoop.instances[-1]

    assert result == "done"
    assert "System: request system" in instance.prompts[-1]
    assert "User: hello" in instance.prompts[-1]
    assert "Assistant: prior" in instance.prompts[-1]
    assert instance.config.max_new_tokens_per_step == 7
    assert instance.config.temperature == 0.2
    assert "calculator" in instance.tool_registry.describe()
    assert "current_time" in instance.tool_registry.describe()


def test_build_agentic_prompt_generate_fn_parses_terminal_prompt(monkeypatch):
    monkeypatch.setattr("src.serving.agentic_runtime.AgentLoop", _FakeAgentLoop)
    _FakeAgentLoop.instances.clear()

    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    generate_fn = build_agentic_prompt_generate_fn(
        model,
        tokenizer,
        system_prompt="fallback system",
        agent_config=AgentConfig(max_steps=2, max_new_tokens_per_step=4),
    )

    prompt = build_chatml_prompt(
        [
            {"role": "user", "content": "How many words are here?"},
            {"role": "assistant", "content": "One two three."},
        ],
        "terminal system",
    )

    result = generate_fn(prompt)
    instance = _FakeAgentLoop.instances[-1]

    assert result == "done"
    assert "System: terminal system" in instance.prompts[-1]
    assert "How many words are here?" in instance.prompts[-1]
    assert "Assistant: One two three." in instance.prompts[-1]
    assert instance.config.max_new_tokens_per_step == 4


@patch("src.data.tokenizer.AureliusTokenizer.load", side_effect=RuntimeError("boom"))
def test_load_tokenizer_for_chat_falls_back_to_byte_tokenizer(mock_load, tmp_path):
    tokenizer = load_tokenizer_for_chat(tmp_path / "missing")
    assert isinstance(tokenizer, ByteTokenizer)
