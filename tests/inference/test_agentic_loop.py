"""Tests for src/inference/agentic_loop.py"""

import pytest
import torch
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

from src.inference.agentic_loop import (
    Tool,
    AgentStep,
    AgentConfig,
    ToolRegistry,
    AgentLoop,
    build_agent_prompt,
    CALCULATOR_TOOL,
    WORD_COUNT_TOOL,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    return AureliusConfig(
        n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
        head_dim=16, d_ff=128, vocab_size=256, max_seq_len=64,
    )


@pytest.fixture
def model(small_config):
    torch.manual_seed(42)
    m = AureliusTransformer(small_config)
    m.eval()
    return m


def encode(s: str) -> list[int]:
    """Trivial byte-level encoder, capped at max_seq_len=64."""
    return [min(ord(c), 255) for c in s[:50]]


def decode(ids: list[int]) -> str:
    """Trivial decoder: printable ASCII clamped to [32, 126]."""
    return "".join(chr(max(32, min(126, i))) for i in ids)


@pytest.fixture
def registry():
    return ToolRegistry([CALCULATOR_TOOL, WORD_COUNT_TOOL])


@pytest.fixture
def agent(model, registry):
    cfg = AgentConfig(max_steps=3, max_new_tokens_per_step=8)
    return AgentLoop(model, encode, decode, registry, cfg)


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------

def test_tool_registry_execute_calculator(registry):
    result = registry.execute("calculator", {"expr": "2+2"})
    assert result == "4"


def test_tool_registry_execute_unknown_tool(registry):
    result = registry.execute("no_such_tool", {})
    assert "error" in result.lower() or "not found" in result.lower()


def test_tool_registry_contains(registry):
    assert "calculator" in registry
    assert "word_count" in registry
    assert "nonexistent" not in registry


def test_tool_registry_describe(registry):
    desc = registry.describe()
    assert "calculator" in desc
    assert "word_count" in desc


# ---------------------------------------------------------------------------
# AgentLoop._parse_tool_call tests
# ---------------------------------------------------------------------------

def test_parse_tool_call_valid(agent):
    text = '<tool>{"name": "calculator", "args": {"expr": "3*3"}}</tool>'
    result = agent._parse_tool_call(text)
    assert result is not None
    assert result["name"] == "calculator"
    assert result["args"] == {"expr": "3*3"}


def test_parse_tool_call_invalid(agent):
    text = "<tool>{this is not json}</tool>"
    result = agent._parse_tool_call(text)
    assert result is None


# ---------------------------------------------------------------------------
# AgentLoop._parse_final_answer tests
# ---------------------------------------------------------------------------

def test_parse_final_answer(agent):
    text = "I reasoned through it.\nFinal Answer: Paris"
    result = agent._parse_final_answer(text)
    assert result == "Paris"


def test_parse_final_answer_missing(agent):
    text = "This text has no final answer marker."
    result = agent._parse_final_answer(text)
    assert result is None


# ---------------------------------------------------------------------------
# AgentLoop.run tests
# ---------------------------------------------------------------------------

def test_agent_loop_run_returns_steps(agent):
    steps = agent.run("What is the capital of France?")
    assert isinstance(steps, list)
    assert len(steps) >= 1
    assert all(isinstance(s, AgentStep) for s in steps)


def test_agent_step_final_is_last(agent):
    steps = agent.run("What is 1+1?")
    # The last step must be marked final (or max_steps exhausted)
    assert steps[-1].is_final is True


# ---------------------------------------------------------------------------
# build_agent_prompt tests
# ---------------------------------------------------------------------------

def test_build_agent_prompt_contains_query(registry):
    query = "What is 7 times 8?"
    prompt = build_agent_prompt(
        system_prompt="You are a helpful agent.",
        tool_registry=registry,
        user_query=query,
    )
    assert query in prompt


def test_build_agent_prompt_contains_tools(registry):
    prompt = build_agent_prompt(
        system_prompt="You are a helpful agent.",
        tool_registry=registry,
        user_query="test",
    )
    assert "calculator" in prompt
    assert "word_count" in prompt


def test_build_agent_prompt_ends_with_thought(registry):
    prompt = build_agent_prompt(
        system_prompt="sys",
        tool_registry=registry,
        user_query="q",
    )
    assert prompt.strip().endswith("Thought:")


# ---------------------------------------------------------------------------
# Built-in tool tests
# ---------------------------------------------------------------------------

def test_word_count_tool():
    result = WORD_COUNT_TOOL.fn({"text": "hello world"})
    assert result == "2"


def test_word_count_tool_empty():
    result = WORD_COUNT_TOOL.fn({"text": ""})
    assert result == "0"


def test_calculator_tool_basic():
    result = CALCULATOR_TOOL.fn({"expr": "2+2"})
    assert result == "4"
