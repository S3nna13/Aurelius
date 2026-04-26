"""Tests for src/inference/agent_loop.py -- ReAct-style agent loop."""

import pytest
import torch

from src.inference.agent_loop import (
    AgentConfig,
    AgentMemory,
    AgentStep,
    ReActAgent,
    SimpleToolExecutor,
    compute_agent_metrics,
    format_react_prompt,
    parse_react_output,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture
def model(small_config):
    torch.manual_seed(0)
    m = AureliusTransformer(small_config)
    m.eval()
    return m


def encode(s: str) -> list[int]:
    return [min(ord(c), 255) for c in s[:64]]


def decode(ids: list[int]) -> str:
    return "".join(chr(max(32, min(126, i))) for i in ids)


@pytest.fixture
def tool_executor():
    ex = SimpleToolExecutor()
    ex.register("add", lambda a, b: str(int(a) + int(b)))
    ex.register("echo", lambda x: x)
    return ex


@pytest.fixture
def agent_config():
    return AgentConfig(max_steps=2, max_tokens_per_step=16)


@pytest.fixture
def agent(model, tool_executor, agent_config):
    return ReActAgent(model, encode, decode, tool_executor, agent_config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_agent_config_defaults():
    cfg = AgentConfig()
    assert cfg.max_steps == 10
    assert cfg.max_tokens_per_step == 64
    assert cfg.temperature == 0.7
    assert cfg.stop_token == "<|end|>"
    assert cfg.reflection_enabled is True
    assert cfg.thought_prefix == "Thought:"
    assert cfg.action_prefix == "Action:"
    assert cfg.observation_prefix == "Observation:"


def test_agent_step_fields():
    step = AgentStep(
        step_num=0,
        thought="I need to add",
        action="add(1, 2)",
        observation="3",
        is_final=False,
    )
    assert step.step_num == 0
    assert step.thought == "I need to add"
    assert step.action == "add(1, 2)"
    assert step.observation == "3"
    assert step.is_final is False


def test_agent_memory_add_step():
    mem = AgentMemory(max_steps=5)
    step = AgentStep(step_num=0, thought="t", action=None, observation=None, is_final=False)
    mem.add_step(step)
    assert len(mem.steps) == 1
    assert mem.steps[0] is step


def test_agent_memory_format_history():
    mem = AgentMemory()
    mem.add_step(
        AgentStep(step_num=0, thought="think", action="echo(hi)", observation="hi", is_final=False)
    )
    history = mem.format_history()
    assert "Thought: think" in history
    assert "Action: echo(hi)" in history
    assert "Observation: hi" in history


def test_agent_memory_get_last_n():
    mem = AgentMemory()
    for i in range(5):
        mem.add_step(
            AgentStep(step_num=i, thought=f"t{i}", action=None, observation=None, is_final=False)
        )
    last2 = mem.get_last_n(2)
    assert len(last2) == 2
    assert last2[0].step_num == 3
    assert last2[1].step_num == 4


def test_parse_react_output_with_action():
    cfg = AgentConfig()
    text = "Thought: I should add numbers\nAction: add(1, 2)"
    thought, action = parse_react_output(text, cfg)
    assert action is not None
    assert "add(1, 2)" in action


def test_parse_react_output_no_action():
    cfg = AgentConfig()
    text = "Thought: I already know the answer is 42"
    thought, action = parse_react_output(text, cfg)
    assert action is None
    assert thought != ""


def test_format_react_prompt_contains_task():
    cfg = AgentConfig()
    prompt = format_react_prompt(
        task="What is 2+2?",
        history="",
        available_tools=["add", "echo"],
        config=cfg,
    )
    assert "What is 2+2?" in prompt
    assert "Task:" in prompt


def test_tool_executor_register_and_execute():
    ex = SimpleToolExecutor()
    ex.register("double", lambda x: str(int(x) * 2))
    result = ex.execute("double(5)")
    assert result == "10"


def test_tool_executor_missing_tool_returns_error():
    ex = SimpleToolExecutor()
    result = ex.execute("nonexistent()")
    assert "error" in result.lower() or "not found" in result.lower()


def test_react_agent_run_returns_steps(agent):
    final_answer, steps = agent.run("What is 1+1?")
    assert isinstance(final_answer, str)
    assert isinstance(steps, list)
    assert len(steps) >= 1
    assert all(isinstance(s, AgentStep) for s in steps)


def test_compute_agent_metrics_keys():
    steps = [
        AgentStep(step_num=0, thought="t", action="add(1,2)", observation="3", is_final=False),
        AgentStep(step_num=1, thought="done", action=None, observation=None, is_final=True),
    ]
    metrics = compute_agent_metrics(steps)
    assert "n_steps" in metrics
    assert "tool_use_rate" in metrics
    assert "completion_rate" in metrics
    assert metrics["n_steps"] == 2.0
    assert metrics["tool_use_rate"] == 0.5
    assert metrics["completion_rate"] == 1.0
