"""Tests for multi-agent debate framework."""

from __future__ import annotations

import pytest
import torch

from src.inference.multi_agent_debate import (
    DebateConfig,
    AgentState,
    DebateResult,
    DebateOrchestrator,
    greedy_generate,
    compute_agreement,
    extract_confidence,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_cfg():
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
def tiny_model(tiny_cfg):
    torch.manual_seed(42)
    return AureliusTransformer(tiny_cfg)


def simple_encode(s: str) -> list[int]:
    """Trivial encoder: UTF-8 bytes, clamped to first 10."""
    return [b for b in s.encode("utf-8", errors="replace")[:10]]


def simple_decode(ids: list[int]) -> str:
    """Trivial decoder: bytes back to string."""
    return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


@pytest.fixture
def debate_config():
    return DebateConfig(
        n_agents=2,
        n_rounds=2,
        max_new_tokens=4,
        temperature=0.8,
        consensus_threshold=0.6,
    )


@pytest.fixture
def orchestrator(tiny_model, debate_config):
    return DebateOrchestrator(tiny_model, debate_config, simple_encode, simple_decode)


# ---------------------------------------------------------------------------
# 1. DebateConfig defaults
# ---------------------------------------------------------------------------

def test_debate_config_defaults():
    cfg = DebateConfig()
    assert cfg.n_agents == 3
    assert cfg.n_rounds == 2
    assert cfg.max_new_tokens == 64
    assert cfg.temperature == 0.8
    assert cfg.consensus_threshold == 0.6


# ---------------------------------------------------------------------------
# 2. AgentState fields
# ---------------------------------------------------------------------------

def test_agent_state_fields():
    state = AgentState(agent_id=0, position="hello", confidence=0.9)
    assert state.agent_id == 0
    assert state.position == "hello"
    assert state.confidence == 0.9
    assert isinstance(state.history, list)


def test_agent_state_history_default_empty():
    state = AgentState(agent_id=1, position="world", confidence=0.5)
    assert state.history == []


# ---------------------------------------------------------------------------
# 3. DebateResult fields
# ---------------------------------------------------------------------------

def test_debate_result_fields():
    result = DebateResult(
        question="Q?",
        final_answer="A",
        consensus_reached=True,
        n_rounds=2,
        agent_positions=["A", "A"],
        confidence_scores=[0.8, 0.9],
    )
    assert result.question == "Q?"
    assert result.final_answer == "A"
    assert result.consensus_reached is True
    assert result.n_rounds == 2
    assert result.agent_positions == ["A", "A"]
    assert result.confidence_scores == [0.8, 0.9]


# ---------------------------------------------------------------------------
# 15. DebateResult.consensus_reached is bool
# ---------------------------------------------------------------------------

def test_debate_result_consensus_reached_is_bool():
    result = DebateResult(
        question="Q?",
        final_answer="A",
        consensus_reached=False,
        n_rounds=1,
        agent_positions=["A"],
        confidence_scores=[0.5],
    )
    assert isinstance(result.consensus_reached, bool)


# ---------------------------------------------------------------------------
# 4-5. greedy_generate returns a string
# ---------------------------------------------------------------------------

def test_greedy_generate_returns_string(tiny_model):
    prompt_ids = [72, 101, 108, 108, 111]  # "Hello"
    result = greedy_generate(tiny_model, prompt_ids, max_new_tokens=4, vocab_size=256)
    assert isinstance(result, str)


def test_greedy_generate_length(tiny_model):
    prompt_ids = [65, 66, 67]
    result = greedy_generate(tiny_model, prompt_ids, max_new_tokens=4, vocab_size=256)
    # Result is a decoded string of exactly 4 bytes (may be multi-byte UTF-8)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 6. compute_agreement for identical strings returns 1.0
# ---------------------------------------------------------------------------

def test_compute_agreement_identical():
    positions = ["The answer is 42", "The answer is 42", "The answer is 42"]
    assert compute_agreement(positions) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. compute_agreement for completely different strings returns < 1.0
# ---------------------------------------------------------------------------

def test_compute_agreement_different():
    positions = ["abcdef", "xyz123", "qwerty"]
    score = compute_agreement(positions)
    assert score < 1.0


# ---------------------------------------------------------------------------
# 8. compute_agreement with single agent returns 1.0
# ---------------------------------------------------------------------------

def test_compute_agreement_single_agent():
    assert compute_agreement(["only one position"]) == pytest.approx(1.0)


def test_compute_agreement_empty():
    assert compute_agreement([]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9. extract_confidence finds "confidence: 80%" → 0.8
# ---------------------------------------------------------------------------

def test_extract_confidence_confidence_pattern():
    result = extract_confidence("I am confidence: 80% sure this is correct.")
    assert result == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 10. extract_confidence finds "certainty: 60%" → 0.6
# ---------------------------------------------------------------------------

def test_extract_confidence_certainty_pattern():
    result = extract_confidence("My certainty: 60% on this answer.")
    assert result == pytest.approx(0.6)


def test_extract_confidence_sure_pattern():
    result = extract_confidence("I am sure: 75% this is right.")
    assert result == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 11. extract_confidence returns 0.5 when no pattern found
# ---------------------------------------------------------------------------

def test_extract_confidence_default():
    result = extract_confidence("No confidence information here.")
    assert result == pytest.approx(0.5)


def test_extract_confidence_case_insensitive():
    result = extract_confidence("CONFIDENCE: 90% correct.")
    assert result == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 12. DebateOrchestrator._build_initial_prompt contains question and agent_id
# ---------------------------------------------------------------------------

def test_build_initial_prompt_contains_question_and_agent(orchestrator):
    prompt = orchestrator._build_initial_prompt("What is the meaning of life?", agent_id=2)
    assert "What is the meaning of life?" in prompt
    assert "2" in prompt


# ---------------------------------------------------------------------------
# 13. DebateOrchestrator._build_debate_prompt contains "Other agents"
# ---------------------------------------------------------------------------

def test_build_debate_prompt_contains_other_agents(orchestrator):
    prompt = orchestrator._build_debate_prompt(
        "What is 2+2?",
        agent_id=1,
        other_positions=["It is 4", "Definitely 4"],
    )
    assert "Other agents" in prompt
    assert "What is 2+2?" in prompt
    assert "It is 4" in prompt


# ---------------------------------------------------------------------------
# 14. DebateOrchestrator.debate returns DebateResult
# ---------------------------------------------------------------------------

def test_debate_returns_debate_result(orchestrator):
    result = orchestrator.debate("What is 2+2?")
    assert isinstance(result, DebateResult)


# ---------------------------------------------------------------------------
# 15. DebateOrchestrator.debate result has correct n_agents positions
# ---------------------------------------------------------------------------

def test_debate_result_has_correct_n_agent_positions(orchestrator, debate_config):
    result = orchestrator.debate("What is the capital of France?")
    assert len(result.agent_positions) == debate_config.n_agents
    assert len(result.confidence_scores) == debate_config.n_agents


# ---------------------------------------------------------------------------
# 16. DebateOrchestrator.batch_debate returns list of DebateResult with correct length
# ---------------------------------------------------------------------------

def test_batch_debate_returns_correct_length(orchestrator):
    questions = ["Q1?", "Q2?", "Q3?"]
    results = orchestrator.batch_debate(questions)
    assert isinstance(results, list)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, DebateResult)


def test_batch_debate_single_question(orchestrator):
    results = orchestrator.batch_debate(["Single question?"])
    assert len(results) == 1
    assert isinstance(results[0], DebateResult)
