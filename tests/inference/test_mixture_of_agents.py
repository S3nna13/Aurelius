"""Tests for Mixture-of-Agents inference."""
from __future__ import annotations

import pytest
import torch

from src.inference.mixture_of_agents import (
    AgentResponse,
    MixtureOfAgents,
    MoAConfig,
    compute_response_confidence,
    greedy_generate_text,
    sample_generate_text,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_cfg():
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


@pytest.fixture(scope="module")
def small_model(small_cfg):
    torch.manual_seed(42)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


def _encode(text):
    """Trivial tokenizer: map each char to its ord value (clamped to 0-255)."""
    return [min(ord(c), 255) for c in str(text)]


def _decode(ids):
    """Trivial detokenizer: map ints back to chars."""
    return "".join(chr(min(max(i, 0), 127)) for i in ids)


def _make_moa(model, n_agents=2, max_new_tokens=4):
    cfg = MoAConfig(
        n_proposers=n_agents,
        n_aggregation_rounds=1,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
    )
    return MixtureOfAgents(
        models=[model] * n_agents,
        config=cfg,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
    )


# ---------------------------------------------------------------------------
# 1. MoAConfig defaults
# ---------------------------------------------------------------------------

def test_moa_config_defaults():
    cfg = MoAConfig()
    assert cfg.n_proposers == 3
    assert cfg.n_aggregation_rounds == 1
    assert cfg.max_new_tokens == 128
    assert cfg.temperature == 0.7
    assert cfg.aggregation_prompt == "Synthesize these responses into a single best answer:\n"


# ---------------------------------------------------------------------------
# 2. AgentResponse fields
# ---------------------------------------------------------------------------

def test_agent_response_fields():
    ar = AgentResponse(agent_id=0, response="hello")
    assert ar.agent_id == 0
    assert ar.response == "hello"
    assert ar.confidence == 1.0
    assert ar.tokens_generated == 0


# ---------------------------------------------------------------------------
# 3. greedy_generate_text returns string
# ---------------------------------------------------------------------------

def test_greedy_generate_text_returns_string(small_model):
    prompt_ids = _encode("hi")
    result = greedy_generate_text(small_model, prompt_ids, max_new_tokens=4, tokenizer_decode=_decode)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 4. sample_generate_text returns string
# ---------------------------------------------------------------------------

def test_sample_generate_text_returns_string(small_model):
    prompt_ids = _encode("hi")
    result = sample_generate_text(
        small_model, prompt_ids, max_new_tokens=4, temperature=0.7, tokenizer_decode=_decode
    )
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 5. compute_response_confidence returns float
# ---------------------------------------------------------------------------

def test_compute_response_confidence_returns_float(small_model):
    prompt_ids = _encode("hi")
    response_ids = _encode("world")
    conf = compute_response_confidence(small_model, prompt_ids, response_ids)
    assert isinstance(conf, float)


# ---------------------------------------------------------------------------
# 6. compute_response_confidence returns negative (log prob)
# ---------------------------------------------------------------------------

def test_compute_response_confidence_is_negative(small_model):
    prompt_ids = _encode("hello")
    response_ids = _encode("world")
    conf = compute_response_confidence(small_model, prompt_ids, response_ids)
    assert conf <= 0.0


# ---------------------------------------------------------------------------
# 7. MixtureOfAgents.propose returns list of AgentResponse
# ---------------------------------------------------------------------------

def test_propose_returns_list_of_agent_response(small_model):
    moa = _make_moa(small_model, n_agents=2, max_new_tokens=4)
    responses = moa.propose("hello")
    assert isinstance(responses, list)
    for r in responses:
        assert isinstance(r, AgentResponse)


# ---------------------------------------------------------------------------
# 8. MixtureOfAgents.propose length = n_proposers
# ---------------------------------------------------------------------------

def test_propose_length_equals_n_proposers(small_model):
    moa = _make_moa(small_model, n_agents=2, max_new_tokens=4)
    responses = moa.propose("hello")
    assert len(responses) == 2


# ---------------------------------------------------------------------------
# 9. MixtureOfAgents.propose each response non-empty string
# ---------------------------------------------------------------------------

def test_propose_responses_are_strings(small_model):
    moa = _make_moa(small_model, n_agents=2, max_new_tokens=4)
    responses = moa.propose("hi")
    for r in responses:
        assert isinstance(r.response, str)


# ---------------------------------------------------------------------------
# 10. MixtureOfAgents.aggregate returns string
# ---------------------------------------------------------------------------

def test_aggregate_returns_string(small_model):
    moa = _make_moa(small_model, n_agents=2, max_new_tokens=4)
    fake_responses = [
        AgentResponse(agent_id=0, response="answer one"),
        AgentResponse(agent_id=1, response="answer two"),
    ]
    result = moa.aggregate("question", fake_responses)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 11. MixtureOfAgents.run returns dict with required keys
# ---------------------------------------------------------------------------

def test_run_returns_dict_with_required_keys(small_model):
    moa = _make_moa(small_model, n_agents=2, max_new_tokens=4)
    result = moa.run("hello")
    assert isinstance(result, dict)
    for key in ("answer", "n_responses", "responses", "mean_confidence"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 12. MixtureOfAgents.run answer is string
# ---------------------------------------------------------------------------

def test_run_answer_is_string(small_model):
    moa = _make_moa(small_model, n_agents=2, max_new_tokens=4)
    result = moa.run("hello")
    assert isinstance(result["answer"], str)


# ---------------------------------------------------------------------------
# 13. MixtureOfAgents.rank_responses sorted by confidence desc
# ---------------------------------------------------------------------------

def test_rank_responses_sorted_by_confidence_desc(small_model):
    moa = _make_moa(small_model, n_agents=2, max_new_tokens=4)
    responses = [
        AgentResponse(agent_id=0, response="a", confidence=-2.0),
        AgentResponse(agent_id=1, response="b", confidence=-0.5),
        AgentResponse(agent_id=2, response="c", confidence=-1.0),
    ]
    ranked = moa.rank_responses(responses)
    confidences = [r.confidence for r in ranked]
    assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# 14. MixtureOfAgents.run n_responses = n_proposers
# ---------------------------------------------------------------------------

def test_run_n_responses_equals_n_proposers(small_model):
    moa = _make_moa(small_model, n_agents=2, max_new_tokens=4)
    result = moa.run("hello")
    assert result["n_responses"] == 2
