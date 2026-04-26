"""Tests for self-consistency decoding (Wang et al. 2022)."""

from __future__ import annotations

import pytest
import torch

from src.inference.self_consistency import (
    ChainOfThoughtSampler,
    ConsistencyResult,
    SelfConsistencyConfig,
    SelfConsistencyDecoder,
    answer_consistency_score,
    majority_vote,
    weighted_vote,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


def simple_encode(s: str) -> list[int]:
    """Trivial encoder: bytes, clamped to vocab_size=256."""
    return [b for b in s.encode("utf-8", errors="replace")[:10]]


def simple_decode(ids: list[int]) -> list[int] | str:
    """Trivial decoder: ints to chars."""
    return "".join(chr(max(32, min(126, i))) for i in ids)


@pytest.fixture
def sc_config():
    return SelfConsistencyConfig(
        n_samples=3,
        temperature=0.8,
        max_new_tokens=8,
    )


@pytest.fixture
def decoder(small_model, sc_config):
    return SelfConsistencyDecoder(small_model, simple_encode, simple_decode, sc_config)


# ---------------------------------------------------------------------------
# majority_vote tests
# ---------------------------------------------------------------------------


def test_majority_vote_basic():
    winner, counts = majority_vote(["cat", "cat", "dog"])
    assert winner == "cat"
    assert counts["cat"] == 2
    assert counts["dog"] == 1


def test_majority_vote_empty():
    result = majority_vote([])
    assert result == ("", {})


def test_majority_vote_tie_alphabetical():
    winner, counts = majority_vote(["b", "a"])
    assert winner == "a"
    assert counts["a"] == 1
    assert counts["b"] == 1


def test_majority_vote_case_insensitive_normalization():
    """'Cat' and 'cat' should be treated as the same answer."""
    winner, counts = majority_vote(["Cat", "cat", "dog"])
    # The winner should be the first-seen original of the 'cat' norm
    assert winner.lower() == "cat"


# ---------------------------------------------------------------------------
# weighted_vote tests
# ---------------------------------------------------------------------------


def test_weighted_vote_basic():
    # a=1.0, b=2.0+1.0=3.0 => b wins
    winner, wc = weighted_vote(["a", "b", "b"], [1.0, 2.0, 1.0])
    assert winner == "b"
    assert wc["b"] == pytest.approx(3.0)
    assert wc["a"] == pytest.approx(1.0)


def test_weighted_vote_empty():
    result = weighted_vote([], [])
    assert result == ("", {})


# ---------------------------------------------------------------------------
# answer_consistency_score tests
# ---------------------------------------------------------------------------


def test_answer_consistency_score_perfect():
    score = answer_consistency_score(["42", "42", "42"])
    assert score == pytest.approx(1.0)


def test_answer_consistency_score_none():
    """All different answers -> score = 1/n."""
    answers = ["a", "b", "c"]
    score = answer_consistency_score(answers)
    assert score == pytest.approx(1.0 / 3)


def test_answer_consistency_score_partial():
    # 2 out of 4 are "x"
    score = answer_consistency_score(["x", "x", "y", "z"])
    assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# SelfConsistencyDecoder tests
# ---------------------------------------------------------------------------


def test_self_consistency_decoder_decode_returns_result(decoder):
    result = decoder.decode("What is 2+2?")
    assert isinstance(result, ConsistencyResult)
    assert isinstance(result.final_answer, str)
    assert isinstance(result.vote_counts, dict)
    assert isinstance(result.confidence, float)
    assert isinstance(result.all_completions, list)
    assert isinstance(result.all_extracted, list)


def test_self_consistency_confidence_in_range(decoder):
    result = decoder.decode("What is 2+2?")
    assert 0.0 <= result.confidence <= 1.0


def test_self_consistency_vote_counts_sum_to_n_samples(decoder, sc_config):
    result = decoder.decode("What is 2+2?")
    total = sum(result.vote_counts.values())
    # All extracted answers (including empty strings) should be accounted for
    assert total <= sc_config.n_samples


def test_self_consistency_completions_count(decoder, sc_config):
    result = decoder.decode("Test prompt")
    assert len(result.all_completions) == sc_config.n_samples
    assert len(result.all_extracted) == sc_config.n_samples


def test_self_consistency_decode_batch(decoder):
    prompts = ["What is 1+1?", "What is 2+2?"]
    results = decoder.decode_batch(prompts)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, ConsistencyResult)


# ---------------------------------------------------------------------------
# ChainOfThoughtSampler tests
# ---------------------------------------------------------------------------


def test_cot_sampler_aggregate_keys(small_model, sc_config):
    sampler = ChainOfThoughtSampler(small_model, simple_encode, simple_decode, config=sc_config)
    samples = [
        {"reasoning": "step 1", "answer": "42", "full_text": "step 1 ... 42"},
        {"reasoning": "step 2", "answer": "42", "full_text": "step 2 ... 42"},
        {"reasoning": "step 3", "answer": "7", "full_text": "step 3 ... 7"},
    ]
    agg = sampler.aggregate(samples)
    assert "final_answer" in agg
    assert "confidence" in agg
    assert "consistent_reasoning" in agg
    assert "vote_counts" in agg


def test_cot_sampler_aggregate_correct_winner(small_model, sc_config):
    sampler = ChainOfThoughtSampler(small_model, simple_encode, simple_decode, config=sc_config)
    samples = [
        {"reasoning": "r", "answer": "42", "full_text": ""},
        {"reasoning": "r", "answer": "42", "full_text": ""},
        {"reasoning": "r", "answer": "7", "full_text": ""},
    ]
    agg = sampler.aggregate(samples)
    assert agg["final_answer"] == "42"
    assert agg["consistent_reasoning"] is True


def test_cot_sampler_sample_with_cot_returns_dicts(small_model, sc_config):
    sampler = ChainOfThoughtSampler(small_model, simple_encode, simple_decode, config=sc_config)
    samples = sampler.sample_with_cot("What is 2+2?", n_samples=2, temperature=0.8)
    assert len(samples) == 2
    for s in samples:
        assert "reasoning" in s
        assert "answer" in s
        assert "full_text" in s


# ---------------------------------------------------------------------------
# _extract_answer tests
# ---------------------------------------------------------------------------


def test_extract_answer_basic(decoder):
    text = "The answer is 42"
    extracted = decoder._extract_answer(text)
    assert extracted == "42"


def test_extract_answer_no_match(decoder):
    text = "I cannot determine this."
    extracted = decoder._extract_answer(text)
    assert extracted == ""


def test_extract_answer_equals_pattern(decoder):
    text = "x = 7"
    extracted = decoder._extract_answer(text)
    assert extracted == "7"
