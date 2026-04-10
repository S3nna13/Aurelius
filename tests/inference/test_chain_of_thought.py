"""Tests for src/inference/chain_of_thought.py — 15 tests."""
from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.chain_of_thought import (
    CoTConfig,
    CoTOutput,
    ReasoningStep,
    ChainOfThoughtGenerator,
    SelfConsistencyDecoder,
    CoTEvaluator,
    parse_reasoning_steps,
    extract_final_answer,
    greedy_decode_n_tokens,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    torch.manual_seed(42)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


# Byte tokenizer as specified in the task brief
def _encode(s: str) -> list[int]:
    return list(s.encode("utf-8", errors="replace"))


def _decode(ids: list[int]) -> str:
    return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def cot_config():
    return CoTConfig(
        max_reasoning_tokens=8,
        max_answer_tokens=4,
        n_samples=2,
        temperature=1.0,
    )


@pytest.fixture(scope="module")
def generator(tiny_model, cot_config):
    return ChainOfThoughtGenerator(
        model=tiny_model,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        config=cot_config,
    )


# ---------------------------------------------------------------------------
# 1. CoTConfig defaults
# ---------------------------------------------------------------------------

def test_cotconfig_defaults():
    cfg = CoTConfig()
    assert cfg.max_reasoning_tokens == 128
    assert cfg.max_answer_tokens == 32
    assert cfg.temperature == 0.7
    assert cfg.n_samples == 1
    assert cfg.cot_trigger == "Let's think step by step."
    assert cfg.answer_trigger == "Therefore, the answer is:"


# ---------------------------------------------------------------------------
# 2. parse_reasoning_steps splits on newline
# ---------------------------------------------------------------------------

def test_parse_reasoning_steps_splits():
    text = "Step one\nStep two\nStep three"
    steps = parse_reasoning_steps(text)
    assert steps == ["Step one", "Step two", "Step three"]


# ---------------------------------------------------------------------------
# 3. parse_reasoning_steps filters empty strings
# ---------------------------------------------------------------------------

def test_parse_reasoning_steps_filters_empty():
    text = "Step one\n\n\nStep two\n"
    steps = parse_reasoning_steps(text)
    assert "" not in steps
    assert all(s.strip() for s in steps)


# ---------------------------------------------------------------------------
# 4. extract_final_answer finds text after trigger
# ---------------------------------------------------------------------------

def test_extract_final_answer_found():
    trigger = "Therefore, the answer is:"
    text = f"Some reasoning here. {trigger} 42"
    result = extract_final_answer(text, trigger)
    assert result == "42"


# ---------------------------------------------------------------------------
# 5. extract_final_answer returns empty string if trigger absent
# ---------------------------------------------------------------------------

def test_extract_final_answer_missing_trigger():
    result = extract_final_answer("No trigger in this text.", "Therefore, the answer is:")
    assert result == ""


# ---------------------------------------------------------------------------
# 6. greedy_decode_n_tokens returns (list, list) of length n_tokens
# ---------------------------------------------------------------------------

def test_greedy_decode_n_tokens_length(tiny_model):
    prompt_ids = _encode("Hello")
    input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
    n = 5
    with torch.no_grad():
        tokens, log_probs = greedy_decode_n_tokens(tiny_model, input_ids, n)
    assert isinstance(tokens, list)
    assert isinstance(log_probs, list)
    assert len(tokens) == n
    assert len(log_probs) == n


# ---------------------------------------------------------------------------
# 7. greedy_decode_n_tokens log_probs are <= 0
# ---------------------------------------------------------------------------

def test_greedy_decode_log_probs_nonpositive(tiny_model):
    prompt_ids = _encode("Test")
    input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
    _, log_probs = greedy_decode_n_tokens(tiny_model, input_ids, 6)
    assert all(lp <= 0.0 for lp in log_probs), f"Got positive log_prob: {log_probs}"


# ---------------------------------------------------------------------------
# 8. ChainOfThoughtGenerator instantiates
# ---------------------------------------------------------------------------

def test_generator_instantiates(tiny_model, cot_config):
    gen = ChainOfThoughtGenerator(
        model=tiny_model,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        config=cot_config,
    )
    assert gen is not None
    assert gen.model is tiny_model


# ---------------------------------------------------------------------------
# 9. ChainOfThoughtGenerator.generate returns CoTOutput
# ---------------------------------------------------------------------------

def test_generator_generate_returns_cot_output(generator):
    prompt_ids = _encode("What is 2+2?")
    output = generator.generate(prompt_ids)
    assert isinstance(output, CoTOutput)


# ---------------------------------------------------------------------------
# 10. CoTOutput has all required fields
# ---------------------------------------------------------------------------

def test_cot_output_has_required_fields(generator):
    prompt_ids = _encode("Solve: 3*4")
    output = generator.generate(prompt_ids)
    assert hasattr(output, "reasoning_steps")
    assert hasattr(output, "final_answer")
    assert hasattr(output, "raw_tokens")
    assert hasattr(output, "n_reasoning_tokens")


# ---------------------------------------------------------------------------
# 11. CoTOutput.n_reasoning_tokens >= 0
# ---------------------------------------------------------------------------

def test_cot_output_n_reasoning_tokens_nonnegative(generator):
    prompt_ids = _encode("Test prompt")
    output = generator.generate(prompt_ids)
    assert output.n_reasoning_tokens >= 0


# ---------------------------------------------------------------------------
# 12. SelfConsistencyDecoder.decode returns (str, dict)
# ---------------------------------------------------------------------------

def test_self_consistency_decode_returns_str_dict(generator):
    decoder = SelfConsistencyDecoder(generator)
    prompt_ids = _encode("What is 1+1?")
    result = decoder.decode(prompt_ids)
    assert isinstance(result, tuple)
    assert len(result) == 2
    best_answer, stats = result
    assert isinstance(best_answer, str)
    assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# 13. SelfConsistencyDecoder.decode stats has correct keys
# ---------------------------------------------------------------------------

def test_self_consistency_stats_keys(generator):
    decoder = SelfConsistencyDecoder(generator)
    prompt_ids = _encode("Compute 5-3.")
    _, stats = decoder.decode(prompt_ids)
    assert "n_samples" in stats
    assert "answer_counts" in stats
    assert "confidence" in stats
    assert stats["n_samples"] == generator.config.n_samples


# ---------------------------------------------------------------------------
# 14. CoTEvaluator.evaluate_answer_presence True/False
# ---------------------------------------------------------------------------

def test_evaluator_answer_presence(generator):
    evaluator = CoTEvaluator(generator)

    # Output with a non-empty final answer -> True
    output_with_answer = CoTOutput(
        reasoning_steps=[],
        final_answer="42",
        raw_tokens=[],
        n_reasoning_tokens=0,
    )
    assert evaluator.evaluate_answer_presence(output_with_answer) is True

    # Output with empty final answer -> False
    output_no_answer = CoTOutput(
        reasoning_steps=[],
        final_answer="",
        raw_tokens=[],
        n_reasoning_tokens=0,
    )
    assert evaluator.evaluate_answer_presence(output_no_answer) is False


# ---------------------------------------------------------------------------
# 15. CoTEvaluator.batch_evaluate returns dict with correct keys
# ---------------------------------------------------------------------------

def test_evaluator_batch_evaluate_keys(generator):
    evaluator = CoTEvaluator(generator)
    outputs = [
        CoTOutput(
            reasoning_steps=[ReasoningStep(0, "step a", 0.8)],
            final_answer="42",
            raw_tokens=[1, 2, 3],
            n_reasoning_tokens=3,
        ),
        CoTOutput(
            reasoning_steps=[],
            final_answer="",
            raw_tokens=[4, 5],
            n_reasoning_tokens=2,
        ),
    ]
    result = evaluator.batch_evaluate(outputs)
    assert "mean_coherence" in result
    assert "answer_presence_rate" in result
    assert "mean_steps" in result
    # Sanity-check values
    assert 0.0 <= result["mean_coherence"] <= 1.0
    assert result["answer_presence_rate"] == pytest.approx(0.5)
    assert result["mean_steps"] == pytest.approx(0.5)
