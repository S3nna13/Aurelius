"""Tests for CoT faithfulness evaluation module."""

from __future__ import annotations

import random

import pytest

from src.eval.cot_faithfulness import (
    CoTFaithfulnessConfig,
    CoTFaithfulnessEvaluator,
    FaithfulnessResult,
    compute_answer_similarity,
    compute_faithfulness_score,
    corrupt_step,
    counterfactual_faithfulness,
    extract_cot_steps,
    measure_step_influence,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared tiny model fixture
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

SAMPLE_COT = "Step 1: add numbers\nStep 2: check result\nAnswer: 42"


# Byte-level tokenizer (fast, no external dependencies)
def byte_encode(s: str) -> list[int]:
    return list(s.encode("utf-8", errors="replace")[:256])


def byte_decode(ids: list[int]) -> str:
    return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def tiny_model():
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


@pytest.fixture(scope="module")
def fast_cfg():
    """Config with minimal interventions for speed."""
    return CoTFaithfulnessConfig(n_interventions=2, max_steps=10)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = CoTFaithfulnessConfig()
    assert cfg.n_interventions == 5
    assert cfg.perturbation_prob == pytest.approx(0.3)
    assert cfg.answer_token == "Answer:"
    assert cfg.max_steps == 10


# ---------------------------------------------------------------------------
# 2. extract_cot_steps — basic split
# ---------------------------------------------------------------------------


def test_extract_cot_steps_basic():
    steps, answer = extract_cot_steps(SAMPLE_COT)
    assert len(steps) == 2
    assert "Step 1" in steps[0]
    assert "Step 2" in steps[1]
    assert answer.strip() == "42"


# ---------------------------------------------------------------------------
# 3. extract_cot_steps — no answer token
# ---------------------------------------------------------------------------


def test_extract_cot_steps_no_answer():
    text = "Step 1: do something\nStep 2: do more"
    steps, answer = extract_cot_steps(text)
    # No answer token found: answer is empty
    assert answer == ""
    # The text lines without answer token are treated as steps
    assert isinstance(steps, list)


# ---------------------------------------------------------------------------
# 4. corrupt_step — produces at least some [MASK] tokens
# ---------------------------------------------------------------------------


def test_corrupt_step_reduces_words():
    rng = random.Random(0)
    step = "add the numbers together carefully now"
    # Use p=1.0 to guarantee masks appear
    corrupted = corrupt_step(step, rng, p=1.0)
    assert "[MASK]" in corrupted
    assert corrupted != step


# ---------------------------------------------------------------------------
# 5. corrupt_step — p=0.0 returns identical text
# ---------------------------------------------------------------------------


def test_corrupt_step_p0_no_change():
    rng = random.Random(0)
    step = "this should not change at all"
    result = corrupt_step(step, rng, p=0.0)
    assert result == step


# ---------------------------------------------------------------------------
# 6. compute_answer_similarity — identical answers
# ---------------------------------------------------------------------------


def test_compute_answer_similarity_identical():
    score = compute_answer_similarity("the answer is 42", "the answer is 42")
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. compute_answer_similarity — both empty
# ---------------------------------------------------------------------------


def test_compute_answer_similarity_empty_both():
    score = compute_answer_similarity("", "")
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. compute_answer_similarity — disjoint tokens
# ---------------------------------------------------------------------------


def test_compute_answer_similarity_disjoint():
    score = compute_answer_similarity("apple banana cherry", "dog elephant fish")
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. compute_answer_similarity — partial overlap
# ---------------------------------------------------------------------------


def test_compute_answer_similarity_partial():
    score = compute_answer_similarity("the cat sat", "the dog sat")
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# 10. compute_faithfulness_score — result in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_faithfulness_score_range():
    for influences in [[], [0.0], [1.0], [0.2, 0.8], [0.5, 0.5, 0.5]]:
        score = compute_faithfulness_score(influences)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 11. measure_step_influence — result in [0, 1]
# ---------------------------------------------------------------------------


def test_measure_step_influence_range(tiny_model, fast_cfg):
    rng = random.Random(7)
    influence = measure_step_influence(
        tiny_model,
        byte_encode,
        byte_decode,
        SAMPLE_COT,
        step_idx=0,
        cfg=fast_cfg,
        rng=rng,
    )
    assert 0.0 <= influence <= 1.0


# ---------------------------------------------------------------------------
# 12. counterfactual_faithfulness — FaithfulnessResult has all fields
# ---------------------------------------------------------------------------


def test_counterfactual_faithfulness_fields(tiny_model, fast_cfg):
    result = counterfactual_faithfulness(
        tiny_model, byte_encode, byte_decode, SAMPLE_COT, fast_cfg, seed=0
    )
    assert isinstance(result, FaithfulnessResult)
    assert hasattr(result, "faithfulness_score")
    assert hasattr(result, "n_steps")
    assert hasattr(result, "step_influences")
    assert hasattr(result, "answer_consistency")
    assert isinstance(result.step_influences, list)
    assert result.n_steps >= 0


# ---------------------------------------------------------------------------
# 13. counterfactual_faithfulness — score in [0, 1]
# ---------------------------------------------------------------------------


def test_counterfactual_faithfulness_score_range(tiny_model, fast_cfg):
    result = counterfactual_faithfulness(
        tiny_model, byte_encode, byte_decode, SAMPLE_COT, fast_cfg, seed=1
    )
    assert 0.0 <= result.faithfulness_score <= 1.0
    assert 0.0 <= result.answer_consistency <= 1.0


# ---------------------------------------------------------------------------
# 14. CoTFaithfulnessEvaluator.evaluate_batch — returns dict with 3 keys
# ---------------------------------------------------------------------------


def test_evaluator_evaluate_batch_keys(tiny_model, fast_cfg):
    evaluator = CoTFaithfulnessEvaluator(tiny_model, byte_encode, byte_decode, fast_cfg)
    texts = [SAMPLE_COT, "Step 1: multiply\nAnswer: 6"]
    result = evaluator.evaluate_batch(texts)
    assert "mean_faithfulness" in result
    assert "mean_consistency" in result
    assert "n_evaluated" in result
    assert result["n_evaluated"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 15. CoTFaithfulnessEvaluator.summarize — returns mean/min/max/mean_n_steps
# ---------------------------------------------------------------------------


def test_evaluator_summarize_keys(tiny_model, fast_cfg):
    evaluator = CoTFaithfulnessEvaluator(tiny_model, byte_encode, byte_decode, fast_cfg)
    results = [
        FaithfulnessResult(
            faithfulness_score=0.6,
            n_steps=2,
            step_influences=[0.5, 0.7],
            answer_consistency=0.8,
        ),
        FaithfulnessResult(
            faithfulness_score=0.4,
            n_steps=3,
            step_influences=[0.3, 0.4, 0.5],
            answer_consistency=0.6,
        ),
    ]
    summary = evaluator.summarize(results)
    assert "mean_faithfulness" in summary
    assert "min_faithfulness" in summary
    assert "max_faithfulness" in summary
    assert "mean_n_steps" in summary
    assert summary["min_faithfulness"] == pytest.approx(0.4)
    assert summary["max_faithfulness"] == pytest.approx(0.6)
    assert summary["mean_faithfulness"] == pytest.approx(0.5)
    assert summary["mean_n_steps"] == pytest.approx(2.5)
