"""Tests for Socratic self-evaluation module."""
from __future__ import annotations

import torch
import pytest

from src.eval.socratic_eval import (
    SocraticConfig,
    CritiqueResult,
    SocraticEvaluator,
    CRITIQUE_QUESTIONS,
    QUESTION_POLARITIES,
    compute_critique_score,
    critique_response,
    greedy_decode,
    score_from_logits,
    socratic_evaluate,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
    head_dim=32, d_ff=128, vocab_size=256, max_seq_len=512,
)

# Byte-level tokenizer (truncate to 256 to stay within vocab)
encode = lambda s: list(s.encode("utf-8")[:256])
decode = lambda ids: bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(42)
    model = AureliusTransformer(TINY_CFG)
    model.train(False)
    return model


@pytest.fixture(scope="module")
def fast_cfg():
    """Config with minimal generation for speed."""
    return SocraticConfig(
        n_critique_questions=5,
        max_new_tokens=3,
        score_from_logits=True,
        temperature=0.7,
    )


@pytest.fixture(scope="module")
def evaluator(tiny_model, fast_cfg):
    return SocraticEvaluator(
        model=tiny_model,
        encode_fn=encode,
        decode_fn=decode,
        cfg=fast_cfg,
    )


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = SocraticConfig()
    assert cfg.n_critique_questions == 5
    assert cfg.max_new_tokens == 20


# ---------------------------------------------------------------------------
# 2. test_critique_questions_count
# ---------------------------------------------------------------------------

def test_critique_questions_count():
    assert len(CRITIQUE_QUESTIONS) == 5


# ---------------------------------------------------------------------------
# 3. test_question_polarities_count
# ---------------------------------------------------------------------------

def test_question_polarities_count():
    assert len(QUESTION_POLARITIES) == 5


# ---------------------------------------------------------------------------
# 4. test_compute_critique_score_positive
# ---------------------------------------------------------------------------

def test_compute_critique_score_positive():
    score = compute_critique_score(yes_prob=0.8, no_prob=0.2, positive_question=True)
    assert score > 0.5, f"Expected high score, got {score}"
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 5. test_compute_critique_score_negative
# ---------------------------------------------------------------------------

def test_compute_critique_score_negative():
    score = compute_critique_score(yes_prob=0.2, no_prob=0.8, positive_question=False)
    assert score > 0.5, f"Expected high score, got {score}"
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 6. test_compute_critique_score_range
# ---------------------------------------------------------------------------

def test_compute_critique_score_range():
    for yes_p in [0.0, 0.1, 0.5, 0.9, 1.0]:
        for no_p in [0.0, 0.1, 0.5, 0.9, 1.0]:
            for polarity in [True, False]:
                s = compute_critique_score(yes_p, no_p, polarity)
                assert 0.0 <= s <= 1.0, f"score {s} out of range for y={yes_p} n={no_p} p={polarity}"


# ---------------------------------------------------------------------------
# 7. test_score_from_logits_probs_valid
# ---------------------------------------------------------------------------

def test_score_from_logits_probs_valid(tiny_model):
    p_yes, p_no = score_from_logits(
        tiny_model, encode, "Hello world", yes_token_id=121, no_token_id=110
    )
    assert 0.0 < p_yes < 1.0, f"p_yes={p_yes} not in (0, 1)"
    assert 0.0 < p_no < 1.0, f"p_no={p_no} not in (0, 1)"


# ---------------------------------------------------------------------------
# 8. test_greedy_decode_shape
# ---------------------------------------------------------------------------

def test_greedy_decode_shape(tiny_model):
    input_ids = torch.tensor([[72, 101, 108, 108, 111]], dtype=torch.long)  # "Hello"
    max_new = 3
    generated = greedy_decode(tiny_model, input_ids, max_new_tokens=max_new)
    assert generated.shape == (max_new,), f"Expected shape ({max_new},), got {generated.shape}"


# ---------------------------------------------------------------------------
# 9. test_greedy_decode_vocab_range
# ---------------------------------------------------------------------------

def test_greedy_decode_vocab_range(tiny_model):
    input_ids = torch.tensor([[72, 101, 108, 108, 111]], dtype=torch.long)
    generated = greedy_decode(tiny_model, input_ids, max_new_tokens=5)
    assert all(0 <= t.item() < TINY_CFG.vocab_size for t in generated), \
        f"Token out of vocab range: {generated.tolist()}"


# ---------------------------------------------------------------------------
# 10. test_critique_response_fields
# ---------------------------------------------------------------------------

def test_critique_response_fields(tiny_model, fast_cfg):
    result = critique_response(
        model=tiny_model,
        encode_fn=encode,
        decode_fn=decode,
        question="What is 2+2?",
        response="The answer is 4.",
        critique_question=CRITIQUE_QUESTIONS[0],
        question_idx=0,
        cfg=fast_cfg,
    )
    assert isinstance(result, CritiqueResult)
    assert isinstance(result.question, str)
    assert isinstance(result.answer, str)
    assert isinstance(result.yes_probability, float)
    assert isinstance(result.no_probability, float)
    assert isinstance(result.score, float)


# ---------------------------------------------------------------------------
# 11. test_critique_response_score_range
# ---------------------------------------------------------------------------

def test_critique_response_score_range(tiny_model, fast_cfg):
    result = critique_response(
        model=tiny_model,
        encode_fn=encode,
        decode_fn=decode,
        question="What is 2+2?",
        response="The answer is 4.",
        critique_question=CRITIQUE_QUESTIONS[1],
        question_idx=1,
        cfg=fast_cfg,
    )
    assert 0.0 <= result.score <= 1.0, f"score {result.score} out of [0, 1]"


# ---------------------------------------------------------------------------
# 12. test_socratic_evaluate_keys
# ---------------------------------------------------------------------------

def test_socratic_evaluate_keys(tiny_model, fast_cfg):
    out = socratic_evaluate(
        model=tiny_model,
        encode_fn=encode,
        decode_fn=decode,
        question="What is the capital of France?",
        response="Paris is the capital of France.",
        cfg=fast_cfg,
    )
    required_keys = {
        "overall_score", "critique_results",
        "consistency_score", "directness_score",
        "error_score", "clarity_score",
    }
    assert required_keys.issubset(out.keys()), \
        f"Missing keys: {required_keys - set(out.keys())}"


# ---------------------------------------------------------------------------
# 13. test_socratic_evaluate_score_range
# ---------------------------------------------------------------------------

def test_socratic_evaluate_score_range(tiny_model, fast_cfg):
    out = socratic_evaluate(
        model=tiny_model,
        encode_fn=encode,
        decode_fn=decode,
        question="What is gravity?",
        response="Gravity is a fundamental force.",
        cfg=fast_cfg,
    )
    assert 0.0 <= out["overall_score"] <= 1.0, \
        f"overall_score {out['overall_score']} out of [0, 1]"


# ---------------------------------------------------------------------------
# 14. test_evaluator_batch_keys
# ---------------------------------------------------------------------------

def test_evaluator_batch_keys(evaluator):
    qa_pairs = [
        ("What is 1+1?", "The answer is 2."),
        ("What color is the sky?", "The sky is blue."),
    ]
    result = evaluator.evaluate_batch(qa_pairs)
    required_keys = {"mean_overall_score", "mean_consistency", "mean_clarity", "n_evaluated"}
    assert required_keys.issubset(result.keys()), \
        f"Missing keys: {required_keys - set(result.keys())}"
    assert len(result) == 4


# ---------------------------------------------------------------------------
# 15. test_evaluator_rank_responses
# ---------------------------------------------------------------------------

def test_evaluator_rank_responses(evaluator):
    question = "What is the capital of France?"
    responses = [
        "Paris is the capital.",
        "London is the capital.",
        "The capital is Berlin.",
    ]
    ranked = evaluator.rank_responses(question, responses)
    assert len(ranked) == len(responses), f"Expected {len(responses)} indices, got {len(ranked)}"
    assert set(ranked) == set(range(len(responses))), \
        f"Indices should cover {{0, 1, 2}}, got {ranked}"
    # All indices in valid range
    for idx in ranked:
        assert 0 <= idx < len(responses), f"Index {idx} out of range"
