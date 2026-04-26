"""Tests for src/inference/in_context_learning.py"""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F

from src.inference.in_context_learning import (
    FewShotExample,
    ICLConfig,
    ICLEvaluator,
    calibrate_logits,
    compute_embedding,
    format_few_shot_prompt,
    select_diverse_examples,
    select_random_examples,
    select_similar_examples,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


def _make_model() -> AureliusTransformer:
    torch.manual_seed(0)
    model = AureliusTransformer(_MODEL_CFG)
    model.eval()
    return model


def _encode(s: str) -> list[int]:
    return [min(ord(c), 255) for c in s[:64]]


def _decode(ids: list[int]) -> str:
    return "".join(chr(max(32, min(126, i))) for i in ids)


def _make_pool(n: int = 6, with_emb: bool = False) -> list[FewShotExample]:
    examples = [FewShotExample(input=f"question {i}", output=f"answer {i}") for i in range(n)]
    if with_emb:
        for ex in examples:
            raw = torch.randn(256)
            ex.embedding = F.normalize(raw, p=2, dim=-1)
    return examples


# ---------------------------------------------------------------------------
# 1. ICLConfig defaults
# ---------------------------------------------------------------------------


def test_icl_config_defaults():
    cfg = ICLConfig()
    assert cfg.n_shots == 4
    assert cfg.selection_strategy == "random"
    assert cfg.max_context_len == 2048
    assert cfg.label_smoothing == 0.0
    assert cfg.calibrate is False
    assert "{input}" in cfg.template
    assert "{output}" in cfg.template


# ---------------------------------------------------------------------------
# 2. format_few_shot_prompt contains all example inputs and outputs
# ---------------------------------------------------------------------------


def test_format_few_shot_prompt_contains_examples():
    pool = _make_pool(3)
    prompt = format_few_shot_prompt(pool, "my query", ICLConfig().template)
    for ex in pool:
        assert ex.input in prompt
        assert ex.output in prompt


# ---------------------------------------------------------------------------
# 3. format_few_shot_prompt: query input present, query output blank
# ---------------------------------------------------------------------------


def test_format_few_shot_prompt_query_input_no_output():
    pool = _make_pool(2)
    query = "unique_query_string"
    prompt = format_few_shot_prompt(pool, query, ICLConfig().template)
    assert query in prompt
    # The output placeholder should not be left as a literal {output}
    assert "{output}" not in prompt
    # The query output should be blank (empty string after "A: ")
    assert "Q: unique_query_string" in prompt


# ---------------------------------------------------------------------------
# 4. select_random_examples returns n items
# ---------------------------------------------------------------------------


def test_select_random_examples_returns_n():
    pool = _make_pool(10)
    rng = random.Random(42)
    result = select_random_examples(pool, 4, rng)
    assert len(result) == 4


# ---------------------------------------------------------------------------
# 5. select_random_examples handles n > pool_size
# ---------------------------------------------------------------------------


def test_select_random_examples_n_greater_than_pool():
    pool = _make_pool(3)
    rng = random.Random(42)
    result = select_random_examples(pool, 10, rng)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# 6. compute_embedding: shape (B, V) and L2-normalised
# ---------------------------------------------------------------------------


def test_compute_embedding_shape_and_normalized():
    B, T, V = 2, 5, 256
    logits = torch.randn(B, T, V)
    token_ids = torch.zeros(B, T, dtype=torch.long)
    emb = compute_embedding(token_ids, logits)

    assert emb.shape == (B, V)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5)


# ---------------------------------------------------------------------------
# 7. select_similar_examples returns n most similar (with embeddings set)
# ---------------------------------------------------------------------------


def test_select_similar_examples_returns_n_most_similar():
    pool = _make_pool(8, with_emb=True)
    query_emb = F.normalize(torch.randn(256), p=2, dim=-1)
    result = select_similar_examples(pool, query_emb, 3)
    assert len(result) == 3
    # Verify they are the top-3 by cosine similarity
    sims = [torch.dot(query_emb, ex.embedding).item() for ex in pool]
    top3_sims = sorted(sims, reverse=True)[:3]
    result_sims = [torch.dot(query_emb, ex.embedding).item() for ex in result]
    assert sorted(result_sims, reverse=True) == pytest.approx(top3_sims, abs=1e-5)


# ---------------------------------------------------------------------------
# 8. select_diverse_examples returns n items
# ---------------------------------------------------------------------------


def test_select_diverse_examples_returns_n():
    pool = _make_pool(8, with_emb=True)
    rng = random.Random(7)
    result = select_diverse_examples(pool, 4, rng)
    assert len(result) == 4


def test_select_diverse_examples_fallback_no_embeddings():
    pool = _make_pool(6, with_emb=False)
    rng = random.Random(7)
    result = select_diverse_examples(pool, 3, rng)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# 9. calibrate_logits shifts logits by content-free distribution
# ---------------------------------------------------------------------------


def test_calibrate_logits_shifts_correctly():
    logits = torch.tensor([2.0, 1.0, 0.0])
    cf_probs = torch.tensor([0.5, 0.3, 0.2])
    calibrated = calibrate_logits(logits, cf_probs)
    expected = logits - torch.log(cf_probs + 1e-9)
    assert torch.allclose(calibrated, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 10. calibrate_logits uniform content-free doesn't change relative ordering
# ---------------------------------------------------------------------------


def test_calibrate_logits_uniform_preserves_ordering():
    logits = torch.tensor([3.0, 1.0, 2.0])
    cf_probs = torch.full((3,), 1.0 / 3.0)
    calibrated = calibrate_logits(logits, cf_probs)
    orig_order = logits.argsort(descending=True).tolist()
    calib_order = calibrated.argsort(descending=True).tolist()
    assert orig_order == calib_order


# ---------------------------------------------------------------------------
# 11. ICLEvaluator.select_examples returns n_shots examples
# ---------------------------------------------------------------------------


def test_icl_evaluator_select_examples_returns_n_shots():
    model = _make_model()
    cfg = ICLConfig(n_shots=3, selection_strategy="random")
    evaluator = ICLEvaluator(model, _encode, _decode, cfg)
    pool = _make_pool(8)
    selected = evaluator.select_examples(pool, "what is this?")
    assert len(selected) == 3


# ---------------------------------------------------------------------------
# 12. ICLEvaluator.build_prompt returns string
# ---------------------------------------------------------------------------


def test_icl_evaluator_build_prompt_returns_string():
    model = _make_model()
    cfg = ICLConfig(n_shots=2, max_context_len=500)
    evaluator = ICLEvaluator(model, _encode, _decode, cfg)
    pool = _make_pool(4)
    selected = evaluator.select_examples(pool, "test query")
    prompt = evaluator.build_prompt(selected, "test query")
    assert isinstance(prompt, str)
    assert len(prompt) <= 500


# ---------------------------------------------------------------------------
# 13. ICLEvaluator.evaluate_sample returns correct keys with valid probs
# ---------------------------------------------------------------------------


def test_icl_evaluator_evaluate_sample_keys_and_values():
    model = _make_model()
    cfg = ICLConfig(n_shots=2)
    evaluator = ICLEvaluator(model, _encode, _decode, cfg)
    pool = _make_pool(4)
    selected = evaluator.select_examples(pool, "hello")
    prompt = evaluator.build_prompt(selected, "hello")
    result = evaluator.evaluate_sample(prompt)

    assert "top1_prob" in result
    assert "entropy" in result
    assert 0.0 <= result["top1_prob"] <= 1.0
    assert result["entropy"] >= 0.0
