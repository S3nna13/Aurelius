"""Tests for generation_eval module."""
import pytest
import torch

from src.eval.generation_eval import (
    GenerationEvalConfig,
    GenerationEvaluator,
    apply_repetition_penalty,
    compute_distinct_n,
    compute_length_stats,
    compute_rouge_l,
    greedy_generate,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(42)
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
    return AureliusTransformer(cfg)


def encode_fn(text: str) -> list:
    """Simple byte-level encoder (clamped to vocab_size=256)."""
    return [b % 256 for b in text.encode("utf-8")][:32]


def decode_fn(ids: list) -> str:
    """Simple byte-level decoder — best effort."""
    bs = bytes([i % 256 for i in ids])
    return bs.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Test GenerationEvalConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = GenerationEvalConfig()
    assert cfg.max_new_tokens == 64
    assert cfg.temperature == 1.0
    assert cfg.do_sample is False
    assert cfg.repetition_penalty == 1.0
    assert cfg.n_gram_n == 4


# ---------------------------------------------------------------------------
# Test greedy_generate output shape
# ---------------------------------------------------------------------------

def test_greedy_generate_shape(small_model):
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = greedy_generate(small_model, input_ids, max_new_tokens=4)
    assert out.shape == (1, 4)


def test_greedy_generate_with_penalty(small_model):
    input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    out = greedy_generate(small_model, input_ids, max_new_tokens=4, repetition_penalty=1.3)
    assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# Test compute_rouge_l
# ---------------------------------------------------------------------------

def test_rouge_l_identical():
    score = compute_rouge_l("hello world", "hello world")
    assert score == pytest.approx(1.0)


def test_rouge_l_no_overlap():
    score = compute_rouge_l("foo bar baz", "alpha beta gamma")
    assert score == pytest.approx(0.0)


def test_rouge_l_partial_overlap():
    score = compute_rouge_l("the cat sat on the mat", "the cat sat")
    assert 0.0 < score < 1.0


def test_rouge_l_empty_hypothesis():
    score = compute_rouge_l("", "some reference text")
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test compute_distinct_n
# ---------------------------------------------------------------------------

def test_distinct_n_all_unique():
    # 4 unique tokens, bigrams: (1,2),(2,3),(3,4) — all unique
    ids = torch.tensor([1, 2, 3, 4])
    score = compute_distinct_n(ids, n=2)
    assert score == pytest.approx(1.0)


def test_distinct_n_all_same():
    # All tokens identical → only 1 unique bigram out of many
    ids = torch.tensor([5] * 20)
    score = compute_distinct_n(ids, n=2)
    # 1 unique out of 19 total
    assert score == pytest.approx(1 / 19)


# ---------------------------------------------------------------------------
# Test compute_length_stats
# ---------------------------------------------------------------------------

def test_length_stats_keys_present():
    texts = ["hello world", "foo bar baz qux", "one"]
    stats = compute_length_stats(texts)
    assert "mean_len" in stats
    assert "min_len" in stats
    assert "max_len" in stats
    assert "std_len" in stats


def test_length_stats_values():
    texts = ["a b", "c d e f"]  # lengths 2 and 4
    stats = compute_length_stats(texts)
    assert stats["mean_len"] == pytest.approx(3.0)
    assert stats["min_len"] == 2
    assert stats["max_len"] == 4


# ---------------------------------------------------------------------------
# Test apply_repetition_penalty
# ---------------------------------------------------------------------------

def test_apply_repetition_penalty_penalizes_repeated():
    # logits for tokens 0..7, token 3 has positive score, token 5 has negative score
    logits = torch.zeros(1, 8)
    logits[0, 3] = 2.0   # positive — should be divided
    logits[0, 5] = -2.0  # negative — should be multiplied (more negative)

    generated_ids = torch.tensor([[3, 5]])
    penalty = 2.0

    result = apply_repetition_penalty(logits, generated_ids, penalty)

    assert result[0, 3].item() == pytest.approx(1.0)   # 2.0 / 2.0
    assert result[0, 5].item() == pytest.approx(-4.0)  # -2.0 * 2.0
    # Unpenalized token unchanged
    assert result[0, 0].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test GenerationEvaluator
# ---------------------------------------------------------------------------

def test_evaluate_sample_keys(small_model):
    cfg = GenerationEvalConfig(max_new_tokens=4, n_gram_n=2)
    evaluator = GenerationEvaluator(small_model, encode_fn, decode_fn, cfg)
    result = evaluator.evaluate_sample("hello world", "hello world test")
    assert "rouge_l" in result
    assert "distinct_n" in result
    assert "length" in result


def test_evaluate_batch_keys(small_model):
    cfg = GenerationEvalConfig(max_new_tokens=4, n_gram_n=2)
    evaluator = GenerationEvaluator(small_model, encode_fn, decode_fn, cfg)
    prompts = ["hello", "world"]
    references = ["hello there", "world peace"]
    result = evaluator.evaluate_batch(prompts, references)
    assert "mean_rouge_l" in result
    assert "mean_distinct_n" in result
    assert "mean_len" in result
    assert "min_len" in result
    assert "max_len" in result
    assert "std_len" in result
