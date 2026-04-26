"""Tests for cross-lingual transfer evaluation module."""

from __future__ import annotations

import pytest
import torch

from src.eval.crosslingual_eval import (
    LANG_TEMPLATES,
    CrosslingualConfig,
    CrosslingualEvaluator,
    LanguagePair,
    TransferResult,
    compute_language_similarity,
    compute_perplexity,
    compute_token_accuracy,
    compute_transfer_matrix,
    evaluate_transfer,
    generate_synthetic_parallel_data,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Tiny model fixture
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


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


# Byte-level tokenizer: encode string to list of ints, capped at 256
def encode(s):
    return list(s.encode("utf-8", errors="replace"))[:256]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_config_defaults():
    """CrosslingualConfig has 5 languages and metric='perplexity'."""
    cfg = CrosslingualConfig()
    assert len(cfg.languages) == 5
    assert cfg.metric == "perplexity"


def test_lang_templates_coverage():
    """LANG_TEMPLATES covers all 5 default languages."""
    cfg = CrosslingualConfig()
    for lang in cfg.languages:
        assert lang in LANG_TEMPLATES, f"Missing language: {lang}"
        assert len(LANG_TEMPLATES[lang]) > 0


def test_generate_parallel_data_count():
    """generate_synthetic_parallel_data returns exactly n_samples LanguagePairs."""
    n = 4
    pairs = generate_synthetic_parallel_data(n_samples=n, languages=["en", "fr"])
    assert len(pairs) == n
    for p in pairs:
        assert isinstance(p, LanguagePair)


def test_generate_parallel_data_langs():
    """All generated pairs have valid source and target languages."""
    languages = ["en", "fr", "de"]
    pairs = generate_synthetic_parallel_data(n_samples=4, languages=languages)
    for p in pairs:
        assert p.source_lang in languages
        assert p.target_lang in languages


def test_compute_language_similarity_same():
    """Same language gives similarity of 1.0."""
    for lang in ["en", "fr", "de", "es", "zh"]:
        assert compute_language_similarity(lang, lang) == 1.0


def test_compute_language_similarity_en_fr():
    """en-fr similarity should be greater than 0.5 (European related)."""
    sim = compute_language_similarity("en", "fr")
    assert sim > 0.5, f"Expected > 0.5 for en-fr, got {sim}"


def test_compute_language_similarity_en_zh():
    """en-zh similarity should be less than 0.5 (distant languages)."""
    sim = compute_language_similarity("en", "zh")
    assert sim < 0.5, f"Expected < 0.5 for en-zh, got {sim}"


def test_compute_perplexity_positive(tiny_model):
    """compute_perplexity returns a value > 1.0 for normal text."""
    texts = ["The model processes tokens.", "Output: 42."]
    ppl = compute_perplexity(tiny_model, encode, texts, max_seq_len=128)
    assert ppl > 1.0, f"Expected perplexity > 1.0, got {ppl}"
    assert ppl < float("inf"), "Perplexity should be finite"


def test_compute_token_accuracy_range(tiny_model):
    """compute_token_accuracy returns a value in [0, 1]."""
    texts = ["The model processes tokens.", "Output: 42."]
    acc = compute_token_accuracy(tiny_model, encode, texts, max_seq_len=128)
    assert 0.0 <= acc <= 1.0, f"Expected accuracy in [0, 1], got {acc}"


def test_evaluate_transfer_returns_results(tiny_model):
    """evaluate_transfer returns a list of TransferResult objects."""
    pairs = generate_synthetic_parallel_data(n_samples=4, languages=["en", "fr"])
    cfg = CrosslingualConfig(n_eval_samples=2, max_seq_len=64)
    results = evaluate_transfer(tiny_model, encode, pairs, cfg)
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, TransferResult) for r in results)


def test_evaluate_transfer_result_fields(tiny_model):
    """TransferResult objects have properly populated numeric fields."""
    pairs = generate_synthetic_parallel_data(n_samples=4, languages=["en", "fr"])
    cfg = CrosslingualConfig(n_eval_samples=2, max_seq_len=64)
    results = evaluate_transfer(tiny_model, encode, pairs, cfg)
    for r in results:
        assert isinstance(r.source_score, float)
        assert isinstance(r.target_score, float)
        assert isinstance(r.transfer_gap, float)
        assert r.source_score > 0.0
        assert r.target_score > 0.0
        assert abs(r.transfer_gap - (r.target_score - r.source_score)) < 1e-5


def test_compute_transfer_matrix_keys(tiny_model):
    """compute_transfer_matrix returns dict with (str, str) tuple keys."""
    pairs = generate_synthetic_parallel_data(n_samples=4, languages=["en", "fr"])
    cfg = CrosslingualConfig(n_eval_samples=2, max_seq_len=64)
    results = evaluate_transfer(tiny_model, encode, pairs, cfg)
    matrix = compute_transfer_matrix(results, languages=["en", "fr"])
    assert isinstance(matrix, dict)
    for key in matrix.keys():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(key[0], str)
        assert isinstance(key[1], str)


def test_evaluator_evaluate_keys(tiny_model):
    """CrosslingualEvaluator.evaluate returns dict with all required keys."""
    cfg = CrosslingualConfig(
        languages=["en", "fr"],
        n_eval_samples=2,
        max_seq_len=64,
    )
    evaluator = CrosslingualEvaluator(tiny_model, encode, cfg)
    pairs = generate_synthetic_parallel_data(n_samples=4, languages=["en", "fr"])
    result = evaluator.evaluate(pairs=pairs)
    required_keys = {
        "mean_perplexity_all",
        "best_transfer_pair",
        "worst_transfer_pair",
        "mean_transfer_gap",
        "n_pairs_evaluated",
    }
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_evaluator_find_best_transfer(tiny_model):
    """find_best_transfer_pair returns a non-empty 'src>tgt' format string."""
    cfg = CrosslingualConfig(
        languages=["en", "fr", "de"],
        n_eval_samples=2,
        max_seq_len=64,
    )
    evaluator = CrosslingualEvaluator(tiny_model, encode, cfg)
    pairs = generate_synthetic_parallel_data(n_samples=4, languages=["en", "fr", "de"])
    results = evaluate_transfer(tiny_model, encode, pairs, cfg)
    best = evaluator.find_best_transfer_pair(results)
    assert isinstance(best, str)
    assert len(best) > 0
    assert ">" in best, f"Expected 'src>tgt' format, got: {best}"


def test_zero_shot_transfer_score_range(tiny_model):
    """compute_zero_shot_transfer_score returns a positive float (perplexity ratio)."""
    cfg = CrosslingualConfig(
        languages=["en", "fr"],
        n_eval_samples=2,
        max_seq_len=64,
    )
    evaluator = CrosslingualEvaluator(tiny_model, encode, cfg)
    pairs = generate_synthetic_parallel_data(n_samples=4, languages=["en", "fr"])
    results = evaluate_transfer(tiny_model, encode, pairs, cfg)
    score = evaluator.compute_zero_shot_transfer_score(results)
    assert score > 0.0, f"Expected score > 0, got {score}"
