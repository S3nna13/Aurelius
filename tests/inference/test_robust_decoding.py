"""Tests for Whisper-inspired robust decoding with temperature fallback cascade."""

import torch

from src.inference.robust_decoding import (
    GenerationResult,
    RobustDecoder,
    RobustDecodingConfig,
    compute_compression_ratio,
    compute_mean_logprob,
    generate_with_fallback,
    no_repeat_ngram_logit_processor,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


def _make_model(n_layers=2, d_model=64):
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


# ── compression ratio tests ──────────────────────────────────────────────────


def test_compression_ratio_repetitive():
    """Highly repetitive text should have a very low compression ratio."""
    text = "a" * 200
    ratio = compute_compression_ratio(text)
    assert ratio < 0.5, f"Expected low ratio for repetitive text, got {ratio}"


def test_compression_ratio_diverse():
    """Diverse text should have a higher compression ratio than repetitive text."""
    repetitive = "ab" * 100
    diverse = "The quick brown fox jumps over the lazy dog. " * 5
    r_rep = compute_compression_ratio(repetitive)
    r_div = compute_compression_ratio(diverse)
    assert r_div > r_rep, f"Diverse {r_div} should be > repetitive {r_rep}"


# ── mean log-prob tests ──────────────────────────────────────────────────────


def test_compute_mean_logprob_negative():
    """Mean log-prob should be a negative float."""
    model = _make_model()
    input_ids = torch.randint(0, 256, (1, 8))
    generated_ids = torch.randint(0, 256, (1, 4))
    lp = compute_mean_logprob(model, input_ids, generated_ids)
    assert isinstance(lp, float)
    assert lp < 0.0, f"Expected negative log-prob, got {lp}"


def test_compute_mean_logprob_shape():
    """compute_mean_logprob should return a scalar float, not a tensor."""
    model = _make_model()
    input_ids = torch.randint(0, 256, (1, 6))
    generated_ids = torch.randint(0, 256, (1, 3))
    lp = compute_mean_logprob(model, input_ids, generated_ids)
    assert isinstance(lp, float)


# ── no-repeat n-gram logit processor tests ───────────────────────────────────


def test_no_repeat_ngram_blocks_repeated():
    """Token that would form a repeated n-gram should get -inf logit."""
    # input sequence [1, 2, 3, 1, 2]: prefix [1,2] already followed by 3
    input_ids = torch.tensor([[1, 2, 3, 1, 2]])  # (1, 5)
    logits = torch.zeros(1, 256)
    result = no_repeat_ngram_logit_processor(input_ids, logits, ngram_size=3)
    assert result[0, 3] == float("-inf"), "Token 3 should be blocked"


def test_no_repeat_ngram_allows_new():
    """Tokens that don't form a repeated n-gram should NOT be blocked."""
    input_ids = torch.tensor([[1, 2, 3, 1, 2]])
    logits = torch.zeros(1, 256)
    result = no_repeat_ngram_logit_processor(input_ids, logits, ngram_size=3)
    # Token 5 has never followed [1,2] before — should be allowed
    assert result[0, 5] != float("-inf"), "Novel token should not be blocked"


# ── generate_with_fallback tests ─────────────────────────────────────────────


def test_generate_with_fallback_returns_result():
    """generate_with_fallback should return a GenerationResult."""
    model = _make_model()
    input_ids = torch.randint(0, 256, (1, 4))
    cfg = RobustDecodingConfig(max_new_tokens=8)
    result = generate_with_fallback(model, input_ids, cfg)
    assert isinstance(result, GenerationResult)


def test_generate_with_fallback_is_reliable_field():
    """GenerationResult.is_reliable should be a bool."""
    model = _make_model()
    input_ids = torch.randint(0, 256, (1, 4))
    cfg = RobustDecodingConfig(max_new_tokens=8)
    result = generate_with_fallback(model, input_ids, cfg)
    assert isinstance(result.is_reliable, bool)


def test_generate_with_fallback_temperature_used():
    """temperature_used should be one of the configured temperatures."""
    model = _make_model()
    input_ids = torch.randint(0, 256, (1, 4))
    cfg = RobustDecodingConfig(max_new_tokens=8)
    result = generate_with_fallback(model, input_ids, cfg)
    assert result.temperature_used in cfg.temperatures


# ── GenerationResult field tests ─────────────────────────────────────────────


def test_generation_result_fields():
    """GenerationResult should have all required fields."""
    model = _make_model()
    input_ids = torch.randint(0, 256, (1, 4))
    cfg = RobustDecodingConfig(max_new_tokens=8)
    result = generate_with_fallback(model, input_ids, cfg)
    assert hasattr(result, "token_ids")
    assert hasattr(result, "temperature_used")
    assert hasattr(result, "mean_logprob")
    assert hasattr(result, "compression_ratio")
    assert hasattr(result, "passed_logprob_check")
    assert hasattr(result, "passed_compression_check")
    assert hasattr(result, "is_reliable")
    assert isinstance(result.token_ids, torch.Tensor)
    assert result.token_ids.shape[0] == 1


# ── RobustDecoder tests ───────────────────────────────────────────────────────


def test_robust_decoder_generate():
    """RobustDecoder.generate should return a GenerationResult."""
    model = _make_model()
    cfg = RobustDecodingConfig(max_new_tokens=8)
    decoder = RobustDecoder(model, cfg)
    input_ids = torch.randint(0, 256, (1, 4))
    result = decoder.generate(input_ids)
    assert isinstance(result, GenerationResult)


def test_robust_decoder_batch_generate():
    """RobustDecoder.batch_generate should return a list of correct length."""
    model = _make_model()
    cfg = RobustDecodingConfig(max_new_tokens=8)
    decoder = RobustDecoder(model, cfg)
    inputs = [torch.randint(0, 256, (1, 4)) for _ in range(3)]
    results = decoder.batch_generate(inputs)
    assert isinstance(results, list)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, GenerationResult)
