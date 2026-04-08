"""Tests for speculative rejection sampling."""
import torch
import pytest

from src.inference.speculative_rejection import (
    SpeculativeRejectionConfig,
    RejectionStats,
    log_prob_quality_score,
    nucleus_sample_with_logit,
    speculative_rejection_generate,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(n_layers=2, d_model=64, vocab_size=256, max_seq_len=32):
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )
    return AureliusTransformer(cfg)


def _prompt(length=4, vocab_size=256):
    return torch.randint(0, vocab_size, (1, length))


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = SpeculativeRejectionConfig()
    assert cfg.quality_threshold == 0.3
    assert cfg.max_rejections_per_step == 5


# ---------------------------------------------------------------------------
# 2. log_prob_quality_score range
# ---------------------------------------------------------------------------

def test_log_prob_quality_score_range():
    torch.manual_seed(1)
    logits = torch.randn(256)
    for token in [0, 42, 127, 255]:
        score = log_prob_quality_score(logits, token, temperature=1.0)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"score out of [0,1]: {score}"


# ---------------------------------------------------------------------------
# 3. nucleus_sample_with_logit returns tuple
# ---------------------------------------------------------------------------

def test_nucleus_sample_returns_tuple():
    logits = torch.randn(256)
    result = nucleus_sample_with_logit(logits, top_p=0.9, temperature=1.0)
    assert isinstance(result, tuple)
    assert len(result) == 2
    token_id, log_p = result
    assert isinstance(token_id, int)
    assert isinstance(log_p, float)


# ---------------------------------------------------------------------------
# 4. nucleus_sample_with_logit valid token
# ---------------------------------------------------------------------------

def test_nucleus_sample_valid_token():
    vocab_size = 256
    logits = torch.randn(vocab_size)
    for _ in range(20):
        token_id, _ = nucleus_sample_with_logit(logits, top_p=0.9, temperature=1.0)
        assert 0 <= token_id < vocab_size


# ---------------------------------------------------------------------------
# 5. generate returns tensor and stats
# ---------------------------------------------------------------------------

def test_generate_returns_tensor_and_stats():
    model = _make_model()
    cfg = SpeculativeRejectionConfig(max_new_tokens=4, quality_threshold=0.0)
    prompt = _prompt()
    out, stats = speculative_rejection_generate(model, prompt, cfg)
    assert isinstance(out, torch.Tensor)
    assert isinstance(stats, RejectionStats)


# ---------------------------------------------------------------------------
# 6. max_new_tokens respected
# ---------------------------------------------------------------------------

def test_generate_max_tokens_respected():
    model = _make_model()
    max_new = 6
    cfg = SpeculativeRejectionConfig(max_new_tokens=max_new, quality_threshold=0.0)
    prompt = _prompt()
    out, stats = speculative_rejection_generate(model, prompt, cfg)
    assert len(out) <= max_new


# ---------------------------------------------------------------------------
# 7. rejection_rate non-negative
# ---------------------------------------------------------------------------

def test_rejection_stats_rate_nonneg():
    stats = RejectionStats(total_steps=10, total_rejections=3, tokens_generated=10)
    assert stats.rejection_rate >= 0.0

    empty = RejectionStats()
    assert empty.rejection_rate == 0.0


# ---------------------------------------------------------------------------
# 8. High threshold causes more rejections than low threshold
# ---------------------------------------------------------------------------

def test_high_threshold_more_rejections():
    torch.manual_seed(7)
    model = _make_model()

    cfg_high = SpeculativeRejectionConfig(
        max_new_tokens=10,
        quality_threshold=1.0,   # always reject (score in [0,1) since log_p < 0)
        max_rejections_per_step=5,
    )
    cfg_low = SpeculativeRejectionConfig(
        max_new_tokens=10,
        quality_threshold=0.0,   # never reject
        max_rejections_per_step=5,
    )

    prompt = _prompt()
    _, stats_high = speculative_rejection_generate(model, prompt, cfg_high)
    _, stats_low = speculative_rejection_generate(model, prompt, cfg_low)

    assert stats_high.total_rejections >= stats_low.total_rejections


# ---------------------------------------------------------------------------
# 9. Custom quality_fn is called and affects behavior
# ---------------------------------------------------------------------------

def test_custom_quality_fn():
    model = _make_model()
    call_log: list[int] = []

    def always_reject_fn(partial_seq, token, logits):
        call_log.append(token)
        return 0.0  # always below threshold

    cfg = SpeculativeRejectionConfig(
        max_new_tokens=3,
        quality_threshold=0.5,
        max_rejections_per_step=2,
    )
    prompt = _prompt()
    _, stats = speculative_rejection_generate(model, prompt, cfg, quality_fn=always_reject_fn)

    # The custom fn should have been called — call_log must be non-empty
    assert len(call_log) > 0
    # With always-reject fn and max_rejections=2, each step gets 2 rejections + final accept
    assert stats.total_rejections > 0


# ---------------------------------------------------------------------------
# 10. Deterministic output with very low temperature
# ---------------------------------------------------------------------------

def test_generate_deterministic_with_temp_0():
    model = _make_model()
    cfg = SpeculativeRejectionConfig(
        max_new_tokens=8,
        temperature=0.001,
        quality_threshold=0.0,  # never reject so only temperature matters
    )
    prompt = _prompt(length=4)

    torch.manual_seed(42)
    out1, _ = speculative_rejection_generate(model, prompt, cfg)
    torch.manual_seed(42)
    out2, _ = speculative_rejection_generate(model, prompt, cfg)

    assert torch.equal(out1, out2), "Outputs should be identical with the same seed and near-zero temperature"
