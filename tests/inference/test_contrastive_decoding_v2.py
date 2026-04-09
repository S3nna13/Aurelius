"""Tests for contrastive_decoding_v2 (Li et al. 2023)."""
from __future__ import annotations

import torch
import pytest

from src.inference.contrastive_decoding_v2 import (
    ContrastiveDecodingConfig,
    compute_cd_score,
    apply_adaptive_plausibility,
    contrastive_sample,
    ContrastiveDecoder,
    measure_repetition,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config() -> AureliusConfig:
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


@pytest.fixture
def expert_model():
    torch.manual_seed(0)
    return AureliusTransformer(_small_config())


@pytest.fixture
def amateur_model():
    torch.manual_seed(1)
    return AureliusTransformer(_small_config())


VOCAB = 256
B = 2
MAX_NEW = 4


# ---------------------------------------------------------------------------
# 1. ContrastiveDecodingConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = ContrastiveDecodingConfig()
    assert cfg.alpha == 0.1
    assert cfg.temperature == 1.0
    assert cfg.max_new_tokens == 64
    assert cfg.top_k == 50


# ---------------------------------------------------------------------------
# 2. compute_cd_score — output shape
# ---------------------------------------------------------------------------

def test_compute_cd_score_shape():
    expert = torch.randn(B, VOCAB)
    amateur = torch.randn(B, VOCAB)
    scores = compute_cd_score(expert, amateur, temperature=1.0)
    assert scores.shape == (B, VOCAB)


# ---------------------------------------------------------------------------
# 3. compute_cd_score — expert > amateur gives positive score for that token
# ---------------------------------------------------------------------------

def test_compute_cd_score_favours_expert_token():
    """Token with much higher probability under expert should have a positive CD score."""
    expert = torch.full((1, VOCAB), -10.0)
    expert[0, 42] = 10.0          # expert strongly prefers token 42

    amateur = torch.full((1, VOCAB), -10.0)
    amateur[0, 42] = -10.0        # amateur does NOT prefer token 42

    scores = compute_cd_score(expert, amateur, temperature=1.0)
    assert scores[0, 42].item() > 0.0


# ---------------------------------------------------------------------------
# 4. apply_adaptive_plausibility — masks low-prob tokens
# ---------------------------------------------------------------------------

def test_apply_plausibility_masks_low_prob():
    """Tokens with very low expert probability should be set to -inf."""
    expert = torch.full((1, VOCAB), -100.0)
    expert[0, 0] = 10.0            # only token 0 is plausible
    cd_scores = torch.zeros(1, VOCAB)

    masked = apply_adaptive_plausibility(expert, cd_scores, alpha=0.1)
    # All tokens except 0 should be -inf
    assert masked[0, 0].item() != float("-inf")
    assert masked[0, 1].item() == float("-inf")
    assert masked[0, VOCAB - 1].item() == float("-inf")


# ---------------------------------------------------------------------------
# 5. apply_adaptive_plausibility — keeps high-prob tokens
# ---------------------------------------------------------------------------

def test_apply_plausibility_keeps_high_prob():
    """Tokens with probability near the maximum should NOT be masked."""
    # Uniform expert — all tokens equally plausible
    expert = torch.zeros(1, VOCAB)
    cd_scores = torch.ones(1, VOCAB)

    masked = apply_adaptive_plausibility(expert, cd_scores, alpha=0.1)
    # With uniform probs every token prob == max_prob, so none should be masked
    assert not torch.isinf(masked).any()


# ---------------------------------------------------------------------------
# 6. contrastive_sample — output shape (B,)
# ---------------------------------------------------------------------------

def test_contrastive_sample_shape():
    cfg = ContrastiveDecodingConfig(max_new_tokens=MAX_NEW)
    expert = torch.randn(B, VOCAB)
    amateur = torch.randn(B, VOCAB)
    tokens = contrastive_sample(expert, amateur, cfg)
    assert tokens.shape == (B,)


# ---------------------------------------------------------------------------
# 7. contrastive_sample — output in valid vocab range
# ---------------------------------------------------------------------------

def test_contrastive_sample_valid_range():
    cfg = ContrastiveDecodingConfig(max_new_tokens=MAX_NEW)
    expert = torch.randn(B, VOCAB)
    amateur = torch.randn(B, VOCAB)
    tokens = contrastive_sample(expert, amateur, cfg)
    assert tokens.min().item() >= 0
    assert tokens.max().item() < VOCAB


# ---------------------------------------------------------------------------
# 8. ContrastiveDecoder.generate — output shape (B, max_new_tokens)
# ---------------------------------------------------------------------------

def test_generate_output_shape(expert_model, amateur_model):
    cfg = ContrastiveDecodingConfig(max_new_tokens=MAX_NEW)
    decoder = ContrastiveDecoder(expert_model, amateur_model, cfg)
    input_ids = torch.randint(0, VOCAB, (B, 5))
    generated, _ = decoder.generate(input_ids)
    assert generated.shape == (B, MAX_NEW)


# ---------------------------------------------------------------------------
# 9. ContrastiveDecoder.generate — stats keys present
# ---------------------------------------------------------------------------

def test_generate_stats_keys(expert_model, amateur_model):
    cfg = ContrastiveDecodingConfig(max_new_tokens=MAX_NEW)
    decoder = ContrastiveDecoder(expert_model, amateur_model, cfg)
    input_ids = torch.randint(0, VOCAB, (1, 5))
    _, stats = decoder.generate(input_ids)
    assert "n_tokens" in stats
    assert "mean_cd_score" in stats
    assert stats["n_tokens"] == MAX_NEW
    assert isinstance(stats["mean_cd_score"], float)


# ---------------------------------------------------------------------------
# 10. measure_repetition — all unique → 0.0
# ---------------------------------------------------------------------------

def test_measure_repetition_all_unique():
    ids = torch.arange(32)
    result = measure_repetition(ids, window=16)
    assert result == 0.0


# ---------------------------------------------------------------------------
# 11. measure_repetition — all same token → near 1.0
# ---------------------------------------------------------------------------

def test_measure_repetition_all_same():
    ids = torch.zeros(32, dtype=torch.long)
    result = measure_repetition(ids, window=16)
    # First token has no context, so at most (T-1)/T repeated
    # For T=32 that is 31/32 ≈ 0.969; definitely > 0.9
    assert result > 0.9


# ---------------------------------------------------------------------------
# 12. compute_cd_score — different outputs for temperature 1.0 vs 0.5
# ---------------------------------------------------------------------------

def test_compute_cd_score_temperature_difference():
    torch.manual_seed(42)
    expert = torch.randn(1, VOCAB)
    amateur = torch.randn(1, VOCAB)

    scores_t1 = compute_cd_score(expert, amateur, temperature=1.0)
    scores_t05 = compute_cd_score(expert, amateur, temperature=0.5)

    # The two should not be identical
    assert not torch.allclose(scores_t1, scores_t05)
