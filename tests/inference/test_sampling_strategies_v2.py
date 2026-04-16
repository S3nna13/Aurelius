"""Tests for src/inference/sampling_strategies_v2.py.

Uses a small vocabulary (VOCAB=32) so tests run quickly even on CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.inference.sampling_strategies_v2 import (
    SamplingConfig,
    SamplerPipeline,
    apply_min_p,
    apply_repetition_penalty_sampling,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    apply_typical_p,
    sample_token,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 32  # tiny vocab keeps tests fast


def make_logits(seed: int = 42, vocab: int = VOCAB) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(vocab)


def make_uniform(vocab: int = VOCAB) -> torch.Tensor:
    return torch.zeros(vocab)


# ===========================================================================
# 1. SamplingConfig defaults
# ===========================================================================

def test_sampling_config_defaults():
    cfg = SamplingConfig()
    assert cfg.temperature == 1.0
    assert cfg.top_k == 0
    assert cfg.top_p == 1.0
    assert cfg.min_p == 0.0
    assert cfg.typical_p == 1.0
    assert cfg.repetition_penalty == 1.0
    assert cfg.do_sample is True


# ===========================================================================
# 2. apply_temperature divides logits correctly
# ===========================================================================

def test_apply_temperature_divides_logits():
    logits = make_logits()
    t = 2.5
    result = apply_temperature(logits, t)
    torch.testing.assert_close(result, logits / t)


# ===========================================================================
# 3. apply_temperature raises ValueError on temperature <= 0
# ===========================================================================

@pytest.mark.parametrize("bad_temp", [0.0, -1.0, -0.001])
def test_apply_temperature_raises_on_nonpositive(bad_temp):
    logits = make_logits()
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(logits, bad_temp)


# ===========================================================================
# 4. apply_top_k keeps exactly k tokens (rest -inf)
# ===========================================================================

def test_apply_top_k_keeps_k_tokens():
    logits = make_logits()
    k = 5
    result = apply_top_k(logits, k)
    finite_count = torch.isfinite(result).sum().item()
    assert finite_count == k, f"expected {k} finite tokens, got {finite_count}"


# ===========================================================================
# 5. apply_top_k is a no-op when k=0
# ===========================================================================

def test_apply_top_k_noop_when_k_zero():
    logits = make_logits()
    result = apply_top_k(logits, k=0)
    torch.testing.assert_close(result, logits)


# ===========================================================================
# 6. apply_top_p: cumulative probability of kept tokens >= p
# ===========================================================================

def test_apply_top_p_cumulative_prob_ge_p():
    logits = make_logits()
    p = 0.85
    result = apply_top_p(logits, p)
    # Recompute probs from original logits restricted to kept set
    original_probs = F.softmax(logits, dim=-1)
    kept_mask = torch.isfinite(result)
    kept_mass = original_probs[kept_mask].sum().item()
    assert kept_mass >= p - 1e-5, f"kept mass {kept_mass:.4f} < p={p}"


# ===========================================================================
# 7. apply_top_p: at least one token kept
# ===========================================================================

def test_apply_top_p_at_least_one_token():
    logits = make_logits()
    # Even with p→0 we keep the top token
    result = apply_top_p(logits, p=1e-9)
    assert torch.isfinite(result).any(), "top_p must keep at least one token"


# ===========================================================================
# 8. apply_min_p: at least one token kept
# ===========================================================================

def test_apply_min_p_at_least_one_token():
    logits = make_logits()
    # min_p=1.0 would kill everything — implementation must preserve top token
    result = apply_min_p(logits, min_p=1.0)
    assert torch.isfinite(result).any(), "min_p must keep at least one token"


# ===========================================================================
# 9. apply_min_p filters tokens below threshold
# ===========================================================================

def test_apply_min_p_filters_below_threshold():
    logits = make_logits()
    min_p = 0.4
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max().item()
    threshold = min_p * max_prob

    result = apply_min_p(logits, min_p)
    # Every masked-out token must have had prob < threshold
    for i in range(VOCAB):
        if not torch.isfinite(result[i]):
            assert probs[i].item() < threshold + 1e-6


# ===========================================================================
# 10. apply_typical_p: at least one token kept
# ===========================================================================

def test_apply_typical_p_at_least_one_token():
    logits = make_logits()
    result = apply_typical_p(logits, mass=1e-9)
    assert torch.isfinite(result).any(), "typical_p must keep at least one token"


# ===========================================================================
# 11. apply_typical_p: output shape matches input shape
# ===========================================================================

def test_apply_typical_p_output_shape():
    logits = make_logits()
    result = apply_typical_p(logits, mass=0.5)
    assert result.shape == logits.shape


# ===========================================================================
# 12. apply_repetition_penalty_sampling penalizes past tokens
# ===========================================================================

def test_apply_repetition_penalty_sampling_penalizes_past_tokens():
    logits = torch.zeros(VOCAB)
    logits[3] = 3.0    # positive logit
    logits[7] = -2.0   # negative logit
    past = torch.tensor([3, 7])
    penalty = 2.0

    result = apply_repetition_penalty_sampling(logits, past, penalty)

    # Positive logit divided by penalty
    assert abs(result[3].item() - 3.0 / penalty) < 1e-5
    # Negative logit multiplied by penalty (more negative)
    assert abs(result[7].item() - (-2.0 * penalty)) < 1e-5
    # Token 0 (not in past) unchanged
    assert result[0].item() == 0.0


# ===========================================================================
# 13. sample_token: returns valid int in [0, vocab)
# ===========================================================================

def test_sample_token_returns_valid_int():
    logits = make_logits()
    cfg = SamplingConfig()
    token = sample_token(logits, cfg)
    assert isinstance(token, int)
    assert 0 <= token < VOCAB


# ===========================================================================
# 14. sample_token greedy (do_sample=False) returns argmax
# ===========================================================================

def test_sample_token_greedy_returns_argmax():
    logits = make_logits()
    cfg = SamplingConfig(do_sample=False)
    token = sample_token(logits, cfg)
    expected = int(logits.argmax().item())
    assert token == expected, f"greedy token {token} != argmax {expected}"


# ===========================================================================
# 15. SamplerPipeline.apply_filters: output shape matches input shape
# ===========================================================================

def test_sampler_pipeline_apply_filters_shape():
    logits = make_logits()
    cfg = SamplingConfig(top_k=8, top_p=0.9, temperature=0.7)
    pipeline = SamplerPipeline(cfg)
    result = pipeline.apply_filters(logits)
    assert result.shape == logits.shape


# ===========================================================================
# 16. SamplerPipeline.batch_sample returns (B,) tensor
# ===========================================================================

def test_sampler_pipeline_batch_sample_shape():
    B = 6
    torch.manual_seed(99)
    logits = torch.randn(B, VOCAB)
    cfg = SamplingConfig(top_k=10)
    pipeline = SamplerPipeline(cfg)
    result = pipeline.batch_sample(logits)
    assert result.shape == (B,), f"expected ({B},), got {result.shape}"
    assert result.dtype == torch.long
    # All token ids must be within vocab range
    assert (result >= 0).all() and (result < VOCAB).all()


# ===========================================================================
# 17. SamplerPipeline.sample returns int
# ===========================================================================

def test_sampler_pipeline_sample_returns_int():
    logits = make_logits()
    cfg = SamplingConfig()
    pipeline = SamplerPipeline(cfg)
    token = pipeline.sample(logits)
    assert isinstance(token, int)
    assert 0 <= token < VOCAB


# ===========================================================================
# 18. SamplerPipeline greedy is deterministic
# ===========================================================================

def test_sampler_pipeline_greedy_deterministic():
    logits = make_logits()
    cfg = SamplingConfig(do_sample=False)
    pipeline = SamplerPipeline(cfg)
    t1 = pipeline.sample(logits)
    t2 = pipeline.sample(logits)
    assert t1 == t2


# ===========================================================================
# 19. apply_top_k clamped when k > vocab
# ===========================================================================

def test_apply_top_k_clamps_to_vocab():
    logits = make_logits()
    # k larger than vocab should keep all tokens
    result = apply_top_k(logits, k=VOCAB * 10)
    assert torch.isfinite(result).sum().item() == VOCAB


# ===========================================================================
# 20. sample_token with all filters combined stays in vocab range
# ===========================================================================

def test_sample_token_all_filters_combined():
    logits = make_logits(seed=7)
    cfg = SamplingConfig(
        temperature=0.8,
        top_k=10,
        top_p=0.9,
        min_p=0.05,
        typical_p=0.95,
        repetition_penalty=1.3,
        do_sample=True,
    )
    past = torch.tensor([0, 2, 5])
    for _ in range(20):
        token = sample_token(logits, cfg, past_ids=past)
        assert 0 <= token < VOCAB
