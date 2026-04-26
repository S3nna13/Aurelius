"""Tests for src/inference/sampling_strategies.py."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.inference.sampling_strategies import (
    SamplingConfig,
    SamplingDecoder,
    apply_min_p,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    apply_typical_p,
    sample_token,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64
SEED = 0


def make_logits(seed: int = SEED) -> torch.Tensor:
    """Return a deterministic (VOCAB_SIZE,) float logit tensor."""
    torch.manual_seed(seed)
    return torch.randn(VOCAB_SIZE)


def make_uniform_logits() -> torch.Tensor:
    """Return perfectly uniform logits."""
    return torch.zeros(VOCAB_SIZE)


def tiny_model_fn(input_ids: torch.Tensor) -> torch.Tensor:
    """Mock model: returns uniform logits for each position."""
    T = input_ids.shape[-1]
    return torch.zeros(1, T, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# 1. SamplingConfig defaults
# ---------------------------------------------------------------------------


def test_sampling_config_defaults():
    cfg = SamplingConfig()
    assert cfg.temperature == 1.0
    assert cfg.top_k == 0
    assert cfg.top_p == 1.0
    assert cfg.min_p == 0.0
    assert cfg.repetition_penalty == 1.0
    assert cfg.typical_p == 1.0


# ---------------------------------------------------------------------------
# 2. apply_temperature divides logits
# ---------------------------------------------------------------------------


def test_apply_temperature_divides_logits():
    logits = make_logits()
    t = 2.0
    result = apply_temperature(logits, t)
    torch.testing.assert_close(result, logits / t)


# ---------------------------------------------------------------------------
# 3. High temperature flattens distribution
# ---------------------------------------------------------------------------


def test_apply_temperature_high_flattens_distribution():
    """At very high temperature the softmax distribution should be nearly uniform."""
    logits = make_logits()
    hot = apply_temperature(logits, temperature=1e6)
    probs = F.softmax(hot, dim=-1)
    expected = torch.full((VOCAB_SIZE,), 1.0 / VOCAB_SIZE)
    torch.testing.assert_close(probs, expected, atol=1e-3, rtol=0)


# ---------------------------------------------------------------------------
# 4. temperature=0 returns argmax (one-hot-like)
# ---------------------------------------------------------------------------


def test_apply_temperature_zero_argmax():
    logits = make_logits()
    result = apply_temperature(logits, temperature=0.0)
    argmax = logits.argmax().item()
    # All positions except argmax should be -inf
    assert result[argmax].item() == 0.0
    mask = torch.ones(VOCAB_SIZE, dtype=torch.bool)
    mask[argmax] = False
    assert torch.all(result[mask] == float("-inf"))


# ---------------------------------------------------------------------------
# 5. apply_top_k keeps exactly k tokens (rest are -inf)
# ---------------------------------------------------------------------------


def test_apply_top_k_keeps_k_tokens():
    logits = make_logits()
    k = 10
    result = apply_top_k(logits, k)
    finite_count = torch.isfinite(result).sum().item()
    assert finite_count == k


# ---------------------------------------------------------------------------
# 6. top_k=0 leaves logits unchanged
# ---------------------------------------------------------------------------


def test_apply_top_k_zero_unchanged():
    logits = make_logits()
    result = apply_top_k(logits, k=0)
    torch.testing.assert_close(result, logits)


# ---------------------------------------------------------------------------
# 7. apply_top_p preserves enough probability mass
# ---------------------------------------------------------------------------


def test_apply_top_p_preserves_mass():
    logits = make_logits()
    p = 0.9
    result = apply_top_p(logits, p)
    # Probability mass of kept tokens must be >= p
    probs = F.softmax(result, dim=-1)
    kept_mass = probs[torch.isfinite(result)].sum().item()
    assert kept_mass >= p - 1e-5


# ---------------------------------------------------------------------------
# 8. top_p=1.0 leaves logits unchanged
# ---------------------------------------------------------------------------


def test_apply_top_p_one_unchanged():
    logits = make_logits()
    result = apply_top_p(logits, p=1.0)
    torch.testing.assert_close(result, logits)


# ---------------------------------------------------------------------------
# 9. apply_min_p filters low-probability tokens
# ---------------------------------------------------------------------------


def test_apply_min_p_filters_low_prob():
    """With a large min_p only the highest-probability tokens survive."""
    logits = make_logits()
    min_p = 0.5  # tokens with prob < 0.5 * max_prob are removed
    result = apply_min_p(logits, min_p)
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max().item()
    threshold = min_p * max_prob

    for i in range(VOCAB_SIZE):
        if probs[i].item() < threshold:
            assert result[i].item() == float("-inf"), (
                f"token {i} with prob {probs[i].item():.4f} < threshold "
                f"{threshold:.4f} should be -inf"
            )


# ---------------------------------------------------------------------------
# 10. apply_repetition_penalty reduces repeated token logits
# ---------------------------------------------------------------------------


def test_apply_repetition_penalty_reduces_logits():
    logits = torch.zeros(VOCAB_SIZE)
    # Make token 5 have a positive logit
    logits[5] = 2.0
    # Make token 10 have a negative logit
    logits[10] = -1.0
    input_ids = torch.tensor([5, 10])
    penalty = 2.0
    result = apply_repetition_penalty(logits, input_ids, penalty)

    # Positive logit should be divided by penalty
    assert abs(result[5].item() - 2.0 / penalty) < 1e-5
    # Negative logit should be multiplied by penalty (more negative)
    assert abs(result[10].item() - (-1.0 * penalty)) < 1e-5
    # Unaffected token should be unchanged
    assert result[0].item() == 0.0


# ---------------------------------------------------------------------------
# 11. repetition_penalty=1.0 leaves logits unchanged
# ---------------------------------------------------------------------------


def test_apply_repetition_penalty_one_unchanged():
    logits = make_logits()
    input_ids = torch.arange(VOCAB_SIZE // 2)
    result = apply_repetition_penalty(logits, input_ids, penalty=1.0)
    torch.testing.assert_close(result, logits)


# ---------------------------------------------------------------------------
# 12. sample_token returns scalar in vocab range
# ---------------------------------------------------------------------------


def test_sample_token_returns_scalar_in_vocab_range():
    logits = make_logits()
    cfg = SamplingConfig()
    token = sample_token(logits, cfg)
    assert token.shape == (1,)
    assert 0 <= token.item() < VOCAB_SIZE


# ---------------------------------------------------------------------------
# 13. decode output length equals prompt + max_new_tokens
# ---------------------------------------------------------------------------


def test_decode_output_length():
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    max_new = 5
    cfg = SamplingConfig()
    decoder = SamplingDecoder(model_fn=tiny_model_fn, config=cfg)
    output = decoder.decode(prompt, max_new_tokens=max_new)
    assert output.shape == (len(prompt) + max_new,)


# ---------------------------------------------------------------------------
# 14. decode_batch output shape is (n_sequences, T)
# ---------------------------------------------------------------------------


def test_decode_batch_output_shape():
    prompt = torch.tensor([0, 1], dtype=torch.long)
    n_seq = 4
    max_new = 6
    cfg = SamplingConfig()
    decoder = SamplingDecoder(model_fn=tiny_model_fn, config=cfg)
    batch = decoder.decode_batch(prompt, n_sequences=n_seq, max_new_tokens=max_new)
    expected_T = len(prompt) + max_new
    assert batch.shape == (n_seq, expected_T)


# ---------------------------------------------------------------------------
# 15. Applying all filters together still yields a valid distribution
# ---------------------------------------------------------------------------


def test_all_filters_produce_valid_distribution():
    """Combining temperature + top_k + top_p + min_p should not produce all-inf."""
    torch.manual_seed(7)
    logits = torch.randn(VOCAB_SIZE)
    cfg = SamplingConfig(
        temperature=0.8,
        top_k=20,
        top_p=0.95,
        min_p=0.01,
        repetition_penalty=1.2,
    )
    input_ids = torch.tensor([0, 3, 7])
    # sample_token should not raise or produce NaN
    token = sample_token(logits, cfg, input_ids=input_ids)
    assert 0 <= token.item() < VOCAB_SIZE


# ---------------------------------------------------------------------------
# 16. apply_top_p with very small p keeps at least one token
# ---------------------------------------------------------------------------


def test_apply_top_p_very_small_p_keeps_at_least_one():
    logits = make_logits()
    result = apply_top_p(logits, p=0.01)
    assert torch.isfinite(result).any()


# ---------------------------------------------------------------------------
# 17. apply_typical_p with mass=1.0 is a no-op
# ---------------------------------------------------------------------------


def test_apply_typical_p_one_unchanged():
    logits = make_logits()
    result = apply_typical_p(logits, mass=1.0)
    torch.testing.assert_close(result, logits)


# ---------------------------------------------------------------------------
# 18. apply_typical_p with small mass keeps fewer tokens
# ---------------------------------------------------------------------------


def test_apply_typical_p_small_mass_reduces_tokens():
    logits = make_logits()
    result_full = apply_typical_p(logits, mass=1.0)
    result_small = apply_typical_p(logits, mass=0.5)
    kept_full = torch.isfinite(result_full).sum().item()
    kept_small = torch.isfinite(result_small).sum().item()
    assert kept_small <= kept_full
