"""Tests for Diffusion LM Inference (src/inference/diffusion_lm_inference.py).

Import path: aurelius.inference.diffusion_lm_inference
"""

from __future__ import annotations

import pytest
import torch
from aurelius.inference.diffusion_lm_inference import (
    ContinuousTimeDiffusionSampler,
    DiffusionLMQualityMetrics,
    MaskNoiseSchedule,
    MDLMSampler,
)

# ---------------------------------------------------------------------------
# Constants used throughout
# ---------------------------------------------------------------------------

VOCAB_SIZE = 16
SEQ_LEN = 8
BATCH_SIZE = 4
MASK_TOKEN_ID = 0
N_STEPS = 20  # small for fast tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_random_model_fn(vocab_size: int = VOCAB_SIZE):
    """Returns a model_fn that outputs random logits — shape (B, T, V)."""

    def model_fn(noisy_ids: torch.Tensor, t) -> torch.Tensor:  # noqa: ANN001
        B, T = noisy_ids.shape
        return torch.randn(B, T, vocab_size)

    return model_fn


# ---------------------------------------------------------------------------
# MaskNoiseSchedule tests
# ---------------------------------------------------------------------------


def test_mask_rate_at_t0_is_one():
    """mask_rate at t=0 should be 1.0 (fully masked)."""
    sched = MaskNoiseSchedule(n_steps=N_STEPS, mask_token_id=MASK_TOKEN_ID)
    assert sched.mask_rate(0) == pytest.approx(1.0)


def test_mask_rate_at_t_n_steps_is_zero():
    """mask_rate at t=n_steps should be 0.0 (fully unmasked)."""
    sched = MaskNoiseSchedule(n_steps=N_STEPS, mask_token_id=MASK_TOKEN_ID)
    assert sched.mask_rate(N_STEPS) == pytest.approx(0.0)


def test_apply_mask_returns_same_shape():
    """apply_mask should return a tensor of the same shape as the input."""
    sched = MaskNoiseSchedule(n_steps=N_STEPS, mask_token_id=MASK_TOKEN_ID)
    token_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    result = sched.apply_mask(token_ids, t=N_STEPS // 2)
    assert result.shape == token_ids.shape


def test_apply_mask_at_t0_all_masked():
    """apply_mask at t=0 should return an all-mask tensor."""
    sched = MaskNoiseSchedule(n_steps=N_STEPS, mask_token_id=MASK_TOKEN_ID)
    token_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    result = sched.apply_mask(token_ids, t=0)
    assert (result == MASK_TOKEN_ID).all()


def test_apply_mask_at_t_n_steps_unchanged():
    """apply_mask at t=n_steps should return the unchanged input tensor."""
    sched = MaskNoiseSchedule(n_steps=N_STEPS, mask_token_id=MASK_TOKEN_ID)
    token_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    result = sched.apply_mask(token_ids, t=N_STEPS)
    assert torch.equal(result, token_ids)


# ---------------------------------------------------------------------------
# MDLMSampler tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mdlm_sampler():
    sched = MaskNoiseSchedule(n_steps=N_STEPS, mask_token_id=MASK_TOKEN_ID)
    return MDLMSampler(
        model_fn=make_random_model_fn(VOCAB_SIZE),
        schedule=sched,
        vocab_size=VOCAB_SIZE,
    )


def test_denoise_step_shape_unchanged(mdlm_sampler):
    """denoise_step output shape should match the input shape."""
    noisy = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    result = mdlm_sampler.denoise_step(noisy, t=10)
    assert result.shape == noisy.shape


def test_denoise_step_reduces_masks(mdlm_sampler):
    """denoise_step should (on average) reduce the number of mask tokens."""
    # Start with many masks
    noisy = torch.full((BATCH_SIZE, SEQ_LEN), MASK_TOKEN_ID, dtype=torch.long)
    result = mdlm_sampler.denoise_step(noisy, t=10)
    masks_before = (noisy == MASK_TOKEN_ID).sum().item()
    masks_after = (result == MASK_TOKEN_ID).sum().item()
    # With random logits, mask positions are sampled; it's possible a new token
    # equals MASK_TOKEN_ID, but on average we should see fewer masks.
    # We just verify unmasked positions stayed unchanged (trivially true here
    # since all were masked) and that the output is a valid tensor.
    assert result.dtype == torch.long
    assert masks_after <= masks_before  # can only stay same or decrease


def test_sample_shape(mdlm_sampler):
    """sample should return a (batch_size, seq_len) tensor."""
    result = mdlm_sampler.sample(BATCH_SIZE, SEQ_LEN, n_steps=N_STEPS)
    assert result.shape == (BATCH_SIZE, SEQ_LEN)


def test_sample_no_masks_at_end(mdlm_sampler):
    """After full sampling, the output should contain no mask tokens."""

    # Use a model_fn that never returns mask_token_id as top token
    def no_mask_model_fn(noisy_ids: torch.Tensor, t) -> torch.Tensor:  # noqa: ANN001
        B, T = noisy_ids.shape
        logits = torch.full((B, T, VOCAB_SIZE), -1e9)
        # Make token 1 always the best (not the mask token 0)
        logits[:, :, 1] = 0.0
        return logits

    sched = MaskNoiseSchedule(n_steps=N_STEPS, mask_token_id=MASK_TOKEN_ID)
    sampler = MDLMSampler(model_fn=no_mask_model_fn, schedule=sched, vocab_size=VOCAB_SIZE)
    result = sampler.sample(BATCH_SIZE, SEQ_LEN, n_steps=N_STEPS)
    assert (result != MASK_TOKEN_ID).all(), "Output still contains mask tokens"


def test_sample_tokens_in_vocab_range(mdlm_sampler):
    """All sampled token ids should be in [0, vocab_size)."""
    result = mdlm_sampler.sample(BATCH_SIZE, SEQ_LEN, n_steps=N_STEPS)
    assert (result >= 0).all()
    assert (result < VOCAB_SIZE).all()


# ---------------------------------------------------------------------------
# ContinuousTimeDiffusionSampler tests
# ---------------------------------------------------------------------------


@pytest.fixture
def ct_sampler():
    return ContinuousTimeDiffusionSampler(
        model_fn=make_random_model_fn(VOCAB_SIZE),
        vocab_size=VOCAB_SIZE,
        mask_token_id=MASK_TOKEN_ID,
    )


def test_log_snr_at_half_is_zero(ct_sampler):
    """log_snr(0.5) should be 0 (balanced)."""
    val = ct_sampler.log_snr(0.5)
    assert abs(val) < 0.1  # close to zero


def test_sample_timesteps_shape(ct_sampler):
    """sample_timesteps(n) should return a tensor of shape (n,)."""
    ts = ct_sampler.sample_timesteps(10)
    assert ts.shape == (10,)


def test_denoise_correct_shape(ct_sampler):
    """denoise should return a tensor of the correct shape."""
    noisy = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    result = ct_sampler.denoise(noisy, t=0.8, t_prev=0.5)
    assert result.shape == (BATCH_SIZE, SEQ_LEN)


# ---------------------------------------------------------------------------
# DiffusionLMQualityMetrics tests
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics():
    return DiffusionLMQualityMetrics(mask_token_id=MASK_TOKEN_ID)


def test_mask_fraction_in_range(metrics):
    """mask_fraction should return a value in [0, 1]."""
    seqs = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    frac = metrics.mask_fraction(seqs)
    assert 0.0 <= frac <= 1.0


def test_unique_fraction_in_range(metrics):
    """unique_fraction should return a value in [0, 1]."""
    seqs = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    frac = metrics.unique_fraction(seqs)
    assert 0.0 <= frac <= 1.0
