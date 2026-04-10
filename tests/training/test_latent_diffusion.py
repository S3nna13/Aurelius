"""Tests for src/training/latent_diffusion.py.

Covers:
 1.  LatentDiffusionConfig defaults
 2.  TextEncoder output shape (B, T, latent_dim)
 3.  TextDecoder output shape (B, T, vocab_size)
 4.  LatentDenoiser output shape matches z_t input
 5.  LatentDenoiser handles a batch with different timesteps
 6.  ldm_noise_schedule returns dict with all required keys
 7.  ldm_noise_schedule betas are monotonically increasing
 8.  ldm_noise_schedule alphas_cumprod is strictly decreasing
 9.  ldm_q_sample output shapes match input
10.  ldm_q_sample at t=0: z_t ≈ sqrt_alpha_0 * z0 (mostly signal)
11.  ldm_loss returns a scalar tensor
12.  ldm_loss is finite and positive
13.  LatentDiffusionTrainer.train_step returns dict with 'loss'
14.  LatentDiffusionTrainer.sample returns (B, T) int64 tensor
15.  LatentDiffusionTrainer loss stays finite over 3 consecutive steps
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.latent_diffusion import (
    LatentDiffusionConfig,
    TextEncoder,
    TextDecoder,
    LatentDenoiser,
    ldm_noise_schedule,
    ldm_q_sample,
    ldm_loss,
    LatentDiffusionTrainer,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64
LATENT_DIM = 16
N_TIMESTEPS = 20
SEQ_LEN = 8
BATCH = 4


@pytest.fixture
def cfg() -> LatentDiffusionConfig:
    return LatentDiffusionConfig(
        latent_dim=LATENT_DIM,
        n_timesteps=N_TIMESTEPS,
        beta_start=1e-4,
        beta_end=0.02,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
    )


@pytest.fixture
def schedule(cfg: LatentDiffusionConfig) -> dict:
    return ldm_noise_schedule(cfg.n_timesteps, cfg.beta_start, cfg.beta_end)


@pytest.fixture
def encoder(cfg: LatentDiffusionConfig) -> TextEncoder:
    return TextEncoder(cfg.vocab_size, cfg.latent_dim)


@pytest.fixture
def decoder(cfg: LatentDiffusionConfig) -> TextDecoder:
    return TextDecoder(cfg.latent_dim, cfg.vocab_size)


@pytest.fixture
def denoiser(cfg: LatentDiffusionConfig) -> LatentDenoiser:
    return LatentDenoiser(cfg.latent_dim, cfg.n_timesteps)


@pytest.fixture
def input_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


@pytest.fixture
def trainer(
    encoder: TextEncoder,
    decoder: TextDecoder,
    denoiser: LatentDenoiser,
    cfg: LatentDiffusionConfig,
) -> LatentDiffusionTrainer:
    params = (
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(denoiser.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=1e-3)
    return LatentDiffusionTrainer(encoder, decoder, denoiser, cfg, optimizer)


# ---------------------------------------------------------------------------
# Test 1: LatentDiffusionConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults() -> None:
    cfg = LatentDiffusionConfig()
    assert cfg.latent_dim == 64
    assert cfg.n_timesteps == 100
    assert cfg.beta_start == 1e-4
    assert cfg.beta_end == 0.02
    assert cfg.vocab_size == 256
    assert cfg.seq_len == 32


# ---------------------------------------------------------------------------
# Test 2: TextEncoder output shape
# ---------------------------------------------------------------------------

def test_text_encoder_shape(encoder: TextEncoder, input_ids: torch.Tensor) -> None:
    latents = encoder(input_ids)
    assert latents.shape == (BATCH, SEQ_LEN, LATENT_DIM), (
        f"Expected ({BATCH}, {SEQ_LEN}, {LATENT_DIM}), got {latents.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: TextDecoder output shape
# ---------------------------------------------------------------------------

def test_text_decoder_shape(decoder: TextDecoder) -> None:
    latents = torch.randn(BATCH, SEQ_LEN, LATENT_DIM)
    logits = decoder(latents)
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE), (
        f"Expected ({BATCH}, {SEQ_LEN}, {VOCAB_SIZE}), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# Test 4: LatentDenoiser output shape matches z_t input
# ---------------------------------------------------------------------------

def test_denoiser_output_shape(denoiser: LatentDenoiser) -> None:
    z_t = torch.randn(BATCH, SEQ_LEN, LATENT_DIM)
    t = torch.randint(0, N_TIMESTEPS, (BATCH,))
    out = denoiser(z_t, t)
    assert out.shape == z_t.shape, (
        f"Denoiser output shape {out.shape} != input shape {z_t.shape}"
    )


# ---------------------------------------------------------------------------
# Test 5: LatentDenoiser handles batch with different timesteps
# ---------------------------------------------------------------------------

def test_denoiser_different_timesteps(denoiser: LatentDenoiser) -> None:
    z_t = torch.randn(BATCH, SEQ_LEN, LATENT_DIM)
    # Explicitly varied timesteps: 0, 5, 10, 19
    t = torch.tensor([0, 5, 10, N_TIMESTEPS - 1])
    out = denoiser(z_t, t)
    assert out.shape == z_t.shape


# ---------------------------------------------------------------------------
# Test 6: ldm_noise_schedule returns dict with all required keys
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "betas",
    "alphas",
    "alphas_cumprod",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
}


def test_noise_schedule_keys(schedule: dict) -> None:
    assert REQUIRED_KEYS <= set(schedule.keys()), (
        f"Missing keys: {REQUIRED_KEYS - set(schedule.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 7: ldm_noise_schedule betas are monotonically increasing
# ---------------------------------------------------------------------------

def test_noise_schedule_betas_increasing(schedule: dict) -> None:
    betas = schedule["betas"]
    diffs = betas[1:] - betas[:-1]
    assert (diffs >= 0).all(), "betas should be non-decreasing (linear schedule)"


# ---------------------------------------------------------------------------
# Test 8: alphas_cumprod is strictly decreasing
# ---------------------------------------------------------------------------

def test_noise_schedule_alphas_cumprod_decreasing(schedule: dict) -> None:
    acp = schedule["alphas_cumprod"]
    assert acp[-1].item() < acp[0].item(), (
        "alphas_cumprod[-1] should be < alphas_cumprod[0]"
    )
    diffs = acp[1:] - acp[:-1]
    assert (diffs < 0).all(), "alphas_cumprod should be strictly decreasing"


# ---------------------------------------------------------------------------
# Test 9: ldm_q_sample output shapes match input
# ---------------------------------------------------------------------------

def test_q_sample_shapes(schedule: dict, encoder: TextEncoder, input_ids: torch.Tensor) -> None:
    z0 = encoder(input_ids)
    t = torch.randint(0, N_TIMESTEPS, (BATCH,))
    z_t, noise = ldm_q_sample(z0, t, schedule)
    assert z_t.shape == z0.shape, f"z_t shape {z_t.shape} != z0 shape {z0.shape}"
    assert noise.shape == z0.shape, f"noise shape {noise.shape} != z0 shape {z0.shape}"


# ---------------------------------------------------------------------------
# Test 10: ldm_q_sample at t=0 is mostly signal
# ---------------------------------------------------------------------------

def test_q_sample_t0_mostly_signal(schedule: dict, encoder: TextEncoder, input_ids: torch.Tensor) -> None:
    """At t=0, z_t ≈ sqrt_alpha_0 * z0 + small noise contribution."""
    z0 = encoder(input_ids)
    t = torch.zeros(BATCH, dtype=torch.long)
    z_t, _noise = ldm_q_sample(z0, t, schedule)

    sqrt_alpha_0 = schedule["sqrt_alphas_cumprod"][0].item()
    sqrt_one_minus_0 = schedule["sqrt_one_minus_alphas_cumprod"][0].item()

    # The signal coefficient should be much larger than the noise coefficient
    # for a well-chosen beta_start (1e-4 -> sqrt_alpha ~0.9999, noise ~0.01)
    assert sqrt_alpha_0 > 0.99, (
        f"sqrt_alpha_0={sqrt_alpha_0:.6f} should be close to 1 at t=0"
    )
    assert sqrt_one_minus_0 < 0.15, (
        f"sqrt_one_minus_0={sqrt_one_minus_0:.6f} should be small at t=0"
    )

    # z_t should be highly correlated with scaled z0
    signal = sqrt_alpha_0 * z0
    # Element-wise relative difference should be small on average
    rel_err = (z_t - signal).norm() / signal.norm().clamp(min=1e-8)
    assert rel_err.item() < 0.5, (
        f"z_t at t=0 should be close to sqrt_alpha_0 * z0, rel_err={rel_err:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 11: ldm_loss returns a scalar tensor
# ---------------------------------------------------------------------------

def test_ldm_loss_scalar(
    encoder: TextEncoder,
    denoiser: LatentDenoiser,
    schedule: dict,
    input_ids: torch.Tensor,
) -> None:
    loss = ldm_loss(denoiser, encoder, input_ids, schedule)
    assert loss.shape == (), f"loss should be scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# Test 12: ldm_loss is finite and positive
# ---------------------------------------------------------------------------

def test_ldm_loss_finite_positive(
    encoder: TextEncoder,
    denoiser: LatentDenoiser,
    schedule: dict,
    input_ids: torch.Tensor,
) -> None:
    loss = ldm_loss(denoiser, encoder, input_ids, schedule)
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
    assert loss.item() > 0.0, f"loss should be positive, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 13: LatentDiffusionTrainer.train_step returns dict with 'loss'
# ---------------------------------------------------------------------------

def test_trainer_train_step_has_loss(
    trainer: LatentDiffusionTrainer, input_ids: torch.Tensor
) -> None:
    result = trainer.train_step(input_ids)
    assert isinstance(result, dict), "train_step should return a dict"
    assert "loss" in result, f"'loss' not in result keys: {list(result.keys())}"
    assert isinstance(result["loss"], float), (
        f"'loss' should be a float, got {type(result['loss'])}"
    )


# ---------------------------------------------------------------------------
# Test 14: LatentDiffusionTrainer.sample returns (B, T) int64
# ---------------------------------------------------------------------------

def test_trainer_sample_shape_and_dtype(trainer: LatentDiffusionTrainer) -> None:
    token_ids = trainer.sample(batch_size=BATCH, seq_len=SEQ_LEN)
    assert token_ids.shape == (BATCH, SEQ_LEN), (
        f"Expected shape ({BATCH}, {SEQ_LEN}), got {token_ids.shape}"
    )
    assert token_ids.dtype == torch.int64, (
        f"Expected int64, got {token_ids.dtype}"
    )


# ---------------------------------------------------------------------------
# Test 15: Loss stays finite over 3 consecutive train_steps
# ---------------------------------------------------------------------------

def test_trainer_loss_finite_over_steps(
    trainer: LatentDiffusionTrainer, input_ids: torch.Tensor
) -> None:
    losses = []
    for _ in range(3):
        result = trainer.train_step(input_ids)
        losses.append(result["loss"])

    for i, loss_val in enumerate(losses):
        assert isinstance(loss_val, float), f"Step {i}: loss is not a float"
        assert loss_val == loss_val, f"Step {i}: loss is NaN"  # NaN != NaN
        assert loss_val < float("inf"), f"Step {i}: loss is inf"
        assert loss_val > 0.0, f"Step {i}: loss is non-positive"
