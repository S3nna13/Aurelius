"""Tests for src/training/diffusion_lm.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.diffusion_lm import (
    DiffusionConfig,
    DiffusionLMTrainer,
    ScoreNetwork,
    ddpm_sample,
    diffusion_loss,
    get_noise_schedule,
    q_sample,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

T_STEPS = 10  # n_timesteps (small for fast tests)
D = 16  # d_embed
B = 2  # batch size
SEQ = 4  # sequence length
VOCAB = 50  # small vocab


@pytest.fixture()
def config() -> DiffusionConfig:
    return DiffusionConfig(n_timesteps=T_STEPS, d_embed=D)


@pytest.fixture()
def schedule(config):
    return get_noise_schedule(config)


@pytest.fixture()
def score_net(config) -> ScoreNetwork:
    return ScoreNetwork(d_embed=D, hidden_dim=32, n_timesteps=T_STEPS)


@pytest.fixture()
def embed_layer() -> nn.Embedding:
    return nn.Embedding(VOCAB, D)


@pytest.fixture()
def trainer(score_net, embed_layer, config) -> DiffusionLMTrainer:
    opt = torch.optim.Adam(list(score_net.parameters()) + list(embed_layer.parameters()), lr=1e-3)
    return DiffusionLMTrainer(score_net, embed_layer, config, opt)


# ---------------------------------------------------------------------------
# 1. DiffusionConfig defaults
# ---------------------------------------------------------------------------


def test_diffusion_config_defaults():
    cfg = DiffusionConfig()
    assert cfg.n_timesteps == 100
    assert cfg.beta_start == 0.0001
    assert cfg.beta_end == 0.02
    assert cfg.schedule == "linear"
    assert cfg.d_embed == 64


# ---------------------------------------------------------------------------
# 2. get_noise_schedule returns shapes (T,) for both outputs
# ---------------------------------------------------------------------------


def test_noise_schedule_shapes(config, schedule):
    betas, alphas_cumprod = schedule
    assert betas.shape == (T_STEPS,)
    assert alphas_cumprod.shape == (T_STEPS,)


# ---------------------------------------------------------------------------
# 3. betas monotonically increasing (linear schedule)
# ---------------------------------------------------------------------------


def test_betas_monotonically_increasing(schedule):
    betas, _ = schedule
    diffs = betas[1:] - betas[:-1]
    assert (diffs >= 0).all(), "betas should be non-decreasing for linear schedule"


# ---------------------------------------------------------------------------
# 4. alphas_cumprod decreasing from ~1 to near 0
# ---------------------------------------------------------------------------


def test_alphas_cumprod_decreasing(schedule):
    _, alphas_cumprod = schedule
    # Should start close to 1
    assert alphas_cumprod[0] > 0.9, f"alphas_cumprod[0]={alphas_cumprod[0]}"
    # Should be monotonically non-increasing
    diffs = alphas_cumprod[1:] - alphas_cumprod[:-1]
    assert (diffs <= 0).all(), "alphas_cumprod should be non-increasing"
    # Last value should be significantly less than first
    assert alphas_cumprod[-1] < alphas_cumprod[0]


# ---------------------------------------------------------------------------
# 5. q_sample returns same shape as x0
# ---------------------------------------------------------------------------


def test_q_sample_shape(schedule):
    betas, alphas_cumprod = schedule
    x0 = torch.randn(B, SEQ, D)
    t = torch.randint(0, T_STEPS, (B,))
    x_t = q_sample(x0, t, alphas_cumprod)
    assert x_t.shape == x0.shape


# ---------------------------------------------------------------------------
# 6. q_sample t=0 -> x_t approximately x0 (low noise)
# ---------------------------------------------------------------------------


def test_q_sample_t0_low_noise(schedule):
    betas, alphas_cumprod = schedule
    x0 = torch.randn(B, SEQ, D)
    t = torch.zeros(B, dtype=torch.long)
    noise = torch.zeros_like(x0)  # zero noise to isolate scale factor
    x_t = q_sample(x0, t, alphas_cumprod, noise=noise)
    # x_t = sqrt(alpha_0) * x0, alpha_0 is close to 1 so x_t ~ x0
    alpha0 = alphas_cumprod[0].sqrt()
    expected = alpha0 * x0
    assert torch.allclose(x_t, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. q_sample t=T-1 -> x_t approximately noise (high noise, alpha->0)
# ---------------------------------------------------------------------------


def test_q_sample_tmax_high_noise(schedule):
    betas, alphas_cumprod = schedule
    x0 = torch.zeros(B, SEQ, D)  # zero signal
    noise = torch.ones(B, SEQ, D)  # unit noise
    t = torch.full((B,), T_STEPS - 1, dtype=torch.long)
    x_t = q_sample(x0, t, alphas_cumprod, noise=noise)
    # x_t = sqrt(1 - alpha_T-1) * noise; with small alpha this approaches noise
    sqrt_one_minus = (1.0 - alphas_cumprod[-1]).sqrt()
    expected = sqrt_one_minus * noise
    assert torch.allclose(x_t, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. ScoreNetwork output shape matches x_t
# ---------------------------------------------------------------------------


def test_score_network_output_shape(score_net):
    x_t = torch.randn(B, SEQ, D)
    t = torch.randint(0, T_STEPS, (B,))
    out = score_net(x_t, t)
    assert out.shape == (B, SEQ, D)


# ---------------------------------------------------------------------------
# 9. ScoreNetwork is differentiable
# ---------------------------------------------------------------------------


def test_score_network_differentiable(score_net):
    x_t = torch.randn(B, SEQ, D, requires_grad=True)
    t = torch.randint(0, T_STEPS, (B,))
    out = score_net(x_t, t)
    loss = out.sum()
    loss.backward()
    assert x_t.grad is not None
    assert not torch.isnan(x_t.grad).any()


# ---------------------------------------------------------------------------
# 10. diffusion_loss returns scalar
# ---------------------------------------------------------------------------


def test_diffusion_loss_scalar(score_net, schedule):
    betas, alphas_cumprod = schedule
    x0 = torch.randn(B, SEQ, D)
    t = torch.randint(0, T_STEPS, (B,))
    loss = diffusion_loss(score_net, x0, t, alphas_cumprod)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 11. diffusion_loss non-negative
# ---------------------------------------------------------------------------


def test_diffusion_loss_nonnegative(score_net, schedule):
    betas, alphas_cumprod = schedule
    x0 = torch.randn(B, SEQ, D)
    t = torch.randint(0, T_STEPS, (B,))
    loss = diffusion_loss(score_net, x0, t, alphas_cumprod)
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# 12. DiffusionLMTrainer.train_step returns required keys
# ---------------------------------------------------------------------------


def test_trainer_train_step_keys(trainer):
    input_ids = torch.randint(0, VOCAB, (B, SEQ))
    result = trainer.train_step(input_ids)
    assert "loss" in result, "train_step must return 'loss'"
    assert "t_mean" in result, "train_step must return 't_mean'"


# ---------------------------------------------------------------------------
# 13. DiffusionLMTrainer.generate returns (n_samples, seq_len, d_embed)
# ---------------------------------------------------------------------------


def test_trainer_generate_shape(trainer, config):
    n_samples = 3
    seq_len = SEQ
    out = trainer.generate(n_samples, seq_len)
    assert out.shape == (n_samples, seq_len, config.d_embed)


# ---------------------------------------------------------------------------
# 14. ddpm_sample output shape correct
# ---------------------------------------------------------------------------


def test_ddpm_sample_shape(score_net, schedule):
    betas, alphas_cumprod = schedule
    shape = (B, SEQ, D)
    out = ddpm_sample(score_net, shape, alphas_cumprod, betas, n_steps=3)
    assert out.shape == shape
