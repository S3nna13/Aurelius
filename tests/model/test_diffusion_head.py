"""Tests for masked diffusion language model head (diffusion_head.py)."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.model.diffusion_head import (
    MASK_TOKEN_ID,
    DiffusionSchedule,
    DiffusionLMHead,
    MaskedDiffusionTrainer,
    DiffusionDecoder,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# Small config for all tests
SMALL_CONFIG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)


# ---------------------------------------------------------------------------
# DiffusionSchedule tests
# ---------------------------------------------------------------------------

def test_schedule_mask_rate_range():
    """mask_rate in [0, 1] for all t, both schedules."""
    for schedule in ("linear", "cosine"):
        sched = DiffusionSchedule(T=100, schedule=schedule)
        for t in range(0, 101, 10):
            rate = sched.mask_rate(t)
            assert 0.0 <= rate.item() <= 1.0, (
                f"{schedule} schedule: mask_rate({t}) = {rate.item()} out of [0,1]"
            )


def test_schedule_linear():
    """Linear schedule: mask_rate(0) == 0, mask_rate(T) == 1."""
    sched = DiffusionSchedule(T=100, schedule="linear")
    assert sched.mask_rate(0).item() == pytest.approx(0.0, abs=1e-6)
    assert sched.mask_rate(100).item() == pytest.approx(1.0, abs=1e-6)


def test_schedule_cosine():
    """Cosine schedule is monotonically increasing: mask_rate(T//2) < mask_rate(T)."""
    sched = DiffusionSchedule(T=100, schedule="cosine")
    half_rate = sched.mask_rate(50).item()
    full_rate = sched.mask_rate(100).item()
    assert half_rate < full_rate, (
        f"Cosine schedule not monotone: mask_rate(50)={half_rate}, mask_rate(100)={full_rate}"
    )


def test_add_noise_shape():
    """Noisy ids have the same shape as input."""
    sched = DiffusionSchedule(T=100)
    x = torch.randint(1, 256, (4, 16))
    noisy = sched.add_noise(x, t=50)
    assert noisy.shape == x.shape


def test_add_noise_masks_tokens():
    """At t=T (full noise), most/all tokens should be masked."""
    sched = DiffusionSchedule(T=100, mask_token_id=MASK_TOKEN_ID)
    # Use tokens != MASK_TOKEN_ID so masking is detectable
    x = torch.randint(1, 256, (8, 64))
    noisy = sched.add_noise(x, t=100)
    frac_masked = (noisy == MASK_TOKEN_ID).float().mean().item()
    # At t=T mask_rate=1.0, expect essentially all masked (stochastic, so >90%)
    assert frac_masked > 0.9, f"Expected >90% masked at t=T, got {frac_masked:.3f}"


def test_add_noise_clean_at_t0():
    """At t=0, no tokens should be masked."""
    sched = DiffusionSchedule(T=100, mask_token_id=MASK_TOKEN_ID)
    x = torch.randint(1, 256, (4, 16))
    noisy = sched.add_noise(x, t=0)
    assert torch.equal(noisy, x), "At t=0 no tokens should be masked"


def test_get_timesteps_range():
    """All sampled timesteps are in [1, T]."""
    sched = DiffusionSchedule(T=50)
    ts = sched.get_timesteps(batch_size=1000)
    assert ts.min().item() >= 1
    assert ts.max().item() <= 50


# ---------------------------------------------------------------------------
# DiffusionLMHead tests
# ---------------------------------------------------------------------------

def test_diffusion_head_output_shape():
    """(2, 16, 64) hidden states -> (2, 16, vocab_size) logits."""
    head = DiffusionLMHead(d_model=64, vocab_size=256)
    hidden = torch.randn(2, 16, 64)
    logits = head(hidden)
    assert logits.shape == (2, 16, 256)


def test_diffusion_loss_on_masked_only():
    """Loss is only computed where noisy_ids == MASK_TOKEN_ID."""
    head = DiffusionLMHead(d_model=64, vocab_size=256, mask_token_id=MASK_TOKEN_ID)
    B, L, V = 2, 8, 256

    original_ids = torch.randint(1, V, (B, L))
    # Only position 0 is masked
    noisy_ids = original_ids.clone()
    noisy_ids[:, 0] = MASK_TOKEN_ID

    # Logits that perfectly predict original at masked position
    logits = torch.zeros(B, L, V)
    for b in range(B):
        logits[b, 0, original_ids[b, 0]] = 100.0  # confident correct prediction

    loss = head.diffusion_loss(logits, original_ids, noisy_ids)
    # High confidence correct prediction -> near-zero loss
    assert loss.item() < 0.01, f"Expected near-zero loss, got {loss.item()}"


def test_diffusion_loss_zero_no_masks():
    """If no tokens are masked, loss should be 0."""
    head = DiffusionLMHead(d_model=64, vocab_size=256, mask_token_id=MASK_TOKEN_ID)
    B, L, V = 2, 8, 256
    # Use tokens != MASK_TOKEN_ID
    original_ids = torch.randint(1, V, (B, L))
    noisy_ids = original_ids.clone()  # no masking
    logits = torch.randn(B, L, V)

    loss = head.diffusion_loss(logits, original_ids, noisy_ids)
    assert loss.item() == pytest.approx(0.0, abs=1e-6), (
        f"Expected 0 loss when no masks, got {loss.item()}"
    )


# ---------------------------------------------------------------------------
# MaskedDiffusionTrainer tests
# ---------------------------------------------------------------------------

def _make_trainer():
    """Helper: create a small trainer for testing."""
    config = SMALL_CONFIG
    model = AureliusTransformer(config)
    head = DiffusionLMHead(d_model=config.d_model, vocab_size=config.vocab_size)
    schedule = DiffusionSchedule(T=10, schedule="linear", mask_token_id=MASK_TOKEN_ID)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=1e-4
    )
    trainer = MaskedDiffusionTrainer(model, head, schedule, optimizer)
    return trainer


def test_diffusion_trainer_step_keys():
    """train_step returns dict with 'loss', 'mask_rate', 'n_masked'."""
    trainer = _make_trainer()
    # Use tokens != 0 so masking is detectable
    input_ids = torch.randint(1, 256, (2, 16))
    result = trainer.train_step(input_ids)
    assert "loss" in result
    assert "mask_rate" in result
    assert "n_masked" in result


def test_diffusion_trainer_loss_positive():
    """Loss is positive when masked tokens are present."""
    trainer = _make_trainer()
    # Force high mask rate by using a high timestep
    # Patch get_timesteps to always return T
    original_get = trainer.schedule.get_timesteps
    trainer.schedule.get_timesteps = lambda batch_size, device=None: torch.full(
        (batch_size,), trainer.schedule.T, dtype=torch.long
    )
    input_ids = torch.randint(1, 256, (2, 16))
    result = trainer.train_step(input_ids)
    trainer.schedule.get_timesteps = original_get
    assert result["loss"] > 0.0, f"Expected loss > 0, got {result['loss']}"


# ---------------------------------------------------------------------------
# DiffusionDecoder tests
# ---------------------------------------------------------------------------

def test_diffusion_decoder_shape():
    """decode returns (1, gen_len) tensor."""
    config = SMALL_CONFIG
    model = AureliusTransformer(config)
    head = DiffusionLMHead(d_model=config.d_model, vocab_size=config.vocab_size)
    schedule = DiffusionSchedule(T=10, mask_token_id=MASK_TOKEN_ID)

    decoder = DiffusionDecoder(model, head, schedule, n_steps=3)
    prompt_ids = torch.randint(1, 256, (1, 4))
    gen_len = 8

    generated = decoder.decode(prompt_ids, gen_len=gen_len, n_steps=3)
    assert generated.shape == (1, gen_len), (
        f"Expected (1, {gen_len}), got {generated.shape}"
    )
