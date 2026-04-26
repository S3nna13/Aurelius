"""Tests for src/model/energy_model.py — Energy-Based Model Scoring."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.energy_model import (
    EBMConfig,
    EBMTrainer,
    EnergyHead,
    _get_hidden_states,
    compute_sequence_energy,
    contrastive_divergence_loss,
    langevin_step,
)
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
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
EBM_CFG = EBMConfig(
    d_model=64, energy_hidden=128, noise_scale=0.1, n_mcmc_steps=10, step_size=0.01, temperature=1.0
)
BATCH = 2
SEQ = 8


@pytest.fixture
def model():
    torch.manual_seed(42)
    m = AureliusTransformer(TINY_CFG)
    m.eval()
    return m


@pytest.fixture
def energy_head():
    torch.manual_seed(42)
    return EnergyHead(EBM_CFG)


@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, 256, (BATCH, SEQ))


@pytest.fixture
def neg_ids():
    torch.manual_seed(1)
    return torch.randint(0, 256, (BATCH, SEQ))


# ---------------------------------------------------------------------------
# EBMConfig tests
# ---------------------------------------------------------------------------


def test_ebm_config_defaults():
    """EBMConfig has correct default values."""
    cfg = EBMConfig()
    assert cfg.d_model == 64
    assert cfg.energy_hidden == 128
    assert cfg.noise_scale == 0.1
    assert cfg.n_mcmc_steps == 10
    assert cfg.step_size == 0.01
    assert cfg.temperature == 1.0


def test_ebm_config_custom():
    """EBMConfig accepts custom values."""
    cfg = EBMConfig(
        d_model=128,
        energy_hidden=256,
        noise_scale=0.2,
        n_mcmc_steps=20,
        step_size=0.05,
        temperature=2.0,
    )
    assert cfg.d_model == 128
    assert cfg.energy_hidden == 256
    assert cfg.noise_scale == 0.2


# ---------------------------------------------------------------------------
# EnergyHead tests
# ---------------------------------------------------------------------------


def test_energy_head_output_shape(energy_head):
    """EnergyHead returns (B,) energy scalars."""
    hidden = torch.randn(BATCH, SEQ, 64)
    energy = energy_head(hidden)
    assert energy.shape == (BATCH,)


def test_energy_head_scalar_per_sequence(energy_head):
    """Each sequence gets one scalar energy value."""
    hidden = torch.randn(3, 10, 64)
    energy = energy_head(hidden)
    assert energy.shape == (3,)
    assert energy.dtype == torch.float32


def test_energy_head_gradient_flows(energy_head):
    """Gradients flow through the energy head."""
    hidden = torch.randn(BATCH, SEQ, 64, requires_grad=True)
    energy = energy_head(hidden)
    energy.sum().backward()
    assert hidden.grad is not None
    assert hidden.grad.shape == hidden.shape


def test_energy_head_different_inputs_different_outputs(energy_head):
    """Different hidden states produce different energy values."""
    h1 = torch.randn(1, SEQ, 64)
    h2 = torch.randn(1, SEQ, 64) + 5.0
    e1 = energy_head(h1)
    e2 = energy_head(h2)
    assert not torch.allclose(e1, e2), "Different inputs should give different energies"


def test_energy_head_deterministic(energy_head):
    """Same input gives same output (no randomness in forward pass)."""
    hidden = torch.randn(BATCH, SEQ, 64)
    e1 = energy_head(hidden)
    e2 = energy_head(hidden)
    assert torch.allclose(e1, e2)


# ---------------------------------------------------------------------------
# _get_hidden_states tests
# ---------------------------------------------------------------------------


def test_get_hidden_states_shape(model, input_ids):
    """Hidden states have correct shape (B, S, d_model)."""
    hidden = _get_hidden_states(model, input_ids)
    assert hidden.shape == (BATCH, SEQ, TINY_CFG.d_model)


def test_get_hidden_states_no_grad_leak(model, input_ids):
    """Hidden states computation works under no_grad context."""
    with torch.no_grad():
        hidden = _get_hidden_states(model, input_ids)
    assert hidden.shape == (BATCH, SEQ, TINY_CFG.d_model)


# ---------------------------------------------------------------------------
# compute_sequence_energy tests
# ---------------------------------------------------------------------------


def test_compute_sequence_energy_shape(model, energy_head, input_ids):
    """compute_sequence_energy returns (B,) tensor."""
    energy = compute_sequence_energy(model, energy_head, input_ids)
    assert energy.shape == (BATCH,)


def test_compute_sequence_energy_differentiable(model, energy_head, input_ids):
    """Energy is differentiable w.r.t. energy head parameters."""
    energy = compute_sequence_energy(model, energy_head, input_ids)
    energy.sum().backward()
    for p in energy_head.parameters():
        assert p.grad is not None


# ---------------------------------------------------------------------------
# contrastive_divergence_loss tests
# ---------------------------------------------------------------------------


def test_cd_loss_scalar():
    """CD loss is a scalar."""
    pos = torch.tensor([1.0, 2.0, 3.0])
    neg = torch.tensor([4.0, 5.0, 6.0])
    loss = contrastive_divergence_loss(pos, neg)
    assert loss.dim() == 0


def test_cd_loss_value():
    """CD loss = mean(pos) - mean(neg)."""
    pos = torch.tensor([1.0, 2.0, 3.0])
    neg = torch.tensor([4.0, 5.0, 6.0])
    loss = contrastive_divergence_loss(pos, neg)
    expected = 2.0 - 5.0  # -3.0
    assert torch.isclose(loss, torch.tensor(expected))


def test_cd_loss_zero_when_equal():
    """CD loss is zero when pos and neg energies are identical."""
    e = torch.tensor([1.0, 2.0, 3.0])
    loss = contrastive_divergence_loss(e, e)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_cd_loss_sign():
    """Loss is negative when pos_energy < neg_energy (desired outcome)."""
    pos = torch.tensor([1.0, 1.0])
    neg = torch.tensor([5.0, 5.0])
    loss = contrastive_divergence_loss(pos, neg)
    assert loss.item() < 0, "Loss should be negative when pos < neg"


# ---------------------------------------------------------------------------
# langevin_step tests
# ---------------------------------------------------------------------------


def test_langevin_step_output_shape(model, energy_head, input_ids):
    """Langevin step returns token ids with same shape as input."""
    new_ids = langevin_step(model, energy_head, input_ids, EBM_CFG)
    assert new_ids.shape == input_ids.shape


def test_langevin_step_valid_token_range(model, energy_head, input_ids):
    """Returned token ids are within valid vocabulary range."""
    new_ids = langevin_step(model, energy_head, input_ids, EBM_CFG)
    assert (new_ids >= 0).all()
    assert (new_ids < TINY_CFG.vocab_size).all()


def test_langevin_step_dtype(model, energy_head, input_ids):
    """Returned token ids have integer dtype."""
    new_ids = langevin_step(model, energy_head, input_ids, EBM_CFG)
    assert new_ids.dtype == torch.int64


# ---------------------------------------------------------------------------
# EBMTrainer tests
# ---------------------------------------------------------------------------


def test_trainer_train_step_returns_dict(model, input_ids, neg_ids):
    """train_step returns dict with required keys."""
    trainer = EBMTrainer(model, EBM_CFG)
    result = trainer.train_step(input_ids, neg_ids)
    assert isinstance(result, dict)
    assert "cd_loss" in result
    assert "pos_energy" in result
    assert "neg_energy" in result


def test_trainer_train_step_values_are_floats(model, input_ids, neg_ids):
    """train_step returns Python floats, not tensors."""
    trainer = EBMTrainer(model, EBM_CFG)
    result = trainer.train_step(input_ids, neg_ids)
    assert isinstance(result["cd_loss"], float)
    assert isinstance(result["pos_energy"], float)
    assert isinstance(result["neg_energy"], float)


def test_trainer_updates_energy_head(model, input_ids, neg_ids):
    """Training step updates energy head parameters."""
    trainer = EBMTrainer(model, EBM_CFG)
    params_before = [p.clone() for p in trainer.energy_head.parameters()]
    trainer.train_step(input_ids, neg_ids)
    params_after = list(trainer.energy_head.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
    assert changed, "Energy head parameters should change after a training step"


def test_trainer_multiple_steps(model, input_ids, neg_ids):
    """Multiple training steps run without error."""
    trainer = EBMTrainer(model, EBM_CFG)
    for _ in range(3):
        result = trainer.train_step(input_ids, neg_ids)
    assert "cd_loss" in result
