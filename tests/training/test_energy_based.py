"""Tests for energy-based model training with contrastive divergence."""

from __future__ import annotations

import math

import pytest
import torch
import torch.optim as optim

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.energy_based import (
    EBMConfig,
    EBMTrainer,
    ReplayBuffer,
    compute_energy,
    gibbs_sample_step,
)

torch.manual_seed(42)

VOCAB_SIZE = 256
SEQ_LEN = 8
BATCH = 2
N_NEG = 2


@pytest.fixture(scope="module")
def small_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def model(small_cfg: AureliusConfig) -> AureliusTransformer:
    m = AureliusTransformer(small_cfg)
    m.eval()
    return m


@pytest.fixture(scope="module")
def ebm_cfg() -> EBMConfig:
    return EBMConfig(
        cd_steps=1,
        n_negative_samples=N_NEG,
        replay_buffer_size=10,
    )


@pytest.fixture(scope="module")
def pos_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


def test_ebmconfig_defaults():
    cfg = EBMConfig()
    assert cfg.cd_steps == 1
    assert cfg.mcmc_step_size == 0.1
    assert cfg.n_negative_samples == 4
    assert cfg.energy_margin == 0.5
    assert cfg.replay_buffer_size == 100


def test_compute_energy_shape(model, pos_ids):
    energy = compute_energy(model, pos_ids)
    assert energy.shape == (BATCH,)


def test_compute_energy_nonnegative(model, pos_ids):
    energy = compute_energy(model, pos_ids)
    assert (energy >= 0).all(), f"Expected non-negative energies, got {energy}"


def test_compute_energy_different_inputs(model):
    ids_a = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    ids_b = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    while torch.equal(ids_a, ids_b):
        ids_b = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    e_a = compute_energy(model, ids_a)
    e_b = compute_energy(model, ids_b)
    assert not torch.allclose(e_a, e_b)


def test_gibbs_sample_step_shape(model, pos_ids):
    out = gibbs_sample_step(model, pos_ids, n_steps=1)
    assert out.shape == pos_ids.shape


def test_gibbs_sample_step_valid_tokens(model, pos_ids):
    out = gibbs_sample_step(model, pos_ids, n_steps=1)
    assert (out >= 0).all() and (out < VOCAB_SIZE).all()


def test_gibbs_sample_step_modifies_tokens(model):
    ids = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN))
    out = gibbs_sample_step(model, ids, n_steps=3)
    assert not torch.equal(ids, out)


def test_replay_buffer_starts_empty():
    buf = ReplayBuffer(buffer_size=10)
    assert len(buf) == 0


def test_replay_buffer_add_increases_length():
    buf = ReplayBuffer(buffer_size=10)
    samples = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    buf.add(samples)
    assert len(buf) == BATCH


def test_replay_buffer_sample_correct_count():
    buf = ReplayBuffer(buffer_size=20)
    samples = torch.randint(0, VOCAB_SIZE, (5, SEQ_LEN))
    buf.add(samples)
    drawn = buf.sample(3)
    assert drawn is not None
    assert drawn.shape == (3, SEQ_LEN)


def test_replay_buffer_sample_none_when_empty():
    buf = ReplayBuffer(buffer_size=10)
    result = buf.sample(3)
    assert result is None


def test_replay_buffer_evicts_at_capacity():
    capacity = 5
    buf = ReplayBuffer(buffer_size=capacity)
    for _ in range(3):
        buf.add(torch.randint(0, VOCAB_SIZE, (3, SEQ_LEN)))
    assert len(buf) <= capacity


def test_ebmtrainer_train_step_keys(small_cfg, ebm_cfg):
    m = AureliusTransformer(small_cfg)
    m.train()
    optimizer = optim.Adam(m.parameters(), lr=1e-4)
    trainer = EBMTrainer(m, ebm_cfg, optimizer)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    result = trainer.train_step(ids)
    assert "loss" in result
    assert "energy_pos" in result
    assert "energy_neg" in result


def test_ebmtrainer_train_step_loss_finite(small_cfg, ebm_cfg):
    m = AureliusTransformer(small_cfg)
    m.train()
    optimizer = optim.Adam(m.parameters(), lr=1e-4)
    trainer = EBMTrainer(m, ebm_cfg, optimizer)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    result = trainer.train_step(ids)
    assert isinstance(result["loss"], float)
    assert not (result["loss"] != result["loss"]), "Loss should not be NaN"
    assert abs(result["loss"]) < 1e9, "Loss should be finite"
