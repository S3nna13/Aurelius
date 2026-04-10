"""Tests for reward_distillation module."""
import pytest
import torch
import torch.nn as nn
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.alignment.reward_distillation import (
    RewardDistillConfig,
    RewardHead,
    distillation_loss,
    preference_loss,
    generate_synthetic_pairs,
    RewardDistillationTrainer,
)

D_MODEL = 64
VOCAB_SIZE = 256
BATCH = 2
SEQ_LEN = 8
N_PAIRS = 2


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2, d_model=D_MODEL, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=VOCAB_SIZE, max_seq_len=512,
    )


@pytest.fixture
def student_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def teacher_model(small_cfg):
    torch.manual_seed(1)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def input_ids():
    torch.manual_seed(42)
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


# --- RewardDistillConfig ---

def test_reward_distill_config_defaults():
    cfg = RewardDistillConfig()
    assert cfg.temperature == 2.0
    assert cfg.alpha == 0.5
    assert cfg.n_synthetic_pairs == 4
    assert cfg.margin == 0.5


# --- RewardHead ---

def test_reward_head_output_shape():
    head = RewardHead(D_MODEL)
    hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = head(hidden)
    assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"


def test_reward_head_differentiable():
    head = RewardHead(D_MODEL)
    hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    out = head(hidden)
    loss = out.sum()
    loss.backward()
    assert hidden.grad is not None


# --- distillation_loss ---

def test_distillation_loss_returns_scalar():
    s = torch.randn(BATCH)
    t = torch.randn(BATCH)
    loss = distillation_loss(s, t, temperature=2.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_distillation_loss_same_inputs_near_zero():
    rewards = torch.tensor([1.0, 2.0])
    loss = distillation_loss(rewards, rewards.clone(), temperature=2.0)
    assert loss.item() < 1e-6


def test_distillation_loss_different_inputs_positive():
    s = torch.tensor([10.0, -10.0])
    t = torch.tensor([-10.0, 10.0])
    loss = distillation_loss(s, t, temperature=2.0)
    assert loss.item() > 0


# --- preference_loss ---

def test_preference_loss_returns_scalar():
    chosen = torch.tensor([1.0, 2.0])
    rejected = torch.tensor([0.0, 0.5])
    loss = preference_loss(chosen, rejected, margin=0.5)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_preference_loss_chosen_much_better_low_loss():
    chosen = torch.tensor([10.0, 10.0])
    rejected = torch.tensor([-10.0, -10.0])
    loss = preference_loss(chosen, rejected, margin=0.5)
    assert loss.item() < 0.1


def test_preference_loss_chosen_much_worse_high_loss():
    chosen = torch.tensor([-10.0, -10.0])
    rejected = torch.tensor([10.0, 10.0])
    loss = preference_loss(chosen, rejected, margin=0.5)
    assert loss.item() > 5.0


# --- generate_synthetic_pairs ---

def test_generate_synthetic_pairs_returns_n_pairs(student_model):
    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))
    pairs = generate_synthetic_pairs(student_model, prompt, n_pairs=N_PAIRS, temperature=0)
    assert len(pairs) == N_PAIRS


def test_generate_synthetic_pairs_each_is_tuple(student_model):
    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))
    pairs = generate_synthetic_pairs(student_model, prompt, n_pairs=N_PAIRS, temperature=0)
    for pair in pairs:
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        chosen, rejected = pair
        assert isinstance(chosen, torch.Tensor)
        assert isinstance(rejected, torch.Tensor)


def test_generate_synthetic_pairs_same_length(student_model):
    prompt = torch.randint(0, VOCAB_SIZE, (1, 4))
    pairs = generate_synthetic_pairs(student_model, prompt, n_pairs=N_PAIRS, temperature=0)
    for chosen, rejected in pairs:
        assert chosen.shape[0] == rejected.shape[0]


# --- RewardDistillationTrainer ---

def test_trainer_train_step_returns_required_keys(student_model, teacher_model, input_ids):
    config = RewardDistillConfig()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    trainer = RewardDistillationTrainer(student_model, teacher_model, config, optimizer)
    result = trainer.train_step(input_ids)
    assert "loss" in result
    assert "distill_loss" in result
    assert "pref_loss" in result


def test_trainer_train_step_loss_is_positive_float(student_model, teacher_model, input_ids):
    config = RewardDistillConfig()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    trainer = RewardDistillationTrainer(student_model, teacher_model, config, optimizer)
    result = trainer.train_step(input_ids)
    assert isinstance(result["loss"], float)
    assert result["loss"] > 0
