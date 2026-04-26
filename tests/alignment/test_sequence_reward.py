"""Tests for sequence_reward module."""

import pytest
import torch

from src.alignment.sequence_reward import (
    RewardHead,
    RewardModelTrainerV2,
    SequenceRewardConfig,
    SequenceRewardModel,
    bradley_terry_loss,
    compute_reward_accuracy,
    compute_reward_margin,
    pool_hidden_states,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

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

B, T, D = 2, 8, 64


@pytest.fixture
def backbone():
    torch.manual_seed(42)
    return AureliusTransformer(TINY_CFG)


@pytest.fixture
def cfg():
    return SequenceRewardConfig()


@pytest.fixture
def reward_model(backbone, cfg):
    return SequenceRewardModel(backbone, cfg)


@pytest.fixture
def trainer(reward_model, cfg):
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-4)
    return RewardModelTrainerV2(reward_model, optimizer, cfg)


# ---- Config tests ----


def test_config_defaults():
    cfg = SequenceRewardConfig()
    assert cfg.pooling == "last"
    assert cfg.margin == 0.5


# ---- RewardHead tests ----


def test_reward_head_shape():
    head = RewardHead(D, 64)
    x = torch.randn(B, D)
    out = head(x)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


def test_reward_head_output_scalar():
    head = RewardHead(D, 64)
    x = torch.randn(B, D)
    out = head(x)
    # each element is scalar (1-D values per batch item), unbounded
    assert out.ndim == 1
    assert torch.isfinite(out).all()


# ---- pool_hidden_states tests ----


def test_pool_hidden_states_last_shape():
    h = torch.randn(B, T, D)
    out = pool_hidden_states(h, None, "last")
    assert out.shape == (B, D)


def test_pool_hidden_states_mean_shape():
    h = torch.randn(B, T, D)
    out = pool_hidden_states(h, None, "mean")
    assert out.shape == (B, D)


def test_pool_hidden_states_max_shape():
    h = torch.randn(B, T, D)
    out = pool_hidden_states(h, None, "max")
    assert out.shape == (B, D)


# ---- bradley_terry_loss tests ----


def test_bradley_terry_loss_scalar():
    chosen = torch.tensor([1.0, 2.0, 0.5])
    rejected = torch.tensor([0.0, 0.5, 1.0])
    loss = bradley_terry_loss(chosen, rejected)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_bradley_terry_loss_better_chosen():
    # Large margin => chosen is clearly better => low loss
    chosen = torch.tensor([10.0, 10.0])
    rejected = torch.tensor([-10.0, -10.0])
    loss = bradley_terry_loss(chosen, rejected, margin=0.0)
    assert loss.item() < 0.01


# ---- Accuracy and margin tests ----


def test_compute_reward_accuracy_all_correct():
    chosen = torch.tensor([1.0, 2.0, 3.0])
    rejected = torch.tensor([0.0, 0.5, 1.0])
    acc = compute_reward_accuracy(chosen, rejected)
    assert acc == 1.0


def test_compute_reward_accuracy_all_wrong():
    chosen = torch.tensor([0.0, 0.5, 1.0])
    rejected = torch.tensor([1.0, 2.0, 3.0])
    acc = compute_reward_accuracy(chosen, rejected)
    assert acc == 0.0


def test_compute_reward_margin_positive():
    chosen = torch.tensor([2.0, 3.0])
    rejected = torch.tensor([0.0, 1.0])
    margin = compute_reward_margin(chosen, rejected)
    assert margin > 0.0


# ---- SequenceRewardModel tests ----


def test_sequence_reward_model_shape(reward_model):
    ids = torch.randint(0, 256, (B, T))
    out = reward_model(ids)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


def test_sequence_reward_model_score_batch_no_grad(reward_model):
    ids = torch.randint(0, 256, (B, T))
    scores = reward_model.score_batch(ids)
    assert scores.shape == (B,)
    # score_batch should not require grad
    assert not scores.requires_grad


# ---- Trainer tests ----


def test_trainer_train_step_keys(trainer):
    chosen = torch.randint(0, 256, (B, T))
    rejected = torch.randint(0, 256, (B, T))
    result = trainer.train_step(chosen, rejected)
    expected_keys = {
        "loss",
        "accuracy",
        "mean_margin",
        "mean_chosen_reward",
        "mean_rejected_reward",
    }
    assert set(result.keys()) == expected_keys


def test_trainer_loss_positive(trainer):
    chosen = torch.randint(0, 256, (B, T))
    rejected = torch.randint(0, 256, (B, T))
    result = trainer.train_step(chosen, rejected)
    assert result["loss"] > 0


def test_normalize_rewards_zero_mean(trainer):
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = trainer.normalize_rewards(rewards)
    assert abs(normalized.mean().item()) < 1e-5
