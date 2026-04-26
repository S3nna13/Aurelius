"""Tests for src/alignment/rm_trainer.py"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from src.alignment.rm_trainer import (
    ELORatingSystem,
    RewardModelTrainer,
    RMTrainerConfig,
    compute_accuracy,
    preference_loss,
    train_reward_model,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def backbone(small_cfg: AureliusConfig) -> AureliusTransformer:
    torch.manual_seed(42)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def trainer(backbone: AureliusTransformer) -> RewardModelTrainer:
    cfg = RMTrainerConfig(learning_rate=1e-4)
    return RewardModelTrainer(backbone, cfg)


@pytest.fixture
def chosen_ids() -> Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 256, (2, 8))


@pytest.fixture
def rejected_ids() -> Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 256, (2, 8))


# ---------------------------------------------------------------------------
# 1. RMTrainerConfig defaults
# ---------------------------------------------------------------------------


def test_rmtrainerconfig_defaults():
    cfg = RMTrainerConfig()
    assert cfg.learning_rate == 1e-5
    assert cfg.batch_size == 4
    assert cfg.gradient_clip == 1.0
    assert cfg.label_smoothing == 0.0
    assert cfg.margin == 0.0
    assert cfg.use_elo is False
    assert cfg.elo_k == 32.0


# ---------------------------------------------------------------------------
# 2. preference_loss returns scalar
# ---------------------------------------------------------------------------


def test_preference_loss_returns_scalar():
    chosen = torch.tensor([1.0, 2.0, 0.5])
    rejected = torch.tensor([0.0, 0.5, 1.0])
    loss = preference_loss(chosen, rejected)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 3. preference_loss is lower when chosen >> rejected
# ---------------------------------------------------------------------------


def test_preference_loss_lower_when_correct_ordering():
    # When chosen >> rejected, loss should be small (high confidence)
    chosen_correct = torch.tensor([5.0, 5.0, 5.0])
    rejected_correct = torch.tensor([-5.0, -5.0, -5.0])
    loss_correct = preference_loss(chosen_correct, rejected_correct)

    # When chosen == rejected, loss = log(2) ~ 0.693
    equal = torch.zeros(3)
    loss_equal = preference_loss(equal, equal)

    assert loss_correct < loss_equal


# ---------------------------------------------------------------------------
# 4. preference_loss with margin increases loss when margin not satisfied
# ---------------------------------------------------------------------------


def test_preference_loss_margin_increases_loss():
    chosen = torch.tensor([1.0, 1.0])
    rejected = torch.tensor([0.0, 0.0])

    loss_no_margin = preference_loss(chosen, rejected, margin=0.0)
    # margin=2.0 means we require chosen - rejected >= 2.0; only 1.0 gap -> higher loss
    loss_with_margin = preference_loss(chosen, rejected, margin=2.0)

    assert loss_with_margin > loss_no_margin


# ---------------------------------------------------------------------------
# 5. compute_accuracy returns 1.0 when all chosen > rejected
# ---------------------------------------------------------------------------


def test_compute_accuracy_all_correct():
    chosen = torch.tensor([2.0, 3.0, 4.0])
    rejected = torch.tensor([0.0, 1.0, 2.0])
    acc = compute_accuracy(chosen, rejected)
    assert acc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. compute_accuracy returns 0.0 when all rejected > chosen
# ---------------------------------------------------------------------------


def test_compute_accuracy_all_wrong():
    chosen = torch.tensor([0.0, 1.0, 2.0])
    rejected = torch.tensor([2.0, 3.0, 4.0])
    acc = compute_accuracy(chosen, rejected)
    assert acc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. ELORatingSystem update increases winner rating
# ---------------------------------------------------------------------------


def test_elo_update_increases_winner():
    elo = ELORatingSystem(initial_rating=1000.0, k=32.0)
    elo.get_rating("A")
    elo.get_rating("B")
    before_winner = elo.ratings["A"]
    elo.update("A", "B")
    assert elo.ratings["A"] > before_winner


# ---------------------------------------------------------------------------
# 8. ELORatingSystem update decreases loser rating
# ---------------------------------------------------------------------------


def test_elo_update_decreases_loser():
    elo = ELORatingSystem(initial_rating=1000.0, k=32.0)
    elo.get_rating("A")
    elo.get_rating("B")
    before_loser = elo.ratings["B"]
    elo.update("A", "B")
    assert elo.ratings["B"] < before_loser


# ---------------------------------------------------------------------------
# 9. ELORatingSystem get_ranking returns sorted list
# ---------------------------------------------------------------------------


def test_elo_get_ranking_sorted():
    elo = ELORatingSystem()
    elo.ratings = {"C": 900.0, "A": 1100.0, "B": 1000.0}
    ranking = elo.get_ranking()
    ratings_ordered = [r for _, r in ranking]
    assert ratings_ordered == sorted(ratings_ordered, reverse=True)
    assert ranking[0][0] == "A"


# ---------------------------------------------------------------------------
# 10. RewardModelTrainer train_step returns correct keys
# ---------------------------------------------------------------------------


def test_train_step_returns_correct_keys(trainer, chosen_ids, rejected_ids):
    metrics = trainer.train_step(chosen_ids, rejected_ids)
    assert set(metrics.keys()) == {"loss", "accuracy", "chosen_reward", "rejected_reward"}


# ---------------------------------------------------------------------------
# 11. RewardModelTrainer train_step loss is finite
# ---------------------------------------------------------------------------


def test_train_step_loss_is_finite(trainer, chosen_ids, rejected_ids):
    metrics = trainer.train_step(chosen_ids, rejected_ids)
    assert math.isfinite(metrics["loss"])
    assert metrics["loss"] > 0.0


# ---------------------------------------------------------------------------
# 12. RewardModelTrainer evaluate returns accuracy in [0, 1]
# ---------------------------------------------------------------------------


def test_evaluate_accuracy_in_range(trainer, chosen_ids, rejected_ids):
    result = trainer.evaluate(chosen_ids, rejected_ids)
    assert "accuracy" in result
    assert "reward_margin" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert math.isfinite(result["reward_margin"])


# ---------------------------------------------------------------------------
# 13. train_reward_model returns losses and accuracies lists
# ---------------------------------------------------------------------------


def test_train_reward_model_returns_history(backbone):
    torch.manual_seed(7)
    cfg = RMTrainerConfig(learning_rate=1e-4)
    data = [
        (torch.randint(0, 256, (2, 8)), torch.randint(0, 256, (2, 8))),
        (torch.randint(0, 256, (2, 8)), torch.randint(0, 256, (2, 8))),
    ]
    history = train_reward_model(backbone, data, cfg, n_epochs=2)
    assert "losses" in history
    assert "accuracies" in history
    # 2 batches x 2 epochs = 4 entries
    assert len(history["losses"]) == 4
    assert len(history["accuracies"]) == 4
    assert all(math.isfinite(line) for line in history["losses"])
    assert all(0.0 <= a <= 1.0 for a in history["accuracies"])
