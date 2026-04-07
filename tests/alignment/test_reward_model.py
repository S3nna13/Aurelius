"""Tests for the reward model."""
import torch
import pytest
from src.alignment.reward_model import RewardModel, bradley_terry_loss, build_reward_fn, RewardModelTrainer, RewardModelConfig
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_backbone():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )
    return AureliusTransformer(cfg)


def test_reward_model_output_shape(small_backbone):
    """RewardModel.forward must return (B,) shaped tensor."""
    rm = RewardModel(small_backbone)
    ids = torch.randint(0, 256, (3, 16))
    scores = rm(ids)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()


def test_reward_model_score_returns_float(small_backbone):
    """score() must return a Python float."""
    rm = RewardModel(small_backbone)
    ids = torch.randint(0, 256, (1, 8))
    s = rm.score(ids)
    assert isinstance(s, float)
    import math
    assert math.isfinite(s)


def test_bradley_terry_loss_scalar():
    """Bradley-Terry loss must be a finite positive scalar."""
    chosen = torch.tensor([1.0, 2.0, 0.5])
    rejected = torch.tensor([0.0, 0.5, 1.0])
    loss = bradley_terry_loss(chosen, rejected)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss > 0


def test_bradley_terry_loss_equal_is_log2():
    """When chosen == rejected, loss = log(2) approx 0.693."""
    import math
    scores = torch.zeros(4)
    loss = bradley_terry_loss(scores, scores)
    assert abs(loss.item() - math.log(2)) < 0.01


def test_freeze_backbone(small_backbone):
    """freeze_backbone=True must make only reward_head trainable."""
    rm = RewardModel(small_backbone, freeze_backbone=True)
    trainable = [n for n, p in rm.named_parameters() if p.requires_grad]
    assert all("reward_head" in n for n in trainable)
    assert len(trainable) > 0  # head must be trainable


def test_train_step_updates_head(small_backbone):
    """train_step must update reward_head weights."""
    rm = RewardModel(small_backbone, freeze_backbone=True)
    trainer = RewardModelTrainer(rm)

    before = rm.reward_head.weight.clone()

    chosen = torch.randint(0, 256, (2, 16))
    rejected = torch.randint(0, 256, (2, 16))
    loss = trainer.train_step(chosen, rejected)

    import math
    assert math.isfinite(loss)
    assert not torch.equal(rm.reward_head.weight, before), "Reward head weights not updated"


def test_build_reward_fn(small_backbone):
    """build_reward_fn must return a callable that returns a float."""
    from unittest.mock import MagicMock
    rm = RewardModel(small_backbone)

    tok = MagicMock()
    tok.encode = lambda text: [1, 2, 3, 4, 5]

    fn = build_reward_fn(rm, tok)
    result = fn("hello", "world")
    import math
    assert isinstance(result, float)
    assert math.isfinite(result)
