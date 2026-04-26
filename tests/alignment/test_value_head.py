import pytest
import torch

from src.alignment.value_head import PPOConfig, ValueHead, ppo_loss
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_backbone():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    torch.manual_seed(0)
    return AureliusTransformer(cfg)


@pytest.fixture
def value_model(small_backbone):
    return ValueHead(small_backbone, PPOConfig(d_model=64))


def test_value_head_output_shapes(value_model):
    input_ids = torch.randint(0, 256, (2, 8))
    loss, logits, values = value_model(input_ids)
    assert logits.shape == (2, 8, 256)
    assert values.shape == (2, 8)


def test_value_head_no_loss_without_labels(value_model):
    input_ids = torch.randint(0, 256, (1, 6))
    loss, logits, values = value_model(input_ids)
    assert loss is None


def test_value_head_loss_with_labels(value_model):
    input_ids = torch.randint(0, 256, (1, 6))
    loss, logits, values = value_model(input_ids, labels=input_ids)
    assert loss is not None
    import math

    assert math.isfinite(loss.item())


def test_value_head_values_finite(value_model):
    import math

    input_ids = torch.randint(0, 256, (2, 8))
    _, _, values = value_model(input_ids)
    assert all(math.isfinite(v) for v in values.flatten().tolist())


def test_ppo_loss_returns_scalar():
    B, S, V = 2, 8, 64
    logits_new = torch.randn(B, S, V)
    logits_old = torch.randn(B, S, V)
    values = torch.randn(B, S)
    targets = torch.randn(B, S)
    advantages = torch.randn(B, S)
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[:, :2] = False  # first 2 tokens are prompt

    cfg = PPOConfig(d_model=64)
    loss, metrics = ppo_loss(logits_new, logits_old, values, targets, advantages, mask, cfg)
    assert loss.ndim == 0
    import math

    assert math.isfinite(loss.item())


def test_ppo_loss_metrics_keys():
    B, S, V = 1, 6, 32
    logits_new = torch.randn(B, S, V)
    logits_old = logits_new.clone()  # identical -> ratio = 1
    values = torch.zeros(B, S)
    targets = torch.zeros(B, S)
    advantages = torch.zeros(B, S)
    mask = torch.ones(B, S, dtype=torch.bool)

    cfg = PPOConfig(d_model=64)
    _, metrics = ppo_loss(logits_new, logits_old, values, targets, advantages, mask, cfg)
    assert all(k in metrics for k in ["policy_loss", "value_loss", "entropy", "ratio_mean"])


def test_ppo_loss_identical_policies_ratio_one():
    """When new == old policy, ratio should be ~1."""
    B, S, V = 1, 6, 32
    logits = torch.randn(B, S, V)
    values = torch.zeros(B, S)
    advantages = torch.zeros(B, S)
    mask = torch.ones(B, S, dtype=torch.bool)
    cfg = PPOConfig(d_model=64)
    _, metrics = ppo_loss(logits, logits.clone(), values, values, advantages, mask, cfg)
    assert abs(metrics["ratio_mean"] - 1.0) < 0.01
