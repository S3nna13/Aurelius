"""Tests for KTO (Kahneman-Tversky Optimization) implementation."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.alignment.kto import KTODataset, KTOLoss, KTOTrainer
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def tiny_model(tiny_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(tiny_cfg)


@pytest.fixture
def kto_loss():
    return KTOLoss(beta=0.1)


def _make_logps(batch_size: int, seed: int = 42):
    torch.manual_seed(seed)
    policy_logps = -torch.rand(batch_size) * 5.0
    ref_logps = -torch.rand(batch_size) * 5.0
    return policy_logps, ref_logps


def _mixed_labels(batch_size: int) -> torch.Tensor:
    labels = torch.zeros(batch_size, dtype=torch.long)
    labels[: batch_size // 2] = 1
    return labels


# ---------------------------------------------------------------------------
# KTOLoss tests
# ---------------------------------------------------------------------------

def test_kto_loss_scalar(kto_loss):
    """forward returns (loss, dict); loss is scalar."""
    policy_logps, ref_logps = _make_logps(4)
    labels = _mixed_labels(4)
    loss, metrics = kto_loss(policy_logps, ref_logps, labels)
    assert loss.ndim == 0, "loss should be a scalar"
    assert isinstance(metrics, dict)


def test_kto_loss_positive(kto_loss):
    """loss > 0."""
    policy_logps, ref_logps = _make_logps(4)
    labels = _mixed_labels(4)
    loss, _ = kto_loss(policy_logps, ref_logps, labels)
    assert loss.item() > 0, "KTO loss should be positive"


def test_kto_metrics_keys(kto_loss):
    """metrics dict has kl, z_ref, desirable_reward, undesirable_reward, reward_margin."""
    policy_logps, ref_logps = _make_logps(4)
    labels = _mixed_labels(4)
    _, metrics = kto_loss(policy_logps, ref_logps, labels)
    required_keys = {"kl", "z_ref", "desirable_reward", "undesirable_reward", "reward_margin"}
    assert required_keys.issubset(metrics.keys()), (
        f"Missing keys: {required_keys - set(metrics.keys())}"
    )


def test_kto_reward_margin_sign(kto_loss):
    """When policy >> ref for desirable, reward_margin > 0."""
    policy_logps = torch.tensor([-0.1, -0.2, -0.3, -4.0, -4.5, -5.0])
    ref_logps = torch.tensor([-4.0, -4.5, -5.0, -0.1, -0.2, -0.3])
    labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.long)

    _, metrics = kto_loss(policy_logps, ref_logps, labels)
    assert metrics["reward_margin"] > 0, (
        f"Expected reward_margin > 0, got {metrics['reward_margin']}"
    )


def test_kto_all_desirable(kto_loss):
    """Batch of all desirable labels works without error."""
    policy_logps, ref_logps = _make_logps(4)
    labels = torch.ones(4, dtype=torch.long)
    loss, metrics = kto_loss(policy_logps, ref_logps, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_kto_all_undesirable(kto_loss):
    """Batch of all undesirable labels works without error."""
    policy_logps, ref_logps = _make_logps(4)
    labels = torch.zeros(4, dtype=torch.long)
    loss, metrics = kto_loss(policy_logps, ref_logps, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_kto_compute_rewards_shape(kto_loss):
    """compute_rewards returns (B,) tensor."""
    policy_logps, ref_logps = _make_logps(8)
    rewards = kto_loss.compute_rewards(policy_logps, ref_logps)
    assert rewards.shape == (8,), f"Expected (8,), got {rewards.shape}"


def test_kto_kl_nonneg(kto_loss):
    """estimate_kl returns non-negative value (KL >= 0 in expectation)."""
    policy_logps = torch.tensor([-1.0, -1.5, -2.0, -0.5])
    ref_logps = torch.tensor([-3.0, -3.5, -4.0, -2.5])
    kl = kto_loss.estimate_kl(policy_logps, ref_logps)
    assert kl.ndim == 0, "estimate_kl should return a scalar"
    assert kl.item() >= 0, f"Expected KL >= 0, got {kl.item()}"


# ---------------------------------------------------------------------------
# KTOTrainer tests
# ---------------------------------------------------------------------------

def test_kto_trainer_step_keys(tiny_model):
    """train_step returns dict with loss key."""
    ref_model = copy.deepcopy(tiny_model)
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)
    trainer = KTOTrainer(tiny_model, ref_model, optimizer, beta=0.1)

    torch.manual_seed(10)
    prompt_ids = torch.randint(0, 256, (1, 8))
    response_ids = torch.randint(0, 256, (1, 4))

    metrics = trainer.train_step(prompt_ids, response_ids, label=1)
    assert "loss" in metrics, f"Expected 'loss' in metrics, got keys: {list(metrics.keys())}"


# ---------------------------------------------------------------------------
# KTODataset tests
# ---------------------------------------------------------------------------

def test_kto_dataset_add_and_len():
    """dataset grows after add()."""
    dataset = KTODataset()
    assert len(dataset) == 0

    prompt_ids = torch.randint(0, 256, (1, 8))
    response_ids = torch.randint(0, 256, (1, 4))

    dataset.add(prompt_ids, response_ids, label=1)
    assert len(dataset) == 1

    dataset.add(prompt_ids, response_ids, label=0)
    assert len(dataset) == 2


def test_kto_dataset_label_distribution():
    """label_distribution returns correct counts."""
    dataset = KTODataset()
    prompt_ids = torch.randint(0, 256, (1, 8))
    response_ids = torch.randint(0, 256, (1, 4))

    for _ in range(3):
        dataset.add(prompt_ids, response_ids, label=1)
    for _ in range(2):
        dataset.add(prompt_ids, response_ids, label=0)

    dist = dataset.label_distribution()
    assert dist["n_desirable"] == 3
    assert dist["n_undesirable"] == 2
    assert abs(dist["ratio"] - 3 / 5) < 1e-6


def test_kto_get_batch():
    """get_batch returns list of correct length."""
    dataset = KTODataset()
    prompt_ids = torch.randint(0, 256, (1, 8))
    response_ids = torch.randint(0, 256, (1, 4))

    for i in range(10):
        dataset.add(prompt_ids, response_ids, label=i % 2)

    batch = dataset.get_batch(4)
    assert isinstance(batch, list)
    assert len(batch) == 4

    for ex in batch:
        assert "prompt_ids" in ex
        assert "response_ids" in ex
        assert "label" in ex
