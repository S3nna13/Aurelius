"""Tests for DP-SGD training module (dp_training.py)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.dp_training import (
    DPConfig,
    DPTrainer,
    PrivacyAccountant,
    add_dp_noise,
    clip_gradients,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SMALL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

BATCH_SIZE = 2
SEQ_LEN = 8
DATASET_SIZE = 100


@pytest.fixture
def dp_config():
    return DPConfig()


@pytest.fixture
def accountant(dp_config):
    return PrivacyAccountant(dp_config, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE)


@pytest.fixture
def small_model():
    torch.manual_seed(42)
    return AureliusTransformer(SMALL_CFG)


@pytest.fixture
def dp_trainer(small_model, dp_config):
    optimizer = torch.optim.SGD(small_model.parameters(), lr=1e-3)
    return DPTrainer(
        model=small_model,
        optimizer=optimizer,
        config=dp_config,
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
    )


@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, SMALL_CFG.vocab_size, (BATCH_SIZE, SEQ_LEN))


# ---------------------------------------------------------------------------
# 1. DPConfig defaults
# ---------------------------------------------------------------------------


def test_dpconfig_defaults():
    cfg = DPConfig()
    assert cfg.max_grad_norm == 1.0
    assert cfg.noise_multiplier == 1.0
    assert cfg.delta == 1e-5
    assert cfg.target_epsilon == 8.0
    assert cfg.accounting_method == "rdp"


# ---------------------------------------------------------------------------
# 2. PrivacyAccountant sampling rate
# ---------------------------------------------------------------------------


def test_privacy_accountant_sampling_rate(accountant):
    expected = BATCH_SIZE / DATASET_SIZE
    assert abs(accountant.sampling_rate - expected) < 1e-9


# ---------------------------------------------------------------------------
# 3. compute_rdp is positive
# ---------------------------------------------------------------------------


def test_compute_rdp_positive(accountant):
    rdp = accountant.compute_rdp(steps=10)
    assert rdp > 0.0


# ---------------------------------------------------------------------------
# 4. compute_epsilon increases with steps
# ---------------------------------------------------------------------------


def test_compute_epsilon_increases_with_steps(accountant):
    eps1 = accountant.compute_epsilon(steps=1)
    eps2 = accountant.compute_epsilon(steps=100)
    assert eps2 > eps1


# ---------------------------------------------------------------------------
# 5. Larger noise_multiplier → smaller epsilon
# ---------------------------------------------------------------------------


def test_larger_noise_smaller_epsilon():
    cfg_low_noise = DPConfig(noise_multiplier=0.5)
    cfg_high_noise = DPConfig(noise_multiplier=2.0)
    acc_low = PrivacyAccountant(cfg_low_noise, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE)
    acc_high = PrivacyAccountant(cfg_high_noise, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE)
    eps_low = acc_low.compute_epsilon(steps=10)
    eps_high = acc_high.compute_epsilon(steps=10)
    assert eps_low > eps_high


# ---------------------------------------------------------------------------
# 6. is_budget_exceeded returns False at step 0
# ---------------------------------------------------------------------------


def test_budget_not_exceeded_at_step_zero(accountant):
    assert accountant.is_budget_exceeded(steps=0) is False


# ---------------------------------------------------------------------------
# 7. is_budget_exceeded can return True with tiny target epsilon
# ---------------------------------------------------------------------------


def test_budget_exceeded_with_small_target():
    cfg = DPConfig(target_epsilon=0.0001, noise_multiplier=0.1)
    acc = PrivacyAccountant(cfg, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE)
    # After enough steps with low noise, budget should be exceeded
    assert acc.is_budget_exceeded(steps=1000) is True


# ---------------------------------------------------------------------------
# 8. clip_gradients returns a float (avg grad norm)
# ---------------------------------------------------------------------------


def test_clip_gradients_returns_float():
    params = [nn.Parameter(torch.randn(4, 4)) for _ in range(3)]
    for p in params:
        p.grad = torch.randn_like(p)
    result = clip_gradients(params, max_grad_norm=1.0)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 9. clip_gradients actually clips grad norms to ≤ max_grad_norm
# ---------------------------------------------------------------------------


def test_clip_gradients_clips_norms():
    max_norm = 1.0
    params = [nn.Parameter(torch.ones(10) * 10.0) for _ in range(3)]
    for p in params:
        p.grad = torch.ones_like(p) * 10.0  # large grad

    clip_gradients(params, max_grad_norm=max_norm)

    for p in params:
        assert p.grad.norm(2).item() <= max_norm + 1e-6


# ---------------------------------------------------------------------------
# 10. add_dp_noise modifies gradients
# ---------------------------------------------------------------------------


def test_add_dp_noise_modifies_gradients():
    torch.manual_seed(123)
    params = [nn.Parameter(torch.zeros(5)) for _ in range(2)]
    for p in params:
        p.grad = torch.zeros_like(p)

    original_grads = [p.grad.clone() for p in params]
    add_dp_noise(params, noise_multiplier=1.0, max_grad_norm=1.0, batch_size=BATCH_SIZE)

    # At least one gradient should have changed
    changed = any(not torch.allclose(p.grad, orig) for p, orig in zip(params, original_grads))
    assert changed


# ---------------------------------------------------------------------------
# 11. DPTrainer.train_step returns required keys
# ---------------------------------------------------------------------------


def test_train_step_returns_required_keys(dp_trainer, input_ids):
    result = dp_trainer.train_step(input_ids)
    required_keys = {"loss", "epsilon", "grad_norm", "steps", "budget_exceeded"}
    assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# 12. DPTrainer.train_step epsilon increases over multiple steps
# ---------------------------------------------------------------------------


def test_train_step_epsilon_increases(dp_trainer, input_ids):
    result1 = dp_trainer.train_step(input_ids)
    result2 = dp_trainer.train_step(input_ids)
    assert result2["epsilon"] > result1["epsilon"]


# ---------------------------------------------------------------------------
# 13. DPTrainer.train_step budget_exceeded starts False
# ---------------------------------------------------------------------------


def test_train_step_budget_not_exceeded_initially(small_model, input_ids):
    # Use a generous target_epsilon so budget is not exceeded on the first step.
    # With noise_multiplier=10.0 and target_epsilon=100.0, epsilon(1) will be well below target.
    cfg = DPConfig(noise_multiplier=10.0, target_epsilon=100.0)
    optimizer = torch.optim.SGD(small_model.parameters(), lr=1e-3)
    trainer = DPTrainer(
        model=small_model,
        optimizer=optimizer,
        config=cfg,
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
    )
    result = trainer.train_step(input_ids)
    assert result["budget_exceeded"] is False


# ---------------------------------------------------------------------------
# 14. DPTrainer.get_privacy_spent returns (epsilon, delta) tuple
# ---------------------------------------------------------------------------


def test_get_privacy_spent_returns_tuple(dp_trainer, input_ids):
    dp_trainer.train_step(input_ids)
    result = dp_trainer.get_privacy_spent()
    assert isinstance(result, tuple)
    assert len(result) == 2
    epsilon, delta = result
    assert isinstance(epsilon, float)
    assert abs(delta - dp_trainer.config.delta) < 1e-12


# ---------------------------------------------------------------------------
# 15. DPConfig noise_multiplier affects privacy
# ---------------------------------------------------------------------------


def test_noise_multiplier_affects_privacy():
    """Higher noise_multiplier should yield lower epsilon (more private)."""
    steps = 50

    cfg_tight = DPConfig(noise_multiplier=0.5)
    acc_tight = PrivacyAccountant(cfg_tight, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE)

    cfg_loose = DPConfig(noise_multiplier=5.0)
    acc_loose = PrivacyAccountant(cfg_loose, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE)

    assert acc_tight.compute_epsilon(steps) > acc_loose.compute_epsilon(steps)
