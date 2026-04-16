"""Tests for MDPO (Mirror Descent Policy Optimization) — 12+ tests."""
from __future__ import annotations

import math
import torch
import pytest

from src.alignment.mdpo import (
    MDPOConfig,
    MDPOBatch,
    MDPOTrainer,
    mdpo_loss,
    sequence_log_probs,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Tiny model config: n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
# head_dim=16, d_ff=128, vocab_size=256, max_seq_len=32
TINY_CFG = dict(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=32,
)

B = 2
T = 16
PROMPT_LEN = 4


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(**TINY_CFG)
    return AureliusTransformer(cfg)


@pytest.fixture
def tiny_ref_model():
    torch.manual_seed(1)
    cfg = AureliusConfig(**TINY_CFG)
    return AureliusTransformer(cfg)


@pytest.fixture
def input_ids():
    torch.manual_seed(42)
    return torch.randint(0, 256, (B, T))


@pytest.fixture
def rewards():
    torch.manual_seed(7)
    return torch.randn(B)


# ---------------------------------------------------------------------------
# 1. MDPOConfig defaults are sensible
# ---------------------------------------------------------------------------

def test_mdpo_config_defaults():
    cfg = MDPOConfig()
    assert cfg.kl_coef == pytest.approx(0.1)
    assert cfg.lr == pytest.approx(1e-5)
    assert cfg.n_steps == 4
    assert cfg.max_grad_norm == pytest.approx(1.0)
    assert cfg.reward_scale == pytest.approx(1.0)
    assert cfg.entropy_coef == pytest.approx(0.01)
    assert cfg.max_seq_len == 64


# ---------------------------------------------------------------------------
# 2. sequence_log_probs returns (B, T-prompt_len-1) shape
# ---------------------------------------------------------------------------

def test_sequence_log_probs_shape(tiny_model, input_ids):
    lp = sequence_log_probs(tiny_model, input_ids, prompt_len=PROMPT_LEN)
    expected_len = T - PROMPT_LEN - 1
    assert lp.shape == (B, expected_len), (
        f"Expected ({B}, {expected_len}), got {lp.shape}"
    )


# ---------------------------------------------------------------------------
# 3. sequence_log_probs returns finite values
# ---------------------------------------------------------------------------

def test_sequence_log_probs_finite(tiny_model, input_ids):
    lp = sequence_log_probs(tiny_model, input_ids, prompt_len=PROMPT_LEN)
    assert torch.isfinite(lp).all(), "sequence_log_probs contains non-finite values"


# ---------------------------------------------------------------------------
# 4. sequence_log_probs values are negative (log probs <= 0)
# ---------------------------------------------------------------------------

def test_sequence_log_probs_non_positive(tiny_model, input_ids):
    lp = sequence_log_probs(tiny_model, input_ids, prompt_len=PROMPT_LEN)
    assert (lp <= 0).all(), "Log probs must be <= 0; found positive values"


# ---------------------------------------------------------------------------
# 5. mdpo_loss returns (scalar_tensor, dict) tuple
# ---------------------------------------------------------------------------

def test_mdpo_loss_returns_tuple(tiny_model, input_ids, rewards):
    S = T - PROMPT_LEN - 1
    torch.manual_seed(10)
    lp = torch.randn(B, S) - 5.0
    ref_lp = torch.randn(B, S) - 5.0
    result = mdpo_loss(lp, ref_lp, rewards)
    assert isinstance(result, tuple) and len(result) == 2
    loss, metrics = result
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()  # scalar
    assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# 6. mdpo_loss dict has keys: 'loss', 'reward', 'kl', 'entropy'
# ---------------------------------------------------------------------------

def test_mdpo_loss_dict_keys(rewards):
    S = T - PROMPT_LEN - 1
    torch.manual_seed(11)
    lp = torch.randn(B, S) - 5.0
    ref_lp = torch.randn(B, S) - 5.0
    _, metrics = mdpo_loss(lp, ref_lp, rewards)
    for key in ("loss", "reward", "kl", "entropy"):
        assert key in metrics, f"Missing key '{key}' in metrics dict"


# ---------------------------------------------------------------------------
# 7. mdpo_loss loss is finite
# ---------------------------------------------------------------------------

def test_mdpo_loss_finite(rewards):
    S = T - PROMPT_LEN - 1
    torch.manual_seed(12)
    lp = torch.randn(B, S) - 5.0
    ref_lp = torch.randn(B, S) - 5.0
    loss, _ = mdpo_loss(lp, ref_lp, rewards)
    assert math.isfinite(loss.item()), f"Loss is not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# 8. mdpo_loss KL is non-negative (given log_probs >= ref_log_probs on avg)
#    We test that when policy == ref, KL == 0
# ---------------------------------------------------------------------------

def test_mdpo_loss_kl_zero_when_equal(rewards):
    S = T - PROMPT_LEN - 1
    torch.manual_seed(13)
    lp = torch.randn(B, S) - 5.0
    # KL should be 0 when log_probs == ref_log_probs
    _, metrics = mdpo_loss(lp, lp.clone(), rewards)
    assert abs(metrics["kl"]) < 1e-5, f"KL should be 0 when policies equal, got {metrics['kl']}"


# Test KL can be non-negative in general use
def test_mdpo_loss_kl_nonneg_typical(rewards):
    S = T - PROMPT_LEN - 1
    torch.manual_seed(14)
    # policy log probs slightly above ref => positive KL
    ref_lp = torch.randn(B, S) - 5.0
    lp = ref_lp + 0.5  # log pi > log pi_ref => KL > 0
    _, metrics = mdpo_loss(lp, ref_lp, rewards)
    assert metrics["kl"] >= 0.0, f"Expected non-negative KL, got {metrics['kl']}"


# ---------------------------------------------------------------------------
# 9. mdpo_loss entropy is non-negative
# ---------------------------------------------------------------------------

def test_mdpo_loss_entropy_nonneg(rewards):
    S = T - PROMPT_LEN - 1
    torch.manual_seed(15)
    # log_probs <= 0 => entropy = -mean(log_probs) >= 0
    lp = -torch.rand(B, S) * 5.0  # values in [-5, 0]
    ref_lp = lp.clone()
    _, metrics = mdpo_loss(lp, ref_lp, rewards)
    assert metrics["entropy"] >= 0.0, (
        f"Entropy should be non-negative, got {metrics['entropy']}"
    )


# ---------------------------------------------------------------------------
# 10. MDPOTrainer constructs without error
# ---------------------------------------------------------------------------

def test_mdpo_trainer_constructs(tiny_model, tiny_ref_model):
    cfg = MDPOConfig()
    trainer = MDPOTrainer(tiny_model, tiny_ref_model, cfg)
    assert trainer.model is tiny_model
    assert trainer.ref_model is tiny_ref_model
    assert trainer.config is cfg


# ---------------------------------------------------------------------------
# 11. MDPOTrainer.make_batch returns MDPOBatch with correct ref_log_probs shape
# ---------------------------------------------------------------------------

def test_mdpo_trainer_make_batch_shape(tiny_model, tiny_ref_model, input_ids, rewards):
    torch.manual_seed(20)
    cfg = MDPOConfig()
    trainer = MDPOTrainer(tiny_model, tiny_ref_model, cfg)
    batch = trainer.make_batch(input_ids, rewards, prompt_len=PROMPT_LEN)
    assert isinstance(batch, MDPOBatch)
    expected_S = T - PROMPT_LEN - 1
    assert batch.ref_log_probs.shape == (B, expected_S), (
        f"Expected ref_log_probs shape ({B}, {expected_S}), got {batch.ref_log_probs.shape}"
    )
    assert batch.prompt_len == PROMPT_LEN


# ---------------------------------------------------------------------------
# 12. MDPOTrainer.train_step returns dict with required keys
# ---------------------------------------------------------------------------

def test_mdpo_trainer_train_step_keys(tiny_model, tiny_ref_model, input_ids, rewards):
    torch.manual_seed(30)
    cfg = MDPOConfig(n_steps=1)  # 1 step for speed
    trainer = MDPOTrainer(tiny_model, tiny_ref_model, cfg)
    batch = trainer.make_batch(input_ids, rewards, prompt_len=PROMPT_LEN)
    metrics = trainer.train_step(batch)
    for key in ("loss", "reward", "kl", "entropy"):
        assert key in metrics, f"Missing key '{key}' in train_step metrics"


# ---------------------------------------------------------------------------
# Bonus: train_step returns finite metrics
# ---------------------------------------------------------------------------

def test_mdpo_trainer_train_step_finite(tiny_model, tiny_ref_model, input_ids, rewards):
    torch.manual_seed(31)
    cfg = MDPOConfig(n_steps=1)
    trainer = MDPOTrainer(tiny_model, tiny_ref_model, cfg)
    batch = trainer.make_batch(input_ids, rewards, prompt_len=PROMPT_LEN)
    metrics = trainer.train_step(batch)
    for key, val in metrics.items():
        assert math.isfinite(val), f"Metric '{key}' is not finite: {val}"
