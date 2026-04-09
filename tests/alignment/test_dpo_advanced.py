"""Tests for advanced DPO variants: DPO, IPO, and SLiC losses."""

from __future__ import annotations

import math
import pytest
import torch
import torch.optim as optim

from src.alignment.dpo_advanced import (
    DPOAdvancedConfig,
    compute_log_probs,
    dpo_loss,
    ipo_loss,
    slic_loss,
    DPOAdvancedTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=512,
    )


@pytest.fixture
def policy_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def ref_model(small_cfg):
    torch.manual_seed(1)
    m = AureliusTransformer(small_cfg)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _make_ids(batch_size=2, seq_len=16, vocab_size=256, seed=42):
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (batch_size, seq_len))


# ---------------------------------------------------------------------------
# Test 1: DPOAdvancedConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = DPOAdvancedConfig()
    assert cfg.beta == 0.1
    assert cfg.loss_type == "dpo"
    assert cfg.label_smoothing == 0.0
    assert cfg.slic_delta == 1.0
    assert cfg.reference_free is False


# ---------------------------------------------------------------------------
# Test 2: dpo_loss returns scalar
# ---------------------------------------------------------------------------

def test_dpo_loss_scalar_output():
    B = 4
    policy_chosen = torch.randn(B)
    policy_rejected = torch.randn(B)
    ref_chosen = torch.randn(B)
    ref_rejected = torch.randn(B)

    loss, margin = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
    assert loss.ndim == 0, "loss must be a scalar"
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 3: dpo_loss is small when chosen >> rejected
# ---------------------------------------------------------------------------

def test_dpo_loss_chosen_better_gives_small_loss():
    B = 4
    # Large advantage for chosen → chosen_ratio >> rejected_ratio → small loss
    # h = beta * (chosen_ratio - rejected_ratio) = 10.0 * (10 - (-10)) = 200
    # -log_sigmoid(200) ≈ 0
    policy_chosen = torch.full((B,), 10.0)
    policy_rejected = torch.full((B,), -10.0)
    ref_chosen = torch.zeros(B)
    ref_rejected = torch.zeros(B)

    loss, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=10.0)
    assert loss.item() < 0.01, f"Expected small loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 4: dpo_loss with label smoothing
# ---------------------------------------------------------------------------

def test_dpo_loss_label_smoothing():
    B = 4
    policy_chosen = torch.randn(B)
    policy_rejected = torch.randn(B)
    ref_chosen = torch.zeros(B)
    ref_rejected = torch.zeros(B)

    loss_no_smooth, _ = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected,
        beta=0.1, label_smoothing=0.0
    )
    loss_smooth, _ = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected,
        beta=0.1, label_smoothing=0.1
    )
    # Label smoothing should produce a different (typically higher) loss
    assert loss_no_smooth.item() != pytest.approx(loss_smooth.item(), abs=1e-4), \
        "Label smoothing should change the loss"
    assert torch.isfinite(loss_smooth)


# ---------------------------------------------------------------------------
# Test 5: ipo_loss returns scalar
# ---------------------------------------------------------------------------

def test_ipo_loss_scalar_output():
    B = 4
    policy_chosen = torch.randn(B)
    policy_rejected = torch.randn(B)
    ref_chosen = torch.randn(B)
    ref_rejected = torch.randn(B)

    loss, margin = ipo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
    assert loss.ndim == 0, "IPO loss must be a scalar"
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 6: ipo_loss is near zero at optimum (h = 1/(2*beta))
# ---------------------------------------------------------------------------

def test_ipo_loss_at_optimum():
    beta = 0.1
    target = 1.0 / (2.0 * beta)  # = 5.0

    B = 4
    # Set h = chosen_ratio - rejected_ratio = target by using policy > ref
    policy_chosen = torch.full((B,), target)
    policy_rejected = torch.zeros(B)
    ref_chosen = torch.zeros(B)
    ref_rejected = torch.zeros(B)

    loss, _ = ipo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta)
    assert abs(loss.item()) < 1e-5, f"IPO loss at optimum should be ~0, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 7: slic_loss returns scalar
# ---------------------------------------------------------------------------

def test_slic_loss_scalar_output():
    B = 4
    policy_chosen = torch.randn(B)
    policy_rejected = torch.randn(B)
    ref_chosen = torch.randn(B)
    ref_rejected = torch.randn(B)

    loss, margin = slic_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, delta=1.0)
    assert loss.ndim == 0, "SLiC loss must be a scalar"
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0, "SLiC loss (hinge) must be non-negative"


# ---------------------------------------------------------------------------
# Test 8: slic_loss is zero when margin exceeds delta
# ---------------------------------------------------------------------------

def test_slic_loss_satisfied_margin():
    delta = 1.0
    B = 4
    # chosen >> rejected → margin >> delta → hinge is 0
    policy_chosen = torch.full((B,), 10.0)
    policy_rejected = torch.full((B,), -10.0)
    ref_chosen = torch.zeros(B)
    ref_rejected = torch.zeros(B)

    loss, _ = slic_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, delta=delta)
    assert abs(loss.item()) < 1e-6, f"SLiC loss should be 0 when margin is satisfied, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 9: compute_log_probs shape is (B,)
# ---------------------------------------------------------------------------

def test_compute_log_probs_shape(policy_model):
    B, seq_len = 2, 16
    input_ids = _make_ids(batch_size=B, seq_len=seq_len)
    labels = input_ids.clone()
    labels[:, -1] = -100  # mask last token

    lp = compute_log_probs(policy_model, input_ids, labels)
    assert lp.shape == (B,), f"Expected shape ({B},), got {lp.shape}"
    assert torch.isfinite(lp).all(), "Log-probs must be finite"
    assert (lp <= 0).all(), "Log-probs must be non-positive"


# ---------------------------------------------------------------------------
# Test 10: DPOAdvancedTrainer with loss_type="dpo" returns correct keys
# ---------------------------------------------------------------------------

def test_trainer_dpo_returns_keys(policy_model, ref_model):
    cfg = DPOAdvancedConfig(beta=0.1, loss_type="dpo")
    optimizer = optim.SGD(policy_model.parameters(), lr=1e-4)
    trainer = DPOAdvancedTrainer(policy_model, ref_model, cfg, optimizer)

    chosen_ids = _make_ids(batch_size=2, seq_len=16, seed=10)
    rejected_ids = _make_ids(batch_size=2, seq_len=16, seed=20)

    result = trainer.train_step(chosen_ids, rejected_ids)

    assert set(result.keys()) == {"loss", "reward_margin", "chosen_reward", "rejected_reward"}
    assert isinstance(result["loss"], float)
    assert isinstance(result["reward_margin"], float)
    assert math.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# Test 11: DPOAdvancedTrainer with loss_type="ipo" returns correct keys
# ---------------------------------------------------------------------------

def test_trainer_ipo_returns_keys(policy_model, ref_model):
    cfg = DPOAdvancedConfig(beta=0.1, loss_type="ipo")
    optimizer = optim.SGD(policy_model.parameters(), lr=1e-4)
    trainer = DPOAdvancedTrainer(policy_model, ref_model, cfg, optimizer)

    chosen_ids = _make_ids(batch_size=2, seq_len=16, seed=11)
    rejected_ids = _make_ids(batch_size=2, seq_len=16, seed=21)

    result = trainer.train_step(chosen_ids, rejected_ids)

    assert set(result.keys()) == {"loss", "reward_margin", "chosen_reward", "rejected_reward"}
    assert isinstance(result["loss"], float)
    assert math.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# Test 12: DPOAdvancedTrainer with reference_free=True (no ref model needed)
# ---------------------------------------------------------------------------

def test_trainer_reference_free(policy_model):
    cfg = DPOAdvancedConfig(beta=0.1, loss_type="dpo", reference_free=True)
    optimizer = optim.SGD(policy_model.parameters(), lr=1e-4)
    # ref_model is None — should be fine with reference_free=True
    trainer = DPOAdvancedTrainer(policy_model, ref_model=None, config=cfg, optimizer=optimizer)

    chosen_ids = _make_ids(batch_size=2, seq_len=16, seed=12)
    rejected_ids = _make_ids(batch_size=2, seq_len=16, seed=22)

    result = trainer.train_step(chosen_ids, rejected_ids)

    assert set(result.keys()) == {"loss", "reward_margin", "chosen_reward", "rejected_reward"}
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["reward_margin"])
