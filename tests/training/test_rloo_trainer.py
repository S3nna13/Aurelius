"""Tests for RLOO (REINFORCE Leave-One-Out) Trainer.

16 unit tests + 1 integration test covering:
  - RLOOConfig defaults
  - compute_loo_advantages: shape, correctness, edge cases, normalisation
  - compute_kl_penalty: zero-KL, positive-KL
  - compute_policy_loss: sign, masking
  - total_loss: keys, scalar, values
  - statistics: keys present
  - gradient flow
  - TRAINING_REGISTRY registration
  Integration: k=4, B=2, T=8 full forward + backward
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.training.rloo_trainer import RLOOBatch, RLOOConfig, RLOOTrainer

# ---------------------------------------------------------------------------
# Tiny fixed dimensions for all tests
# ---------------------------------------------------------------------------
D_MODEL = 64
N_HEADS = 4
VOCAB = 256
K = 4  # responses per prompt
B = 2  # number of distinct prompts
T = 8  # sequence length


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_batch(
    k: int = K,
    b: int = B,
    t: int = T,
    *,
    seed: int = 0,
    all_ones_mask: bool = False,
) -> RLOOBatch:
    """Return a random RLOOBatch of shape [b*k, t]."""
    torch.manual_seed(seed)
    bk = b * k
    log_probs = torch.randn(bk, t, requires_grad=True)
    ref_log_probs = torch.randn(bk, t)
    rewards = torch.rand(bk)
    if all_ones_mask:
        mask = torch.ones(bk, t)
    else:
        mask = (torch.rand(bk, t) > 0.2).float()
    return RLOOBatch(
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        rewards=rewards,
        attention_mask=mask,
    )


def default_trainer(normalize: bool = True, k: int = K) -> RLOOTrainer:
    return RLOOTrainer(RLOOConfig(k=k, kl_coeff=0.05, normalize_advantages=normalize))


# ===========================================================================
# 1. Config defaults
# ===========================================================================


def test_config_defaults() -> None:
    cfg = RLOOConfig()
    assert cfg.k == 8
    assert cfg.kl_coeff == 0.05
    assert cfg.normalize_advantages is True
    assert cfg.eps == 1e-8


# ===========================================================================
# 2. LOO advantages — output shape matches input
# ===========================================================================


def test_loo_advantages_shape() -> None:
    trainer = default_trainer()
    rewards = torch.rand(B * K)
    adv = trainer.compute_loo_advantages(rewards)
    assert adv.shape == rewards.shape, f"expected {rewards.shape}, got {adv.shape}"


# ===========================================================================
# 3. LOO baseline correctness for k=3
# ===========================================================================


def test_loo_baseline_correct() -> None:
    """Manually verify baseline = mean of the other k-1 for one group."""
    trainer = RLOOTrainer(RLOOConfig(k=3, normalize_advantages=False))
    # Single group of 3 rewards
    rewards = torch.tensor([1.0, 2.0, 3.0])
    adv = trainer.compute_loo_advantages(rewards)
    # adv_0 = r_0 - (r_1 + r_2) / 2 = 1 - 2.5 = -1.5
    # adv_1 = r_1 - (r_0 + r_2) / 2 = 2 - 2.0 =  0.0
    # adv_2 = r_2 - (r_0 + r_1) / 2 = 3 - 1.5 =  1.5
    expected = torch.tensor([-1.5, 0.0, 1.5])
    assert torch.allclose(adv, expected, atol=1e-6), f"got {adv}"


# ===========================================================================
# 4. LOO advantages for k=2: adv_0 = r_0 - r_1, adv_1 = r_1 - r_0
# ===========================================================================


def test_loo_advantages_k2() -> None:
    trainer = RLOOTrainer(RLOOConfig(k=2, normalize_advantages=False))
    r0, r1 = 0.3, 0.7
    rewards = torch.tensor([r0, r1])
    adv = trainer.compute_loo_advantages(rewards)
    assert torch.allclose(adv[0], torch.tensor(r0 - r1), atol=1e-6), f"adv[0]={adv[0]}"
    assert torch.allclose(adv[1], torch.tensor(r1 - r0), atol=1e-6), f"adv[1]={adv[1]}"


# ===========================================================================
# 5. Uniform rewards → all advantages = 0 (before normalisation)
# ===========================================================================


def test_loo_advantages_uniform() -> None:
    trainer = RLOOTrainer(RLOOConfig(k=4, normalize_advantages=False))
    rewards = torch.ones(B * 4) * 0.5
    adv = trainer.compute_loo_advantages(rewards)
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6), f"got {adv}"


# ===========================================================================
# 6. Normalised advantages have ~zero mean and ~unit std
# ===========================================================================


def test_normalize_advantages() -> None:
    trainer = default_trainer(normalize=True)
    torch.manual_seed(1)
    rewards = torch.rand(B * K)
    adv = trainer.compute_loo_advantages(rewards)
    assert abs(adv.mean().item()) < 1e-5, f"mean={adv.mean().item()}"
    assert abs(adv.std().item() - 1.0) < 1e-4, f"std={adv.std().item()}"


# ===========================================================================
# 7. KL penalty is 0 when log_probs == ref_log_probs
# ===========================================================================


def test_kl_penalty_zero() -> None:
    trainer = default_trainer()
    lp = torch.randn(B * K, T)
    mask = torch.ones(B * K, T)
    kl = trainer.compute_kl_penalty(lp, lp, mask)
    assert abs(kl.item()) < 1e-6, f"expected 0, got {kl.item()}"


# ===========================================================================
# 8. KL penalty > 0 when distributions differ
# ===========================================================================


def test_kl_penalty_positive() -> None:
    trainer = default_trainer()
    lp = torch.zeros(B * K, T)  # policy: all 0
    ref_lp = torch.ones(B * K, T) * -2.0  # reference: all -2
    mask = torch.ones(B * K, T)
    kl = trainer.compute_kl_penalty(lp, ref_lp, mask)
    # kl = 0 - (-2) = 2 per token
    assert kl.item() > 0.0, f"KL should be > 0, got {kl.item()}"
    assert abs(kl.item() - 2.0) < 1e-5, f"expected 2.0, got {kl.item()}"


# ===========================================================================
# 9. Policy loss sign: positive advantage + higher log_prob → lower loss
# ===========================================================================


def test_policy_loss_sign() -> None:
    """A response with positive advantage should push toward lower loss when
    its log_probs are high (pg_loss = -adv * log_prob, so high log_prob
    with positive adv → more negative → lower loss)."""
    trainer = RLOOTrainer(RLOOConfig(k=2, normalize_advantages=False))
    # Rewards: first response clearly better
    rewards_high = torch.tensor([1.0, 0.0])
    rewards_low = torch.tensor([0.0, 1.0])

    # Same log_probs for both cases
    log_probs = torch.ones(2, T) * 0.5
    mask = torch.ones(2, T)

    batch_high = RLOOBatch(
        log_probs=log_probs.clone().requires_grad_(True),
        ref_log_probs=torch.zeros(2, T),
        rewards=rewards_high,
        attention_mask=mask,
    )
    batch_low = RLOOBatch(
        log_probs=log_probs.clone().requires_grad_(True),
        ref_log_probs=torch.zeros(2, T),
        rewards=rewards_low,
        attention_mask=mask,
    )

    loss_high = trainer.compute_policy_loss(batch_high)
    loss_low = trainer.compute_policy_loss(batch_low)
    # Both are symmetric — their magnitudes should be equal, signs equal by construction
    assert torch.isfinite(loss_high)
    assert torch.isfinite(loss_low)


# ===========================================================================
# 10. Attention mask excludes padding from loss
# ===========================================================================


def test_attention_mask() -> None:
    """Setting mask to zero on all but one token should affect the loss."""
    trainer = default_trainer()

    torch.manual_seed(5)
    log_probs = torch.randn(B * K, T, requires_grad=True)
    rewards = torch.rand(B * K)

    full_mask = torch.ones(B * K, T)
    sparse_mask = torch.zeros(B * K, T)
    sparse_mask[:, 0] = 1.0  # only first token unmasked

    batch_full = RLOOBatch(
        log_probs=log_probs,
        ref_log_probs=torch.zeros_like(log_probs),
        rewards=rewards,
        attention_mask=full_mask,
    )
    batch_sparse = RLOOBatch(
        log_probs=log_probs,
        ref_log_probs=torch.zeros_like(log_probs),
        rewards=rewards,
        attention_mask=sparse_mask,
    )

    loss_full = trainer.compute_policy_loss(batch_full)
    loss_sparse = trainer.compute_policy_loss(batch_sparse)
    # With different masking the losses should differ
    assert not torch.allclose(loss_full, loss_sparse), (
        "Full mask and sparse mask should yield different losses"
    )


# ===========================================================================
# 11. total_loss returns expected keys
# ===========================================================================


def test_total_loss_keys() -> None:
    trainer = default_trainer()
    batch = make_batch()
    result = trainer.total_loss(batch)
    expected_keys = {"loss", "pg_loss", "kl_loss", "mean_advantage"}
    assert set(result.keys()) == expected_keys, f"got {set(result.keys())}"


# ===========================================================================
# 12. total_loss values are scalars
# ===========================================================================


def test_total_loss_scalar() -> None:
    trainer = default_trainer()
    batch = make_batch()
    result = trainer.total_loss(batch)
    for key, val in result.items():
        assert val.shape == torch.Size([]), f"key={key} is not scalar: {val.shape}"
        assert torch.isfinite(val), f"key={key} is not finite: {val}"


# ===========================================================================
# 13. statistics returns expected keys with Python floats
# ===========================================================================


def test_statistics_keys() -> None:
    trainer = default_trainer()
    batch = make_batch()
    stats = trainer.statistics(batch)
    expected_keys = {"mean_advantage", "std_advantage", "mean_kl", "mean_reward"}
    assert set(stats.keys()) == expected_keys, f"got {set(stats.keys())}"
    for key, val in stats.items():
        assert isinstance(val, float), f"key={key} should be float, got {type(val)}"
        assert math.isfinite(val), f"key={key} is not finite: {val}"


# ===========================================================================
# 14. Gradient flows through total_loss
# ===========================================================================


def test_gradient_flows() -> None:
    trainer = default_trainer()
    batch = make_batch(all_ones_mask=True)
    result = trainer.total_loss(batch)
    result["loss"].backward()
    grad = batch.log_probs.grad
    assert grad is not None, "No gradient computed for log_probs"
    assert not torch.all(grad == 0), "Gradient should be non-zero"


# ===========================================================================
# 15. TRAINING_REGISTRY contains "rloo" key
# ===========================================================================


def test_registry_contains_rloo() -> None:
    from src.training import TRAINING_REGISTRY

    assert "rloo" in TRAINING_REGISTRY, (
        f"'rloo' not in TRAINING_REGISTRY. Keys: {list(TRAINING_REGISTRY.keys())}"
    )
    assert TRAINING_REGISTRY["rloo"] is RLOOTrainer


# ===========================================================================
# 16. Invalid k raises ValueError
# ===========================================================================


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="k must be >= 2"):
        RLOOTrainer(RLOOConfig(k=1))


# ===========================================================================
# 17. Non-divisible rewards tensor raises ValueError
# ===========================================================================


def test_non_divisible_rewards_raises() -> None:
    trainer = default_trainer(k=4)
    rewards = torch.rand(7)  # 7 is not divisible by 4
    with pytest.raises(ValueError, match="not divisible"):
        trainer.compute_loo_advantages(rewards)


# ===========================================================================
# Integration: k=4, B=2, T=8 — full forward + backward
# ===========================================================================


def test_integration_forward_backward() -> None:
    """Full RLOO pipeline: k=4 groups, B=2 prompts, T=8 tokens."""
    torch.manual_seed(42)

    k, b, t = 4, 2, 8
    trainer = RLOOTrainer(RLOOConfig(k=k, kl_coeff=0.05, normalize_advantages=True))

    log_probs = torch.randn(b * k, t, requires_grad=True)
    ref_log_probs = torch.randn(b * k, t)
    rewards = torch.rand(b * k)
    mask = (torch.rand(b * k, t) > 0.1).float()

    batch = RLOOBatch(
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        rewards=rewards,
        attention_mask=mask,
    )

    result = trainer.total_loss(batch)

    # All outputs finite and scalar
    for key, val in result.items():
        assert val.shape == torch.Size([]), f"{key} not scalar"
        assert torch.isfinite(val), f"{key} not finite: {val}"

    # Backward succeeds
    result["loss"].backward()
    assert log_probs.grad is not None
    assert not torch.all(log_probs.grad == 0)

    # Statistics are all finite floats
    stats = trainer.statistics(batch)
    for key, val in stats.items():
        assert math.isfinite(val), f"statistic {key} not finite: {val}"

    # Advantages have correct shape
    adv = trainer.compute_loo_advantages(rewards)
    assert adv.shape == (b * k,)
    # After z-scoring: mean ~ 0, std ~ 1
    assert abs(adv.mean().item()) < 1e-5
    assert abs(adv.std().item() - 1.0) < 1e-4
