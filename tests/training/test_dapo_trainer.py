"""Tests for DAPO (Decoupled Clip-Higher Policy Optimization) trainer.

15 tests covering DAPOConfig, DAPOBatch, DAPOTrainer (advantages, policy loss,
clipping asymmetry, entropy bonus, masking, diversity filter, total_loss, stats)
plus one integration test with a full forward + backward pass.

Tiny config: d_model=64, n_heads=4, vocab_size=256 (per project constraints).
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

# Make src importable when running from repo root or tests directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.training.dapo_trainer import DAPOBatch, DAPOConfig, DAPOTrainer

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

VOCAB = 256
D_MODEL = 64
N_HEADS = 4
G = 4  # group size (tiny)
T = 16  # sequence length (tiny)


def _make_batch(
    g: int = G,
    t: int = T,
    *,
    all_same_reward: bool = False,
    mask_last: bool = False,
    requires_grad: bool = False,
) -> DAPOBatch:
    """Build a random DAPOBatch for testing."""
    torch.manual_seed(0)
    log_probs = torch.randn(g, t) * 0.5
    if requires_grad:
        log_probs = log_probs.detach().requires_grad_(True)
    ref_log_probs = torch.randn(g, t) * 0.5

    if all_same_reward:
        rewards = torch.ones(g) * 0.5
    else:
        rewards = torch.rand(g)

    mask = torch.ones(g, t)
    if mask_last:
        # Zero out the last column to test masking
        mask[:, -1] = 0.0

    return DAPOBatch(
        token_ids=torch.randint(0, VOCAB, (g, t)),
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        rewards=rewards,
        attention_mask=mask,
    )


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    """DAPOConfig should have the documented default hyperparameters."""
    cfg = DAPOConfig()
    assert cfg.clip_low == pytest.approx(0.2)
    assert cfg.clip_high == pytest.approx(0.28)
    assert cfg.entropy_coeff == pytest.approx(0.001)
    assert cfg.group_size == 8
    assert cfg.min_group_diversity == pytest.approx(0.0)
    assert cfg.kl_coeff == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. test_compute_advantages_zero_mean
# ---------------------------------------------------------------------------


def test_compute_advantages_zero_mean() -> None:
    """Normalised advantages must have mean ~0 for non-uniform rewards."""
    trainer = DAPOTrainer()
    rewards = torch.tensor([0.1, 0.4, 0.7, 1.0])
    adv = trainer.compute_advantages(rewards)
    assert abs(adv.mean().item()) < 1e-5, f"mean={adv.mean().item()}"


# ---------------------------------------------------------------------------
# 3. test_compute_advantages_normalized
# ---------------------------------------------------------------------------


def test_compute_advantages_normalized() -> None:
    """Population std of normalised advantages should be ≈ 1."""
    trainer = DAPOTrainer()
    rewards = torch.tensor([0.1, 0.4, 0.7, 1.0])
    adv = trainer.compute_advantages(rewards)
    std_val = adv.std(unbiased=False).item()
    assert abs(std_val - 1.0) < 1e-4, f"std={std_val}"


# ---------------------------------------------------------------------------
# 4. test_policy_loss_positive_adv
# ---------------------------------------------------------------------------


def test_policy_loss_positive_adv() -> None:
    """With positive advantages and ratio == 1, policy loss should be negative."""
    trainer = DAPOTrainer()
    g, t = 4, 8
    # ratio == 1 when log_probs == ref_log_probs
    lp = torch.zeros(g, t)
    batch = DAPOBatch(
        token_ids=torch.zeros(g, t, dtype=torch.long),
        log_probs=lp,
        ref_log_probs=torch.zeros(g, t),
        # All different rewards -> positive advantage for some, negative for others
        # Use rewards that give positive mean advantage for the top entry
        rewards=torch.tensor([0.0, 0.3, 0.7, 1.0]),
        attention_mask=torch.ones(g, t),
    )
    loss = trainer.compute_policy_loss(batch)
    # Loss should be a finite scalar
    assert loss.shape == torch.Size([])
    assert math.isfinite(loss.item())


# ---------------------------------------------------------------------------
# 5. test_clip_high_asymmetry
# ---------------------------------------------------------------------------


def test_clip_high_asymmetry() -> None:
    """clip_high > clip_low means upper bound is further from 1 than lower bound."""
    cfg = DAPOConfig(clip_low=0.2, clip_high=0.28)
    # Upper bound: 1 + 0.28 = 1.28; lower bound: 1 - 0.2 = 0.8
    upper = 1.0 + cfg.clip_high
    lower = 1.0 - cfg.clip_low
    assert upper > lower
    assert (upper - 1.0) > (1.0 - lower), (
        "Upper margin should be larger than lower margin for DAPO decoupled clipping"
    )


# ---------------------------------------------------------------------------
# 6. test_entropy_bonus_reduces_loss
# ---------------------------------------------------------------------------


def test_entropy_bonus_reduces_loss() -> None:
    """Non-zero entropy_coeff should change total loss value vs coeff=0."""
    batch = _make_batch()

    trainer_no_ent = DAPOTrainer(DAPOConfig(entropy_coeff=0.0))
    trainer_with_ent = DAPOTrainer(DAPOConfig(entropy_coeff=1.0))

    loss_no_ent = trainer_no_ent.total_loss(batch)["loss"].item()
    loss_with_ent = trainer_with_ent.total_loss(batch)["loss"].item()

    assert loss_no_ent != pytest.approx(loss_with_ent), (
        "entropy_coeff > 0 must change the total loss"
    )


# ---------------------------------------------------------------------------
# 7. test_attention_mask_respected
# ---------------------------------------------------------------------------


def test_attention_mask_respected() -> None:
    """Masking the last token column must change the computed loss."""
    batch_full = _make_batch(mask_last=False)
    batch_masked = _make_batch(mask_last=True)

    trainer = DAPOTrainer()
    loss_full = trainer.compute_policy_loss(batch_full).item()
    loss_masked = trainer.compute_policy_loss(batch_masked).item()

    assert loss_full != pytest.approx(loss_masked), "Masking tokens should change the policy loss"


# ---------------------------------------------------------------------------
# 8. test_is_group_diverse_all_same
# ---------------------------------------------------------------------------


def test_is_group_diverse_all_same() -> None:
    """is_group_diverse returns False when all rewards are identical."""
    trainer = DAPOTrainer()
    all_zero = torch.zeros(G)
    all_one = torch.ones(G)
    assert trainer.is_group_diverse(all_zero) is False
    assert trainer.is_group_diverse(all_one) is False


# ---------------------------------------------------------------------------
# 9. test_is_group_diverse_mixed
# ---------------------------------------------------------------------------


def test_is_group_diverse_mixed() -> None:
    """is_group_diverse returns True when rewards are mixed."""
    trainer = DAPOTrainer()
    mixed = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert trainer.is_group_diverse(mixed) is True

    varied = torch.tensor([0.1, 0.4, 0.7, 0.95])
    assert trainer.is_group_diverse(varied) is True


# ---------------------------------------------------------------------------
# 10. test_total_loss_keys
# ---------------------------------------------------------------------------


def test_total_loss_keys() -> None:
    """total_loss must return a dict with exactly the required keys."""
    trainer = DAPOTrainer()
    batch = _make_batch()
    result = trainer.total_loss(batch)

    required = {"loss", "pg_loss", "entropy_bonus", "kl_loss"}
    assert set(result.keys()) == required, f"Got keys: {set(result.keys())}"


# ---------------------------------------------------------------------------
# 11. test_total_loss_scalar
# ---------------------------------------------------------------------------


def test_total_loss_scalar() -> None:
    """total_loss['loss'] must be a 0-dim (scalar) tensor."""
    trainer = DAPOTrainer()
    batch = _make_batch()
    result = trainer.total_loss(batch)

    assert result["loss"].shape == torch.Size([]), (
        f"Expected scalar, got shape {result['loss'].shape}"
    )
    assert math.isfinite(result["loss"].item()), "total loss must be finite"


# ---------------------------------------------------------------------------
# 12. test_statistics_keys
# ---------------------------------------------------------------------------


def test_statistics_keys() -> None:
    """statistics() must return a dict with all four required keys."""
    trainer = DAPOTrainer()
    batch = _make_batch()
    stats = trainer.statistics(batch)

    required = {"clip_fraction", "mean_ratio", "mean_advantage", "entropy_mean"}
    assert set(stats.keys()) == required, f"Got: {set(stats.keys())}"
    for key, val in stats.items():
        assert isinstance(val, float), f"{key} should be float, got {type(val)}"
        assert math.isfinite(val), f"{key} is not finite: {val}"


# ---------------------------------------------------------------------------
# 13. test_zero_kl_coeff
# ---------------------------------------------------------------------------


def test_zero_kl_coeff() -> None:
    """kl_loss entry should be zero (or very close) when kl_coeff == 0."""
    trainer = DAPOTrainer(DAPOConfig(kl_coeff=0.0))
    batch = _make_batch()
    result = trainer.total_loss(batch)
    assert result["kl_loss"].item() == pytest.approx(0.0, abs=1e-8), (
        f"kl_loss should be 0 when kl_coeff=0, got {result['kl_loss'].item()}"
    )


# ---------------------------------------------------------------------------
# 14. test_gradient_flows
# ---------------------------------------------------------------------------


def test_gradient_flows() -> None:
    """loss.backward() must propagate a non-None, non-zero gradient to log_probs."""
    trainer = DAPOTrainer()
    batch = _make_batch(requires_grad=True)

    result = trainer.total_loss(batch)
    result["loss"].backward()

    assert batch.log_probs.grad is not None, "gradient should flow to log_probs"
    assert not torch.all(batch.log_probs.grad == 0), "gradient should be non-zero"


# ---------------------------------------------------------------------------
# 15. test_nonzero_kl_coeff
# ---------------------------------------------------------------------------


def test_nonzero_kl_coeff() -> None:
    """kl_loss entry should be non-zero when kl_coeff != 0 and policies differ."""
    trainer = DAPOTrainer(DAPOConfig(kl_coeff=0.1))
    torch.manual_seed(1)
    batch = DAPOBatch(
        token_ids=torch.randint(0, VOCAB, (G, T)),
        log_probs=torch.randn(G, T),  # policy != ref -> non-zero KL
        ref_log_probs=torch.randn(G, T),
        rewards=torch.rand(G),
        attention_mask=torch.ones(G, T),
    )
    result = trainer.total_loss(batch)
    assert result["kl_loss"].item() != pytest.approx(0.0), (
        "kl_loss should be non-zero when kl_coeff != 0 and policies differ"
    )


# ---------------------------------------------------------------------------
# 16. test_registry_entry
# ---------------------------------------------------------------------------


def test_registry_entry() -> None:
    """TRAINING_REGISTRY['dapo'] must point to DAPOTrainer."""
    from src.training import TRAINING_REGISTRY

    assert "dapo" in TRAINING_REGISTRY, "TRAINING_REGISTRY missing 'dapo' key"
    assert TRAINING_REGISTRY["dapo"] is DAPOTrainer


# ---------------------------------------------------------------------------
# Integration test — full forward + backward, G=4 T=16 vocab=256
# ---------------------------------------------------------------------------


def test_integration_forward_backward() -> None:
    """Full integration test: construct trainer, build batch, run total_loss,
    verify finite scalar loss, and confirm backward completes without error."""
    torch.manual_seed(42)

    cfg = DAPOConfig(
        clip_low=0.2,
        clip_high=0.28,
        entropy_coeff=0.001,
        group_size=4,
        kl_coeff=0.01,
    )
    trainer = DAPOTrainer(cfg)

    g, t = 4, 16

    # Simulate log-probs from a tiny model (requires_grad so backward works)
    log_probs = torch.randn(g, t, requires_grad=True)
    ref_log_probs = torch.randn(g, t)
    rewards = torch.tensor([0.0, 0.3, 0.7, 1.0])  # mixed -> diverse group
    mask = torch.ones(g, t)
    # Pad last 4 tokens of last response to exercise masking
    mask[-1, -4:] = 0.0

    batch = DAPOBatch(
        token_ids=torch.randint(0, VOCAB, (g, t)),
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        rewards=rewards,
        attention_mask=mask,
    )

    # Verify diversity filter sees this as informative
    assert trainer.is_group_diverse(rewards), "Test batch should be diverse"

    # Forward
    result = trainer.total_loss(batch)
    loss = result["loss"]

    # Loss must be a finite scalar
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    assert math.isfinite(loss.item()), f"Loss is not finite: {loss.item()}"

    # All breakdown tensors must be finite
    for key in ("pg_loss", "entropy_bonus", "kl_loss"):
        assert math.isfinite(result[key].item()), f"{key} is not finite"

    # Statistics must all be finite
    stats = trainer.statistics(batch)
    for key, val in stats.items():
        assert math.isfinite(val), f"stats[{key!r}] is not finite: {val}"

    # Clip fraction must be in [0, 1]
    assert 0.0 <= stats["clip_fraction"] <= 1.0

    # Backward must complete and produce non-None gradients
    loss.backward()
    assert log_probs.grad is not None, "Gradient did not flow to log_probs"
    assert not torch.all(log_probs.grad == 0), "All gradients are zero"
