"""Tests for WARP: Weight Averaging Rewarded Policies."""
import copy
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.alignment.warp import (
    slerp_two,
    merge_policies_slerp,
    anchor_merge,
    WARPTrainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tiny_model(in_features: int = 16, out_features: int = 8, seed: int = 0) -> nn.Linear:
    """Create a small nn.Linear for testing."""
    torch.manual_seed(seed)
    model = nn.Linear(in_features, out_features)
    return model


def make_state_dict(in_features: int = 16, out_features: int = 8, seed: int = 0):
    """Return a state_dict from a tiny linear model."""
    return make_tiny_model(in_features, out_features, seed).state_dict()


def simple_reward_fn(logits: Tensor) -> Tensor:
    """Toy reward: mean of logits (for testing only)."""
    # logits shape: (B, out) or (B, T, V)
    return logits.mean(dim=-1).mean(dim=-1) if logits.ndim == 3 else logits.mean(dim=-1)


# ---------------------------------------------------------------------------
# 1. slerp_two at t=0.0 returns v0
# ---------------------------------------------------------------------------

def test_slerp_two_t0_returns_v0():
    torch.manual_seed(42)
    v0 = torch.randn(32)
    v1 = torch.randn(32)
    result = slerp_two(0.0, v0, v1)
    assert torch.allclose(result, v0, atol=1e-5), "slerp_two(t=0) must return v0"


# ---------------------------------------------------------------------------
# 2. slerp_two at t=1.0 returns v1
# ---------------------------------------------------------------------------

def test_slerp_two_t1_returns_v1():
    torch.manual_seed(42)
    v0 = torch.randn(32)
    v1 = torch.randn(32)
    result = slerp_two(1.0, v0, v1)
    assert torch.allclose(result, v1, atol=1e-5), "slerp_two(t=1) must return v1"


# ---------------------------------------------------------------------------
# 3. slerp_two at t=0.5 returns a midpoint with approximately preserved norm
# ---------------------------------------------------------------------------

def test_slerp_two_t05_norm_preserved():
    torch.manual_seed(7)
    # Use unit vectors so the expected midpoint norm is well-defined.
    v0 = torch.randn(64)
    v0 = v0 / v0.norm()
    v1 = torch.randn(64)
    v1 = v1 / v1.norm()

    result = slerp_two(0.5, v0, v1)

    # The SLERP midpoint of two unit vectors should also be a unit vector
    # (up to the angle between them). The norm should be close to 1.
    result_norm = result.norm().item()
    assert 0.7 < result_norm <= 1.1, (
        f"SLERP midpoint norm {result_norm:.4f} should be close to 1 for unit inputs"
    )


# ---------------------------------------------------------------------------
# 4. slerp_two output has same shape as inputs
# ---------------------------------------------------------------------------

def test_slerp_two_shape_preserved():
    v0 = torch.randn(3, 4, 5)
    v1 = torch.randn(3, 4, 5)
    result = slerp_two(0.5, v0, v1)
    assert result.shape == v0.shape, (
        f"slerp_two output shape {result.shape} != input shape {v0.shape}"
    )


# ---------------------------------------------------------------------------
# 5. merge_policies_slerp with 2 identical models returns same weights
# ---------------------------------------------------------------------------

def test_merge_policies_slerp_identical_models():
    sd = make_state_dict(seed=1)
    sd2 = {k: v.clone() for k, v in sd.items()}
    merged = merge_policies_slerp([sd, sd2])

    for key in sd:
        assert torch.allclose(merged[key].float(), sd[key].float(), atol=1e-5), (
            f"Merging identical models should return same weights for key {key}"
        )


# ---------------------------------------------------------------------------
# 6. merge_policies_slerp with 3 models doesn't crash, returns valid state_dict
# ---------------------------------------------------------------------------

def test_merge_policies_slerp_three_models():
    sds = [make_state_dict(seed=i) for i in range(3)]
    merged = merge_policies_slerp(sds)

    assert set(merged.keys()) == set(sds[0].keys()), "Merged state_dict keys must match"
    for key in merged:
        assert merged[key].shape == sds[0][key].shape, (
            f"Merged tensor shape mismatch for key {key}"
        )
        assert torch.isfinite(merged[key]).all(), f"Merged weights contain non-finite values for {key}"


# ---------------------------------------------------------------------------
# 7. merge_policies_slerp with custom weights works
# ---------------------------------------------------------------------------

def test_merge_policies_slerp_custom_weights():
    sds = [make_state_dict(seed=i) for i in range(2)]
    weights = [0.8, 0.2]
    merged = merge_policies_slerp(sds, weights=weights)

    assert set(merged.keys()) == set(sds[0].keys())
    for key in merged:
        assert torch.isfinite(merged[key]).all(), f"Custom-weighted merge has non-finite values for {key}"


# ---------------------------------------------------------------------------
# 8. anchor_merge at alpha=0.0 returns SFT weights
# ---------------------------------------------------------------------------

def test_anchor_merge_alpha0_returns_sft():
    sft_sd = make_state_dict(seed=10)
    merged_sd = make_state_dict(seed=20)
    result = anchor_merge(sft_sd, merged_sd, alpha=0.0)

    for key in sft_sd:
        assert torch.allclose(result[key].float(), sft_sd[key].float(), atol=1e-6), (
            f"alpha=0 must return SFT weights for key {key}"
        )


# ---------------------------------------------------------------------------
# 9. anchor_merge at alpha=1.0 returns merged weights
# ---------------------------------------------------------------------------

def test_anchor_merge_alpha1_returns_merged():
    sft_sd = make_state_dict(seed=10)
    merged_sd = make_state_dict(seed=20)
    result = anchor_merge(sft_sd, merged_sd, alpha=1.0)

    for key in merged_sd:
        assert torch.allclose(result[key].float(), merged_sd[key].float(), atol=1e-6), (
            f"alpha=1 must return merged weights for key {key}"
        )


# ---------------------------------------------------------------------------
# 10. anchor_merge at alpha=0.5 is between SFT and merged
# ---------------------------------------------------------------------------

def test_anchor_merge_alpha05_is_midpoint():
    sft_sd = make_state_dict(seed=10)
    merged_sd = make_state_dict(seed=20)
    result = anchor_merge(sft_sd, merged_sd, alpha=0.5)

    for key in sft_sd:
        expected = 0.5 * sft_sd[key].float() + 0.5 * merged_sd[key].float()
        assert torch.allclose(result[key].float(), expected, atol=1e-5), (
            f"alpha=0.5 must return midpoint for key {key}"
        )


# ---------------------------------------------------------------------------
# 11. WARPTrainer.get_kl_penalty returns non-negative tensor
# ---------------------------------------------------------------------------

def test_warp_trainer_kl_penalty_nonnegative():
    torch.manual_seed(0)
    model = make_tiny_model()

    def dummy_reward(logits):
        return logits.mean(dim=-1)

    trainer = WARPTrainer(model, dummy_reward, alpha=0.5, n_policies=2)

    # logits shape (B, V)
    logits = torch.randn(4, 8)
    ref_logits = torch.randn(4, 8)
    kl = trainer.get_kl_penalty(logits, ref_logits)

    assert kl.ndim == 0, "KL penalty must be a scalar"
    assert kl.item() >= -1e-6, f"KL penalty must be non-negative, got {kl.item()}"


# ---------------------------------------------------------------------------
# 12. WARPTrainer.train_policy returns a valid state_dict
# ---------------------------------------------------------------------------

def test_warp_trainer_train_policy_returns_state_dict():
    torch.manual_seed(1)
    model = make_tiny_model(in_features=4, out_features=4)

    def dummy_reward(logits):
        return logits.mean(dim=-1)

    trainer = WARPTrainer(model, dummy_reward, alpha=0.5, n_policies=2, lr=1e-3)

    # input_ids not used directly since model is nn.Linear (we pass float tensor)
    # Patch: use float inputs since nn.Linear doesn't do embedding
    input_ids = torch.randn(2, 4)  # (B, in_features)
    sd = trainer.train_policy(input_ids, n_steps=3)

    assert isinstance(sd, dict), "train_policy must return a dict"
    assert set(sd.keys()) == set(model.state_dict().keys()), "Keys must match model state_dict"
    for key, val in sd.items():
        assert isinstance(val, torch.Tensor), f"Value for {key} must be a Tensor"
        assert torch.isfinite(val).all(), f"State dict value for {key} contains non-finite"


# ---------------------------------------------------------------------------
# 13. WARPTrainer.run completes and returns an nn.Module
# ---------------------------------------------------------------------------

def test_warp_trainer_run_returns_module():
    torch.manual_seed(2)
    model = make_tiny_model(in_features=4, out_features=4)

    def dummy_reward(logits):
        return logits.mean(dim=-1)

    trainer = WARPTrainer(model, dummy_reward, alpha=0.5, n_policies=2, lr=1e-3)
    input_ids = torch.randn(2, 4)
    final_model = trainer.run(input_ids, n_steps=2)

    assert isinstance(final_model, nn.Module), "run() must return an nn.Module"


# ---------------------------------------------------------------------------
# 14. Final model weights differ from SFT model (training changed something)
# ---------------------------------------------------------------------------

def test_warp_trainer_run_changes_weights():
    torch.manual_seed(3)
    model = make_tiny_model(in_features=4, out_features=4)
    original_sd = {k: v.clone() for k, v in model.state_dict().items()}

    def dummy_reward(logits):
        # Push logits in a specific direction so training produces a gradient
        return logits.sum(dim=-1)

    trainer = WARPTrainer(
        model, dummy_reward, alpha=0.9, n_policies=2, lr=1e-2, kl_coef=0.01
    )
    input_ids = torch.randn(2, 4)
    final_model = trainer.run(input_ids, n_steps=5)

    final_sd = final_model.state_dict()

    # At least one parameter should differ from the original SFT weights
    any_changed = any(
        not torch.allclose(final_sd[k].float(), original_sd[k].float(), atol=1e-7)
        for k in original_sd
    )
    assert any_changed, "Final model weights must differ from original SFT weights after WARP"
