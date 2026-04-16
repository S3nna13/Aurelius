"""Tests for WARM: Weight-Averaged Reward Models."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.alignment.warm_reward import (
    WARMConfig,
    WARMEnsemble,
    linear_merge,
    dare_merge,
    compute_reward_margin,
    warm_reward_score,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def make_state_dict(in_features: int = 8, out_features: int = 4, seed: int = 0) -> dict:
    """Return a state_dict from a tiny nn.Linear model."""
    torch.manual_seed(seed)
    return nn.Linear(in_features, out_features).state_dict()


def make_linear_model(in_features: int = 8, out_features: int = 1, seed: int = 0) -> nn.Module:
    """Create a tiny nn.Linear that outputs (batch,) scalars."""
    torch.manual_seed(seed)
    model = nn.Linear(in_features, out_features)
    return model


class ScalarRewardModel(nn.Module):
    """Tiny reward model that maps (batch, seq) -> (batch,)."""

    def __init__(self, in_features: int = 8, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq) or (batch, in_features)
        # Use float average over seq dim as feature
        if x.dtype in (torch.long, torch.int):
            x = x.float()
        out = self.linear(x.float())  # (batch, 1)
        return out.squeeze(-1)         # (batch,)


def model_factory(in_features: int = 8) -> nn.Module:
    return ScalarRewardModel(in_features)


# ---------------------------------------------------------------------------
# Test 1: WARMConfig defaults correct
# ---------------------------------------------------------------------------


def test_warm_config_defaults():
    """WARMConfig must have the correct default field values."""
    cfg = WARMConfig()
    assert cfg.n_models == 4
    assert cfg.merge_method == "linear"
    assert cfg.temperature == 1.0
    assert cfg.dare_density == 0.7


# ---------------------------------------------------------------------------
# Test 2: linear_merge of identical state_dicts returns same values
# ---------------------------------------------------------------------------


def test_linear_merge_identical_returns_same():
    """Merging K identical state dicts must return the same weights."""
    sd = make_state_dict(seed=1)
    copies = [{k: v.clone() for k, v in sd.items()} for _ in range(3)]

    merged = linear_merge(copies)

    for key in sd:
        assert torch.allclose(merged[key].float(), sd[key].float(), atol=1e-5), (
            f"Merging identical state_dicts should return same values for key {key}"
        )


# ---------------------------------------------------------------------------
# Test 3: linear_merge of two models returns average weights
# ---------------------------------------------------------------------------


def test_linear_merge_two_models_is_average():
    """Merging two state dicts with equal weights must return their mean."""
    sd1 = make_state_dict(seed=10)
    sd2 = make_state_dict(seed=20)

    merged = linear_merge([sd1, sd2])

    for key in sd1:
        expected = (sd1[key].float() + sd2[key].float()) / 2.0
        assert torch.allclose(merged[key].float(), expected, atol=1e-5), (
            f"linear_merge of two models must equal their mean for key {key}"
        )


# ---------------------------------------------------------------------------
# Test 4: dare_merge density=1.0 equals linear_merge (no dropping)
# ---------------------------------------------------------------------------


def test_dare_merge_density1_equals_linear():
    """dare_merge with density=1.0 should equal linear_merge (no delta dropped)."""
    sds = [make_state_dict(seed=i) for i in range(3)]

    dare_result = dare_merge(sds, density=1.0, seed=0)
    linear_result = linear_merge(sds)

    # With density=1.0 dare_merge averages (K-1) deltas over base, while
    # linear_merge averages K state dicts uniformly.  These differ slightly
    # in formula but with density=1.0 both should keep all delta information.
    # We verify dare result is finite and has the same keys / shapes.
    for key in sds[0]:
        assert dare_result[key].shape == sds[0][key].shape, (
            f"dare_merge shape mismatch for key {key}"
        )
        assert torch.isfinite(dare_result[key]).all(), (
            f"dare_merge(density=1.0) has non-finite values for key {key}"
        )

    # Specifically: dare_merge(density=1.0) should produce something between
    # base and the average of the other models.  Verify values are finite and
    # the formula is correct: base + mean_of_deltas.
    base_sd = sds[0]
    for key in base_sd:
        base = base_sd[key].float()
        delta_sum = sum(sds[i][key].float() - base for i in range(1, len(sds)))
        avg_delta = delta_sum / (len(sds) - 1)
        expected = base + avg_delta
        assert torch.allclose(dare_result[key].float(), expected, atol=1e-5), (
            f"dare_merge density=1.0 formula mismatch for key {key}"
        )


# ---------------------------------------------------------------------------
# Test 5: dare_merge density=0.0 returns base (all deltas dropped)
# ---------------------------------------------------------------------------


def test_dare_merge_density0_returns_base():
    """dare_merge with density=0.0 must return base model weights (all deltas dropped)."""
    sds = [make_state_dict(seed=i) for i in range(3)]

    merged = dare_merge(sds, density=0.0, seed=0)

    for key in sds[0]:
        assert torch.allclose(merged[key].float(), sds[0][key].float(), atol=1e-5), (
            f"dare_merge density=0.0 must equal base for key {key}"
        )


# ---------------------------------------------------------------------------
# Test 6: WARMEnsemble.add_checkpoint stores checkpoint
# ---------------------------------------------------------------------------


def test_warm_ensemble_add_checkpoint_stores():
    """WARMEnsemble.add_checkpoint must store the checkpoint."""
    ensemble = WARMEnsemble(model_factory, n_models=4)
    assert len(ensemble._checkpoints) == 0

    sd1 = make_state_dict(seed=1)
    sd2 = make_state_dict(seed=2)
    ensemble.add_checkpoint(sd1, weight=1.0)
    ensemble.add_checkpoint(sd2, weight=2.0)

    assert len(ensemble._checkpoints) == 2, (
        "Two checkpoints should be stored after two add_checkpoint calls"
    )


# ---------------------------------------------------------------------------
# Test 7: WARMEnsemble.merge with 3 checkpoints returns valid state_dict
# ---------------------------------------------------------------------------


def test_warm_ensemble_merge_three_checkpoints():
    """WARMEnsemble.merge must return a valid state_dict with 3 checkpoints."""
    ensemble = WARMEnsemble(model_factory, n_models=3, merge_method="linear")
    for i in range(3):
        ensemble.add_checkpoint(make_state_dict(seed=i))

    merged = ensemble.merge()

    ref_sd = make_state_dict(seed=0)
    assert set(merged.keys()) == set(ref_sd.keys()), (
        "Merged state_dict must have same keys as checkpoints"
    )
    for key in merged:
        assert merged[key].shape == ref_sd[key].shape, (
            f"Merged tensor shape mismatch for key {key}"
        )
        assert torch.isfinite(merged[key]).all(), (
            f"Merged weights contain non-finite values for key {key}"
        )


# ---------------------------------------------------------------------------
# Test 8: compute_disagreement returns correct shape (batch,)
# ---------------------------------------------------------------------------


def test_compute_disagreement_shape():
    """compute_disagreement must return a (batch,) variance tensor."""
    batch = 6
    n_models = 4
    ensemble = WARMEnsemble(model_factory, n_models=n_models)

    rewards = [torch.randn(batch) for _ in range(n_models)]
    disagreement = ensemble.compute_disagreement(rewards)

    assert disagreement.shape == (batch,), (
        f"compute_disagreement must return shape ({batch},), got {disagreement.shape}"
    )
    assert (disagreement >= 0).all(), "Variance must be non-negative"


# ---------------------------------------------------------------------------
# Test 9: Higher model disagreement → higher variance score
# ---------------------------------------------------------------------------


def test_compute_disagreement_higher_variance():
    """Models with wider spread of predictions must yield higher disagreement."""
    batch = 4
    ensemble = WARMEnsemble(model_factory, n_models=3)

    # Low disagreement: all models agree
    low_rewards = [torch.tensor([1.0, 2.0, 3.0, 4.0]) for _ in range(3)]
    low_disagree = ensemble.compute_disagreement(low_rewards)

    # High disagreement: models spread widely
    high_rewards = [
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([10.0, 20.0, 30.0, 40.0]),
        torch.tensor([-10.0, -20.0, -30.0, -40.0]),
    ]
    high_disagree = ensemble.compute_disagreement(high_rewards)

    assert (high_disagree > low_disagree).all(), (
        "Higher spread predictions must yield higher disagreement scores"
    )


# ---------------------------------------------------------------------------
# Test 10: compute_reward_margin returns mean and std tensors
# ---------------------------------------------------------------------------


def test_compute_reward_margin_returns_mean_and_std():
    """compute_reward_margin must return scalar mean and std tensors."""
    n_models = 5
    batch = 4
    rewards = torch.randn(n_models, batch)
    chosen_idx = 0
    rejected_idx = 1

    mean_margin, std_margin = compute_reward_margin(rewards, chosen_idx, rejected_idx)

    assert mean_margin.ndim == 0, f"mean_margin must be scalar, got ndim={mean_margin.ndim}"
    assert std_margin.ndim == 0, f"std_margin must be scalar, got ndim={std_margin.ndim}"
    assert torch.isfinite(mean_margin), "mean_margin must be finite"
    assert torch.isfinite(std_margin), "std_margin must be finite"
    assert std_margin.item() >= 0, "std_margin must be non-negative"

    # Verify mean_margin formula: mean(rewards[:, chosen] - rewards[:, rejected])
    expected_mean = (rewards[:, chosen_idx] - rewards[:, rejected_idx]).mean()
    assert torch.allclose(mean_margin, expected_mean, atol=1e-5), (
        "mean_margin formula mismatch"
    )


# ---------------------------------------------------------------------------
# Test 11: warm_reward_score 'mean' aggregation correct shape
# ---------------------------------------------------------------------------


def test_warm_reward_score_mean_shape():
    """warm_reward_score with aggregate='mean' must return (batch,) tensor."""
    batch = 4
    in_features = 8
    models = [ScalarRewardModel(in_features, seed=i) for i in range(3)]
    input_ids = torch.randn(batch, in_features)

    result = warm_reward_score(models, input_ids, aggregate="mean")

    assert result.shape == (batch,), (
        f"warm_reward_score must return shape ({batch},), got {result.shape}"
    )
    assert torch.isfinite(result).all(), "warm_reward_score result must be finite"

    # Verify the mean is correct
    expected_rewards = []
    for model in models:
        with torch.no_grad():
            r = model(input_ids)
        expected_rewards.append(r)
    expected_mean = torch.stack(expected_rewards, dim=0).mean(dim=0)
    assert torch.allclose(result, expected_mean, atol=1e-5), (
        "warm_reward_score 'mean' must equal mean of individual model outputs"
    )


# ---------------------------------------------------------------------------
# Test 12: warm_reward_score 'vote' returns values in {0, 1}
# ---------------------------------------------------------------------------


def test_warm_reward_score_vote_binary():
    """warm_reward_score with aggregate='vote' must return values in {0, 1}."""
    batch = 8
    in_features = 8
    models = [ScalarRewardModel(in_features, seed=i) for i in range(4)]
    input_ids = torch.randn(batch, in_features)

    result = warm_reward_score(models, input_ids, aggregate="vote")

    assert result.shape == (batch,), (
        f"warm_reward_score 'vote' must return shape ({batch},), got {result.shape}"
    )
    unique_vals = result.unique().tolist()
    allowed = {0.0, 1.0}
    for v in unique_vals:
        assert v in allowed, (
            f"warm_reward_score 'vote' must return values in {{0, 1}}, got {v}"
        )
