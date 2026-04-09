"""Tests for src/training/model_merge_advanced.py."""
from __future__ import annotations

import pytest
import torch

from src.training.model_merge_advanced import (
    MergeConfig,
    apply_task_vector,
    compute_task_vector,
    dare_prune,
    merge_models,
    ties_disjoint_merge,
    ties_elect_sign,
    ties_trim,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHAPE = (50, 50)


def _rand_state(keys=("w1", "w2"), shape=SHAPE, seed=0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {k: torch.randn(*shape) for k in keys}


# ---------------------------------------------------------------------------
# MergeConfig defaults
# ---------------------------------------------------------------------------


def test_merge_config_defaults():
    cfg = MergeConfig()
    assert cfg.method == "dare_ties"
    assert cfg.density == 0.5
    assert cfg.lambda_ == 1.0
    assert cfg.top_k_fraction == 0.2
    assert cfg.seed == 42


def test_merge_config_custom():
    cfg = MergeConfig(method="dare", density=0.3, lambda_=2.0, top_k_fraction=0.1, seed=7)
    assert cfg.method == "dare"
    assert cfg.density == 0.3
    assert cfg.lambda_ == 2.0
    assert cfg.top_k_fraction == 0.1
    assert cfg.seed == 7


# ---------------------------------------------------------------------------
# compute_task_vector
# ---------------------------------------------------------------------------


def test_compute_task_vector_correctness():
    base = {"w": torch.ones(4, 4)}
    ft = {"w": torch.ones(4, 4) * 3.0}
    tv = compute_task_vector(ft, base)
    assert "w" in tv
    assert torch.allclose(tv["w"], torch.full((4, 4), 2.0))


def test_compute_task_vector_missing_key_skipped():
    base = {"w1": torch.ones(3), "w2": torch.ones(3)}
    ft = {"w1": torch.ones(3) * 2.0}  # w2 missing
    tv = compute_task_vector(ft, base)
    assert "w1" in tv
    assert "w2" not in tv


def test_compute_task_vector_shape_mismatch_skipped():
    base = {"w": torch.ones(4)}
    ft = {"w": torch.ones(8)}  # different shape
    tv = compute_task_vector(ft, base)
    assert "w" not in tv


# ---------------------------------------------------------------------------
# dare_prune
# ---------------------------------------------------------------------------


def test_dare_prune_sparsity():
    """Fraction of zeros should be approximately 1 - density."""
    torch.manual_seed(0)
    tv = {"w": torch.randn(1000)}
    density = 0.3
    pruned = dare_prune(tv, density=density, seed=42)
    zero_frac = (pruned["w"] == 0).float().mean().item()
    # Allow generous tolerance for stochastic test
    assert abs(zero_frac - (1.0 - density)) < 0.05, f"zero_frac={zero_frac:.3f}"


def test_dare_prune_reproducible_same_seed():
    torch.manual_seed(0)
    tv = {"w": torch.randn(500)}
    pruned1 = dare_prune(tv, density=0.5, seed=99)
    pruned2 = dare_prune(tv, density=0.5, seed=99)
    assert torch.allclose(pruned1["w"], pruned2["w"])


def test_dare_prune_different_seeds_differ():
    torch.manual_seed(0)
    tv = {"w": torch.randn(500)}
    pruned1 = dare_prune(tv, density=0.5, seed=1)
    pruned2 = dare_prune(tv, density=0.5, seed=2)
    assert not torch.allclose(pruned1["w"], pruned2["w"])


def test_dare_prune_rescaling_preserves_expected_magnitude():
    """E[|pruned|] should be close to E[|original|] (unbiased rescale)."""
    torch.manual_seed(0)
    # Use a large tensor to get stable statistics
    tv = {"w": torch.randn(10_000)}
    density = 0.4
    pruned = dare_prune(tv, density=density, seed=42)

    orig_mean_abs = tv["w"].abs().mean().item()
    # Only average over non-zero elements (they are rescaled by 1/density)
    nonzero = pruned["w"][pruned["w"] != 0]
    if len(nonzero) > 0:
        pruned_mean_abs = nonzero.abs().mean().item()
        # The kept values are original * (1/density), so mean(|kept|) = mean(|orig|)/density
        # But the overall mean E[|pruned|] = density * mean(|kept|) = mean(|orig|)
        overall_mean_abs = pruned["w"].abs().mean().item()
        assert abs(overall_mean_abs - orig_mean_abs) < 0.05 * orig_mean_abs, (
            f"overall_mean_abs={overall_mean_abs:.4f}, orig={orig_mean_abs:.4f}"
        )


# ---------------------------------------------------------------------------
# ties_trim
# ---------------------------------------------------------------------------


def test_ties_trim_sparsity():
    """After trim, fraction of zeros should be ~ 1 - top_k_fraction."""
    torch.manual_seed(0)
    tv = {"w": torch.randn(1000)}
    top_k = 0.2
    trimmed = ties_trim(tv, top_k_fraction=top_k)
    zero_frac = (trimmed["w"] == 0).float().mean().item()
    # Due to quantile boundary ties, allow ±5%
    assert abs(zero_frac - (1.0 - top_k)) < 0.05, f"zero_frac={zero_frac:.3f}"


def test_ties_trim_keeps_largest():
    """Trimmed values should all have abs >= threshold of the original."""
    torch.manual_seed(1)
    delta = torch.randn(200)
    tv = {"w": delta}
    top_k = 0.3
    trimmed = ties_trim(tv, top_k_fraction=top_k)
    kept = trimmed["w"][trimmed["w"] != 0]
    # Compute threshold from original
    threshold = torch.quantile(delta.abs(), 1.0 - top_k)
    assert (kept.abs() >= threshold - 1e-6).all(), "Some kept values are below threshold"


# ---------------------------------------------------------------------------
# ties_elect_sign
# ---------------------------------------------------------------------------


def test_ties_elect_sign_majority_vote():
    """Majority vote: sign_sum = sum of signs (not values) across task vectors."""
    # 3 tensors; sign votes at each position:
    # pos 0: sign(1.0)=+1, sign(2.0)=+1, sign(0.5)=+1 → sum=+3 → elected=+1
    # pos 1: sign(-1.0)=-1, sign(-2.0)=-1, sign(3.0)=+1 → sum=-1 → elected=-1
    # pos 2: all positive → sum=+3 → elected=+1
    tv1 = {"w": torch.tensor([1.0, -1.0, 0.5])}
    tv2 = {"w": torch.tensor([2.0, -2.0, 0.5])}
    tv3 = {"w": torch.tensor([0.5, 3.0, 0.5])}
    elected = ties_elect_sign([tv1, tv2, tv3])
    assert elected["w"][0].item() == 1.0
    assert elected["w"][1].item() == -1.0
    assert elected["w"][2].item() == 1.0


def test_ties_elect_sign_tie_broken_positive():
    """When sign votes are tied (sum=0), elected sign should be +1."""
    # 2 tensors: one positive, one negative → sign_sum = 0 → tie → +1
    tv1 = {"w": torch.tensor([1.0])}
    tv2 = {"w": torch.tensor([-1.0])}
    elected = ties_elect_sign([tv1, tv2])
    assert elected["w"][0].item() == 1.0


def test_ties_elect_sign_negative_majority():
    # pos 0: sign(-1)=-1, sign(-2)=-1, sign(-0.5)=-1 → sum=-3 → elected=-1
    # pos 1: sign(1)=+1, sign(1)=+1, sign(-3)=-1 → sum=+1 → elected=+1
    tv1 = {"w": torch.tensor([-1.0, 1.0])}
    tv2 = {"w": torch.tensor([-2.0, 1.0])}
    tv3 = {"w": torch.tensor([-0.5, -3.0])}
    elected = ties_elect_sign([tv1, tv2, tv3])
    assert elected["w"][0].item() == -1.0
    assert elected["w"][1].item() == 1.0


def test_ties_elect_sign_empty():
    assert ties_elect_sign([]) == {}


# ---------------------------------------------------------------------------
# ties_disjoint_merge
# ---------------------------------------------------------------------------


def test_ties_disjoint_merge_zeros_disagreeing():
    """Values that disagree with the elected sign should contribute 0."""
    # elected sign is +1 everywhere
    elected = {"w": torch.tensor([1.0, 1.0])}
    # tv1: both agree; tv2: position 0 disagrees (negative)
    tv1 = {"w": torch.tensor([2.0, 3.0])}
    tv2 = {"w": torch.tensor([-1.0, 4.0])}
    merged = ties_disjoint_merge([tv1, tv2], elected)

    # position 0: only tv1 agrees → 2.0
    assert abs(merged["w"][0].item() - 2.0) < 1e-5
    # position 1: both agree → (3+4)/2 = 3.5
    assert abs(merged["w"][1].item() - 3.5) < 1e-5


def test_ties_disjoint_merge_all_agree():
    elected = {"w": torch.ones(4)}
    tv1 = {"w": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    tv2 = {"w": torch.tensor([3.0, 4.0, 5.0, 6.0])}
    merged = ties_disjoint_merge([tv1, tv2], elected)
    expected = torch.tensor([2.0, 3.0, 4.0, 5.0])
    assert torch.allclose(merged["w"], expected)


# ---------------------------------------------------------------------------
# apply_task_vector
# ---------------------------------------------------------------------------


def test_apply_task_vector_correctness():
    base = {"w": torch.zeros(3)}
    tv = {"w": torch.tensor([1.0, 2.0, 3.0])}
    merged = apply_task_vector(base, tv, lambda_=2.0)
    expected = torch.tensor([2.0, 4.0, 6.0])
    assert torch.allclose(merged["w"], expected)


def test_apply_task_vector_missing_key_copies_base():
    base = {"w1": torch.ones(3), "w2": torch.ones(3) * 5.0}
    tv = {"w1": torch.ones(3)}  # w2 missing from task vector
    merged = apply_task_vector(base, tv, lambda_=1.0)
    assert "w2" in merged
    assert torch.allclose(merged["w2"], base["w2"])


def test_apply_task_vector_lambda_zero():
    """lambda_=0 should return a copy of base unchanged."""
    base = {"w": torch.randn(10)}
    tv = {"w": torch.randn(10)}
    merged = apply_task_vector(base, tv, lambda_=0.0)
    assert torch.allclose(merged["w"], base["w"])


# ---------------------------------------------------------------------------
# merge_models — linear
# ---------------------------------------------------------------------------


def test_merge_models_linear():
    base = _rand_state(seed=0)
    ft1 = _rand_state(seed=1)
    ft2 = _rand_state(seed=2)
    cfg = MergeConfig(method="linear")
    merged = merge_models(base, [ft1, ft2], cfg)

    # Should be average of ft1 and ft2 (not base)
    for key in base:
        expected = (ft1[key].float() + ft2[key].float()) / 2.0
        assert torch.allclose(merged[key].float(), expected, atol=1e-5), key


# ---------------------------------------------------------------------------
# merge_models — dare
# ---------------------------------------------------------------------------


def test_merge_models_dare_returns_all_keys():
    base = _rand_state(seed=0)
    ft1 = _rand_state(seed=1)
    cfg = MergeConfig(method="dare", density=0.5)
    merged = merge_models(base, [ft1], cfg)
    assert set(merged.keys()) == set(base.keys())


def test_merge_models_dare_differs_from_base():
    base = _rand_state(seed=0)
    ft1 = _rand_state(seed=1)
    cfg = MergeConfig(method="dare", density=0.5)
    merged = merge_models(base, [ft1], cfg)
    # At least some parameters should differ from base
    diffs = [(merged[k] - base[k]).abs().max().item() for k in base]
    assert max(diffs) > 0.0


# ---------------------------------------------------------------------------
# merge_models — ties
# ---------------------------------------------------------------------------


def test_merge_models_ties_returns_all_keys():
    base = _rand_state(seed=0)
    ft1 = _rand_state(seed=1)
    ft2 = _rand_state(seed=2)
    cfg = MergeConfig(method="ties", top_k_fraction=0.3)
    merged = merge_models(base, [ft1, ft2], cfg)
    assert set(merged.keys()) == set(base.keys())


def test_merge_models_ties_lambda_zero_equals_base():
    """With lambda_=0, the merged model should equal base regardless of method."""
    base = _rand_state(seed=0)
    ft1 = _rand_state(seed=1)
    ft2 = _rand_state(seed=2)
    cfg = MergeConfig(method="ties", lambda_=0.0)
    merged = merge_models(base, [ft1, ft2], cfg)
    for key in base:
        assert torch.allclose(merged[key].float(), base[key].float(), atol=1e-5), key


# ---------------------------------------------------------------------------
# merge_models — dare_ties
# ---------------------------------------------------------------------------


def test_merge_models_dare_ties_returns_all_keys():
    base = _rand_state(seed=0)
    ft1 = _rand_state(seed=1)
    ft2 = _rand_state(seed=2)
    cfg = MergeConfig(method="dare_ties")
    merged = merge_models(base, [ft1, ft2], cfg)
    assert set(merged.keys()) == set(base.keys())


def test_merge_models_unknown_method_raises():
    base = _rand_state(seed=0)
    ft1 = _rand_state(seed=1)
    cfg = MergeConfig(method="unknown_method")
    with pytest.raises(ValueError, match="unknown_method"):
        merge_models(base, [ft1], cfg)


def test_merge_models_empty_finetuned_returns_base():
    base = _rand_state(seed=0)
    cfg = MergeConfig(method="dare_ties")
    merged = merge_models(base, [], cfg)
    for key in base:
        assert torch.allclose(merged[key], base[key])
