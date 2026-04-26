"""Tests for src/training/della.py — DELLA magnitude-based model merging.

All tests use two tiny nn.Linear(16, 16) models (in_features=16, out_features=16)
as base and fine-tuned models so the test suite runs instantly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.della import (
    DELLAConfig,
    DELLAMerger,
    della_merge,
    magnitude_prune_delta,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

IN = OUT = 16  # tiny model dimension


def _linear_state(seed: int) -> dict[str, torch.Tensor]:
    """Return state dict of nn.Linear(16, 16) initialised with a fixed seed."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    m = nn.Linear(IN, OUT)
    # Re-initialise weights/biases with our generator for reproducibility
    nn.init.normal_(m.weight, generator=gen)
    nn.init.normal_(m.bias, generator=gen)
    return {k: v.clone() for k, v in m.state_dict().items()}


# Shared deterministic states used across tests
BASE = _linear_state(0)
FT1 = _linear_state(1)
FT2 = _linear_state(2)
FT3 = _linear_state(3)


# ---------------------------------------------------------------------------
# Tests for magnitude_prune_delta
# ---------------------------------------------------------------------------


def test_magnitude_prune_preserves_top_density_fraction():
    """Top ``density`` fraction (by |magnitude|) must all survive (non-zero)."""
    torch.manual_seed(42)
    delta = torch.randn(100)
    density = 0.4

    pruned = magnitude_prune_delta(delta, density=density, rescale=False)

    # Non-zero positions must correspond to the actual top-density values
    flat_abs = delta.abs()
    threshold = torch.quantile(flat_abs, 1.0 - density)
    expected_survivors = (flat_abs >= threshold).sum().item()
    actual_survivors = (pruned != 0).sum().item()

    # Allow ≤1 element of slack for threshold ties
    assert abs(actual_survivors - expected_survivors) <= 1


def test_magnitude_prune_bottom_fraction_zeroed():
    """Elements strictly below the magnitude threshold must be zero."""
    torch.manual_seed(7)
    delta = torch.randn(200)
    density = 0.3

    pruned = magnitude_prune_delta(delta, density=density, rescale=False)

    flat_abs = delta.abs()
    threshold = torch.quantile(flat_abs, 1.0 - density)
    # All elements below threshold must be zeroed in the output
    below_mask = flat_abs < threshold
    assert (pruned[below_mask] == 0).all(), "Below-threshold elements should be zero"


def test_magnitude_prune_rescale_true_scales_survivors():
    """rescale=True: surviving values must equal original / density."""
    torch.manual_seed(13)
    delta = torch.randn(50)
    density = 0.5

    pruned_rescaled = magnitude_prune_delta(delta, density=density, rescale=True)
    pruned_raw = magnitude_prune_delta(delta, density=density, rescale=False)

    # Where the raw pruned value is non-zero, the rescaled version = raw / density
    nonzero_mask = pruned_raw != 0
    if nonzero_mask.any():
        expected = pruned_raw[nonzero_mask] / density
        actual = pruned_rescaled[nonzero_mask]
        assert torch.allclose(actual, expected, atol=1e-6), (
            "rescale=True should multiply survivors by 1/density"
        )


def test_magnitude_prune_rescale_false_leaves_magnitudes():
    """rescale=False: surviving values must equal the original delta values."""
    torch.manual_seed(99)
    delta = torch.randn(50)
    density = 0.6

    pruned = magnitude_prune_delta(delta, density=density, rescale=False)

    nonzero_mask = pruned != 0
    # Surviving positions should match the original delta exactly
    assert torch.allclose(pruned[nonzero_mask], delta[nonzero_mask], atol=1e-7), (
        "rescale=False should not change surviving delta values"
    )


def test_magnitude_prune_density_1_no_pruning():
    """density=1.0 keeps the entire delta unchanged (before rescale)."""
    delta = torch.randn(64)
    result = magnitude_prune_delta(delta, density=1.0, rescale=False)
    assert torch.allclose(result, delta), "density=1.0 should leave delta unchanged"


def test_magnitude_prune_density_0_all_zeros():
    """density=0.0 should zero out the entire delta."""
    delta = torch.randn(64)
    result = magnitude_prune_delta(delta, density=0.0, rescale=True)
    assert (result == 0).all(), "density=0.0 should produce an all-zero tensor"


# ---------------------------------------------------------------------------
# Tests for della_merge (functional API)
# ---------------------------------------------------------------------------


def test_della_merge_returns_same_keys_as_base():
    """della_merge output must contain exactly the same keys as base_model."""
    config = DELLAConfig(density=0.5, merge_method="mean")
    merged = della_merge(BASE, [FT1, FT2], config)
    assert set(merged.keys()) == set(BASE.keys()), (
        "Merged state dict should have the same keys as the base model"
    )


def test_della_merge_differs_from_base():
    """The merged model should differ from the base (delta was applied)."""
    config = DELLAConfig(density=0.5, rescale=True, merge_method="mean")
    merged = della_merge(BASE, [FT1], config)

    any_diff = any(not torch.allclose(merged[k].float(), BASE[k].float()) for k in BASE)
    assert any_diff, "Merged model must differ from the pretrained base"


# ---------------------------------------------------------------------------
# Tests for DELLAMerger class
# ---------------------------------------------------------------------------


def test_merger_compute_delta_correct_difference():
    """compute_delta should return finetuned - base for every parameter."""
    merger = DELLAMerger(DELLAConfig())
    delta = merger.compute_delta(BASE, FT1)

    for key in BASE:
        expected = FT1[key].float() - BASE[key].float()
        assert torch.allclose(delta[key], expected, atol=1e-6), f"Delta mismatch for key '{key}'"


def test_merger_density_1_no_pruning():
    """With density=1.0 the delta after pruning should equal the raw delta."""
    config = DELLAConfig(density=1.0, rescale=False)
    merger = DELLAMerger(config)

    raw_delta = merger.compute_delta(BASE, FT1)
    pruned = merger.prune_delta(raw_delta)

    for key in raw_delta:
        assert torch.allclose(pruned[key], raw_delta[key], atol=1e-6), (
            f"density=1.0 should not prune key '{key}'"
        )


def test_merger_density_0_all_zeros():
    """With density=0.0 every pruned delta tensor should be all-zero."""
    config = DELLAConfig(density=0.0, rescale=True)
    merger = DELLAMerger(config)

    raw_delta = merger.compute_delta(BASE, FT1)
    pruned = merger.prune_delta(raw_delta)

    for key, tensor in pruned.items():
        assert (tensor == 0).all(), f"density=0.0 should give all-zero delta for key '{key}'"


def test_ties_sign_election_matches_majority():
    """TIES merge: elected sign per position must match the majority sign."""
    config = DELLAConfig(density=1.0, rescale=False, merge_method="ties")
    merger = DELLAMerger(config)

    # Construct two deltas with controlled signs so we can verify majority vote
    # For weights: make FT1 and FT2 diverge from base in a structured way
    merged = merger.merge_with_ties(BASE, [FT1, FT2, FT3])

    # Manually compute what the elected sign should be for each key
    merger_raw = DELLAMerger(DELLAConfig(density=1.0, rescale=False))
    deltas = [merger_raw.prune_delta(merger_raw.compute_delta(BASE, ft)) for ft in [FT1, FT2, FT3]]

    for key in BASE:
        stacked = torch.stack([d[key].float() for d in deltas], dim=0)
        elected_sign = torch.sign(stacked.sum(dim=0))
        # The merged delta should lie in the same direction as elected_sign
        merged_delta = merged[key].float() - BASE[key].float()
        # Where elected_sign != 0, merged_delta should agree in sign
        nonzero = elected_sign != 0
        if nonzero.any():
            sign_agree = torch.sign(merged_delta[nonzero]) == elected_sign[nonzero]
            assert sign_agree.all(), (
                f"TIES: merged sign does not match elected sign for key '{key}'"
            )


def test_weighted_merge_zero_weight_ignores_first_model():
    """weights=[0.0, 1.0] should make the merged result reflect only FT2's delta."""
    config = DELLAConfig(
        density=1.0,
        rescale=False,
        merge_method="weighted",
        weights=[0.0, 1.0],
    )
    merger = DELLAMerger(config)
    merged = merger.merge(BASE, [FT1, FT2])

    # Expected: base + pruned_delta_FT2 (since weight for FT1 = 0)
    raw_delta_ft2 = merger.compute_delta(BASE, FT2)
    pruned_ft2 = merger.prune_delta(raw_delta_ft2)

    for key in BASE:
        expected = BASE[key].float() + pruned_ft2[key].float()
        assert torch.allclose(merged[key].float(), expected, atol=1e-5), (
            f"weights=[0,1] should use only FT2's delta for key '{key}'"
        )


def test_identical_finetuned_models_same_as_one():
    """Merging N identical fine-tuned copies should equal merging a single one."""
    config = DELLAConfig(density=0.5, rescale=True, merge_method="mean")
    merger = DELLAMerger(config)

    merged_one = merger.merge(BASE, [FT1])
    merged_three = merger.merge(BASE, [FT1, FT1, FT1])

    for key in BASE:
        assert torch.allclose(merged_one[key].float(), merged_three[key].float(), atol=1e-5), (
            f"N identical models should give the same result as one for key '{key}'"
        )
