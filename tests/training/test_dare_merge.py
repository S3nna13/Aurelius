"""Tests for src/training/dare_merge.py — DARE and improved TIES merging."""

from __future__ import annotations

import pytest
import torch

from src.training.dare_merge import (
    MergingPipeline,
    compute_task_vector,
    dare_drop,
    dare_ties_merge,
    ties_elect_sign,
    ties_merge,
    ties_trim,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 16  # tiny model dimension


def _make_state(seed: int = 0) -> dict[str, torch.Tensor]:
    """Return a tiny state dict with reproducible random tensors."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return {
        "layer1.weight": torch.randn(D, D, generator=gen),
        "layer1.bias": torch.randn(D, generator=gen),
        "layer2.weight": torch.randn(D, D, generator=gen),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_task_vector_shape():
    base = _make_state(0)
    finetuned = _make_state(1)
    tv = compute_task_vector(finetuned, base)

    assert set(tv.keys()) == set(base.keys())
    for key in base:
        assert tv[key].shape == base[key].shape


def test_dare_drop_zeros_correct_fraction():
    """With drop_rate=0.9, ~90 % of values should be zero."""
    base = _make_state(0)
    finetuned = _make_state(1)
    tv = compute_task_vector(finetuned, base)

    dropped = dare_drop(tv, drop_rate=0.9, seed=42)

    all_values = torch.cat([v.flatten() for v in dropped.values()])
    zero_fraction = (all_values == 0).float().mean().item()
    # Allow ±5 pp tolerance for stochasticity over small tensors
    assert 0.80 <= zero_fraction <= 0.98, f"zero fraction = {zero_fraction:.3f}"


def test_dare_drop_rescales_remaining():
    """Rescaling preserves the expected magnitude of each parameter.

    After DARE, E[|dropped_param|] ≈ E[|original_param|].
    This holds because each value is kept with probability (1-p) and
    rescaled by 1/(1-p), so E[|result|] = (1-p) * |orig| / (1-p) = |orig|.
    We verify this over the whole tensor (including zeros).
    """
    # Use a large tensor so the law-of-large-numbers kicks in.
    torch.manual_seed(7)
    tv = {"big": torch.randn(10_000)}

    drop_rate = 0.9
    dropped = dare_drop(tv, drop_rate=drop_rate, seed=0)

    # Mean absolute value over ALL elements (including zeroed ones)
    orig_mean = tv["big"].abs().mean().item()
    result_mean = dropped["big"].abs().mean().item()

    # Expected magnitudes should be approximately equal (within 15%).
    assert abs(result_mean - orig_mean) / (orig_mean + 1e-8) < 0.15, (
        f"orig_mean={orig_mean:.4f}, result_mean={result_mean:.4f}"
    )


def test_dare_drop_reproducible():
    base = _make_state(0)
    finetuned = _make_state(1)
    tv = compute_task_vector(finetuned, base)

    dropped_a = dare_drop(tv, drop_rate=0.5, seed=99)
    dropped_b = dare_drop(tv, drop_rate=0.5, seed=99)

    for key in dropped_a:
        assert torch.equal(dropped_a[key], dropped_b[key])


def test_ties_trim_keeps_top_fraction():
    """With top_k=0.5, roughly 50 % of values should be non-zero."""
    base = _make_state(0)
    finetuned = _make_state(1)
    tv = compute_task_vector(finetuned, base)

    trimmed = ties_trim(tv, top_k=0.5)

    all_values = torch.cat([v.flatten() for v in trimmed.values()])
    nonzero_fraction = (all_values != 0).float().mean().item()
    assert 0.40 <= nonzero_fraction <= 0.65, f"nonzero fraction = {nonzero_fraction:.3f}"


def test_ties_elect_sign_majority():
    """2 positive + 1 negative task vectors → elected sign is positive."""
    # All entries positive in tv1 and tv2, negative in tv3
    tv1 = {"w": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    tv2 = {"w": torch.tensor([0.5, 1.5, 2.5, 3.5])}
    tv3 = {"w": torch.tensor([-2.0, -3.0, -4.0, -5.0])}

    elected = ties_elect_sign([tv1, tv2, tv3])
    # Sum = [−0.5, 0.5, 1.5, 2.5] → signs [−1, 1, 1, 1]
    assert elected["w"][0].item() == -1.0
    assert elected["w"][1].item() == 1.0
    assert elected["w"][2].item() == 1.0
    assert elected["w"][3].item() == 1.0


def test_ties_merge_output_shape():
    base = _make_state(0)
    ft1 = _make_state(1)
    ft2 = _make_state(2)

    tv1 = compute_task_vector(ft1, base)
    tv2 = compute_task_vector(ft2, base)

    merged = ties_merge([tv1, tv2], base, top_k=0.2, scaling_coeff=1.0)

    assert set(merged.keys()) == set(base.keys())
    for key in base:
        assert merged[key].shape == base[key].shape


def test_dare_ties_merge_differs_from_base():
    """The merged model should differ from the base (fine-tuning had an effect)."""
    base = _make_state(0)
    ft1 = _make_state(1)
    ft2 = _make_state(2)

    merged = dare_ties_merge([ft1, ft2], base, drop_rate=0.5, top_k=0.2, scaling_coeff=1.0, seed=0)

    any_diff = any(not torch.allclose(merged[k].float(), base[k].float()) for k in base)
    assert any_diff, "Merged state is identical to base — merge had no effect."


def test_merging_pipeline_simple_average():
    """Average of two identical fine-tuned models should equal that model."""
    base = _make_state(0)
    ft = _make_state(1)

    pipeline = MergingPipeline(base_state=base, finetuned_states=[ft, ft])
    merged = pipeline.merge(method="simple_average")

    for key in ft:
        assert torch.allclose(merged[key].float(), ft[key].float(), atol=1e-5), (
            f"simple_average of identical models should equal that model (key={key})"
        )


def test_evaluate_merge_quality_keys():
    base = _make_state(0)
    ft = _make_state(1)
    pipeline = MergingPipeline(base_state=base, finetuned_states=[ft])
    merged = pipeline.merge(method="simple_average")

    quality = pipeline.evaluate_merge_quality(merged, ft)

    assert "l2_distance" in quality
    assert "cosine_similarity" in quality
    assert "param_count" in quality
    assert isinstance(quality["l2_distance"], float)
    assert isinstance(quality["cosine_similarity"], float)
    assert isinstance(quality["param_count"], int)


def test_cosine_similarity_identical():
    """When merged == reference, cosine similarity should be ~1.0."""
    base = _make_state(0)
    ft = _make_state(1)
    pipeline = MergingPipeline(base_state=base, finetuned_states=[ft])

    # Use the fine-tuned state as both merged and reference
    quality = pipeline.evaluate_merge_quality(ft, ft)

    assert quality["cosine_similarity"] == pytest.approx(1.0, abs=1e-5), (
        f"Expected cosine_similarity ≈ 1.0, got {quality['cosine_similarity']}"
    )
