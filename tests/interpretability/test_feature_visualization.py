"""Tests for feature_visualization module."""
import pytest
from src.interpretability.feature_visualization import (
    PatchTarget,
    PatchResult,
    ActivationPatcher,
    FeatureDecomposer,
)


# ── PatchTarget enum ────────────────────────────────────────────────────────────

def test_patch_target_residual_stream():
    assert PatchTarget.RESIDUAL_STREAM == "residual_stream"

def test_patch_target_attention_output():
    assert PatchTarget.ATTENTION_OUTPUT == "attention_output"

def test_patch_target_mlp_output():
    assert PatchTarget.MLP_OUTPUT == "mlp_output"


# ── PatchResult fields ──────────────────────────────────────────────────────────

def test_patch_result_fields():
    r = PatchResult(
        target=PatchTarget.RESIDUAL_STREAM,
        layer=2,
        position=4,
        effect=0.5,
        baseline_output=1.0,
        patched_output=1.5,
    )
    assert r.target == PatchTarget.RESIDUAL_STREAM
    assert r.layer == 2
    assert r.position == 4
    assert r.effect == 0.5
    assert r.baseline_output == 1.0
    assert r.patched_output == 1.5

def test_patch_result_is_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(PatchResult)


# ── ActivationPatcher ───────────────────────────────────────────────────────────

def test_activation_patcher_init():
    patcher = ActivationPatcher()
    assert patcher is not None

def test_compute_effect_zero_when_unchanged():
    patcher = ActivationPatcher()
    effect = patcher.compute_effect(1.0, 1.0)
    assert effect == 0.0

def test_compute_effect_positive_when_different():
    patcher = ActivationPatcher()
    effect = patcher.compute_effect(1.0, 2.0)
    assert effect > 0.0

def test_compute_effect_formula():
    patcher = ActivationPatcher()
    baseline, patched = 2.0, 4.0
    expected = abs(patched - baseline) / (abs(baseline) + 1e-8)
    assert abs(patcher.compute_effect(baseline, patched) - expected) < 1e-9

def test_compute_effect_baseline_zero():
    patcher = ActivationPatcher()
    # baseline=0 → denom = 1e-8
    effect = patcher.compute_effect(0.0, 1.0)
    assert effect > 0.0
    assert abs(effect - 1.0 / 1e-8) < 1e-3

def test_patch_replaces_specified_positions():
    patcher = ActivationPatcher()
    original = [1.0, 2.0, 3.0, 4.0]
    patch_vals = [10.0, 20.0]
    positions = [1, 3]
    result = patcher.patch(original, patch_vals, positions)
    assert result[1] == 10.0
    assert result[3] == 20.0

def test_patch_non_specified_positions_unchanged():
    patcher = ActivationPatcher()
    original = [1.0, 2.0, 3.0, 4.0]
    patch_vals = [99.0]
    positions = [2]
    result = patcher.patch(original, patch_vals, positions)
    assert result[0] == 1.0
    assert result[1] == 2.0
    assert result[3] == 4.0

def test_patch_returns_new_list():
    patcher = ActivationPatcher()
    original = [1.0, 2.0, 3.0]
    result = patcher.patch(original, [0.0], [0])
    assert result is not original

def test_patch_all_positions():
    patcher = ActivationPatcher()
    original = [1.0, 2.0, 3.0]
    result = patcher.patch(original, [7.0, 8.0, 9.0], [0, 1, 2])
    assert result == [7.0, 8.0, 9.0]

def test_causal_trace_returns_patch_result():
    patcher = ActivationPatcher()
    result = patcher.causal_trace(PatchTarget.MLP_OUTPUT, 3, 5, 1.0, 2.0)
    assert isinstance(result, PatchResult)

def test_causal_trace_fields():
    patcher = ActivationPatcher()
    result = patcher.causal_trace(PatchTarget.ATTENTION_OUTPUT, 1, 2, 3.0, 6.0)
    assert result.target == PatchTarget.ATTENTION_OUTPUT
    assert result.layer == 1
    assert result.position == 2
    assert result.baseline_output == 3.0
    assert result.patched_output == 6.0

def test_causal_trace_effect_computed():
    patcher = ActivationPatcher()
    result = patcher.causal_trace(PatchTarget.RESIDUAL_STREAM, 0, 0, 2.0, 4.0)
    expected_effect = patcher.compute_effect(2.0, 4.0)
    assert abs(result.effect - expected_effect) < 1e-9

def test_top_effects_returns_top_k():
    patcher = ActivationPatcher()
    results = [
        PatchResult(PatchTarget.MLP_OUTPUT, i, 0, float(i), 1.0, 1.0)
        for i in range(10)
    ]
    top = patcher.top_effects(results, k=3)
    assert len(top) == 3

def test_top_effects_sorted_descending():
    patcher = ActivationPatcher()
    results = [
        PatchResult(PatchTarget.MLP_OUTPUT, 0, 0, 0.3, 1.0, 1.0),
        PatchResult(PatchTarget.MLP_OUTPUT, 1, 0, 0.9, 1.0, 1.0),
        PatchResult(PatchTarget.MLP_OUTPUT, 2, 0, 0.1, 1.0, 1.0),
        PatchResult(PatchTarget.MLP_OUTPUT, 3, 0, 0.7, 1.0, 1.0),
    ]
    top = patcher.top_effects(results, k=2)
    assert top[0].effect >= top[1].effect

def test_top_effects_k_larger_than_list():
    patcher = ActivationPatcher()
    results = [
        PatchResult(PatchTarget.RESIDUAL_STREAM, 0, 0, 0.5, 1.0, 1.0),
    ]
    top = patcher.top_effects(results, k=10)
    assert len(top) == 1

def test_top_effects_default_k():
    patcher = ActivationPatcher()
    results = [
        PatchResult(PatchTarget.MLP_OUTPUT, i, 0, float(i), 1.0, 1.0)
        for i in range(10)
    ]
    top = patcher.top_effects(results)
    assert len(top) == 5


# ── FeatureDecomposer ───────────────────────────────────────────────────────────

def test_feature_decomposer_init():
    fd = FeatureDecomposer(n_components=5)
    assert fd.n_components == 5

def test_feature_decomposer_default_n_components():
    fd = FeatureDecomposer()
    assert fd.n_components == 10

def test_pca_stub_output_shape():
    fd = FeatureDecomposer(n_components=3)
    acts = [[1.0, 2.0, 3.0, 4.0, 5.0]] * 4
    out = fd.pca_stub(acts)
    assert len(out) == 4
    for row in out:
        assert len(row) == 3

def test_pca_stub_fewer_features_than_components():
    fd = FeatureDecomposer(n_components=10)
    acts = [[1.0, 2.0]] * 3
    out = fd.pca_stub(acts)
    assert len(out) == 3
    for row in out:
        assert len(row) == 2  # capped at actual features

def test_pca_stub_mean_centered():
    fd = FeatureDecomposer(n_components=2)
    acts = [[1.0, 2.0], [3.0, 4.0]]
    out = fd.pca_stub(acts)
    mean0 = sum(row[0] for row in out) / len(out)
    assert abs(mean0) < 1e-9

def test_pca_stub_empty():
    fd = FeatureDecomposer(n_components=5)
    out = fd.pca_stub([])
    assert out == []

def test_reconstruction_error_zero_identical():
    fd = FeatureDecomposer()
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    err = fd.reconstruction_error(data, data)
    assert err == 0.0

def test_reconstruction_error_positive_when_different():
    fd = FeatureDecomposer()
    original = [[1.0, 2.0], [3.0, 4.0]]
    reconstructed = [[0.0, 0.0], [0.0, 0.0]]
    err = fd.reconstruction_error(original, reconstructed)
    assert err > 0.0

def test_reconstruction_error_formula():
    fd = FeatureDecomposer()
    original = [[1.0, 2.0]]
    reconstructed = [[0.0, 0.0]]
    # MSE of [1,2] vs [0,0] = (1+4)/2 = 2.5
    err = fd.reconstruction_error(original, reconstructed)
    assert abs(err - 2.5) < 1e-9

def test_reconstruction_error_symmetric():
    fd = FeatureDecomposer()
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[2.0, 3.0], [4.0, 5.0]]
    assert abs(fd.reconstruction_error(a, b) - fd.reconstruction_error(b, a)) < 1e-9
