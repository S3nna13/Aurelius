"""
Tests for src/interpretability/activation_patching.py

Tiny configuration:
    D=16, T=4, B=2, V=8, H=2 (heads for attention similarity tests)
"""

import pytest
import torch
import torch.nn as nn

from src.interpretability.activation_patching import (
    PatchConfig,
    ActivationStore,
    patch_activations,
    compute_patching_effect,
    PatchingExperiment,
    attention_pattern_similarity,
)

# ---------------------------------------------------------------------------
# Shared tiny constants
# ---------------------------------------------------------------------------
D = 16
T = 4
B = 2
V = 8
H = 2

torch.manual_seed(0)


def _make_model() -> nn.Sequential:
    """2-layer linear model: D -> D -> V."""
    return nn.Sequential(
        nn.Linear(D, D, bias=False),
        nn.Linear(D, V, bias=False),
    )


def _make_input() -> torch.Tensor:
    """Random input of shape (B, T, D)."""
    return torch.randn(B, T, D)


# ---------------------------------------------------------------------------
# 1. PatchConfig defaults
# ---------------------------------------------------------------------------

def test_patchconfig_defaults():
    cfg = PatchConfig()
    assert cfg.patch_layers == []
    assert cfg.patch_positions is None
    assert cfg.patch_heads is None
    assert cfg.normalize_effect is True


# ---------------------------------------------------------------------------
# 2. ActivationStore captures hook correctly
# ---------------------------------------------------------------------------

def test_activation_store_hook_captures():
    store = ActivationStore()
    layer = nn.Linear(D, D, bias=False)
    x = torch.randn(B, T, D)
    hook = layer.register_forward_hook(store.hook_fn("layer0"))
    with torch.no_grad():
        out = layer(x)
    hook.remove()
    assert "layer0" in store.store
    assert store.store["layer0"].shape == (B, T, D)


# ---------------------------------------------------------------------------
# 3. ActivationStore.get returns tensor
# ---------------------------------------------------------------------------

def test_activation_store_get_returns_tensor():
    store = ActivationStore()
    layer = nn.Linear(D, D, bias=False)
    x = torch.randn(B, T, D)
    hook = layer.register_forward_hook(store.hook_fn("layer0"))
    with torch.no_grad():
        layer(x)
    hook.remove()
    result = store.get("layer0")
    assert isinstance(result, torch.Tensor)
    assert result.shape == (B, T, D)


# ---------------------------------------------------------------------------
# 4. ActivationStore.clear empties the store
# ---------------------------------------------------------------------------

def test_activation_store_clear_empties():
    store = ActivationStore()
    layer = nn.Linear(D, D, bias=False)
    x = torch.randn(B, T, D)
    hook = layer.register_forward_hook(store.hook_fn("layer0"))
    with torch.no_grad():
        layer(x)
    hook.remove()
    assert len(store.store) > 0
    store.clear()
    assert len(store.store) == 0


# ---------------------------------------------------------------------------
# 5. patch_activations with positions=None replaces all positions
# ---------------------------------------------------------------------------

def test_patch_activations_all_positions_replaces_all():
    target = torch.zeros(B, T, D)
    source = torch.ones(B, T, D)
    patched = patch_activations(target, source, positions=None)
    assert torch.allclose(patched, source), "All positions should be replaced by source"


# ---------------------------------------------------------------------------
# 6. patch_activations with specific positions leaves others unchanged
# ---------------------------------------------------------------------------

def test_patch_activations_specific_positions_leaves_others():
    target = torch.zeros(B, T, D)
    source = torch.ones(B, T, D) * 99.0
    positions = [1, 3]  # only patch positions 1 and 3 out of 0..T-1
    patched = patch_activations(target, source, positions=positions)

    # patched positions should equal source
    for p in positions:
        assert torch.allclose(patched[..., p, :], source[..., p, :])

    # untouched positions should equal original target (zeros)
    for p in range(T):
        if p not in positions:
            assert torch.allclose(patched[..., p, :], target[..., p, :])


# ---------------------------------------------------------------------------
# 7. compute_patching_effect shape matches logits
# ---------------------------------------------------------------------------

def test_compute_patching_effect_shape():
    original = torch.randn(B, T, V)
    patched  = torch.randn(B, T, V)
    clean    = torch.randn(B, T, V)
    effect = compute_patching_effect(original, patched, clean)
    assert effect.shape == original.shape


# ---------------------------------------------------------------------------
# 8. compute_patching_effect with identical original and clean ≈ 0
# ---------------------------------------------------------------------------

def test_compute_patching_effect_identical_original_clean_is_zero():
    """When original == clean, denominator ≈ 1e-8, numerator is arbitrary,
    but more importantly when patched == original the effect should be 0."""
    original = torch.randn(B, T, V)
    clean    = original.clone()   # same as original
    patched  = original.clone()   # no change
    effect = compute_patching_effect(original, patched, clean)
    # numerator = patched - original = 0, so effect = 0 regardless of denom
    assert torch.allclose(effect, torch.zeros_like(effect), atol=1e-6)


# ---------------------------------------------------------------------------
# 9. PatchingExperiment.capture_activations returns dict with correct keys
# ---------------------------------------------------------------------------

def test_capture_activations_returns_correct_keys():
    model = _make_model()
    config = PatchConfig()
    exp = PatchingExperiment(model, config)
    x = _make_input()
    hook_points = ["0", "1"]  # Sequential children names
    acts = exp.capture_activations(x, hook_points)
    for key in hook_points:
        assert key in acts, f"Key '{key}' missing from captured activations"


# ---------------------------------------------------------------------------
# 10. PatchingExperiment.compute_layer_importance returns dict
# ---------------------------------------------------------------------------

def test_compute_layer_importance_returns_dict():
    model = _make_model()
    config = PatchConfig()
    exp = PatchingExperiment(model, config)
    clean_x     = _make_input()
    corrupted_x = _make_input()
    hook_points = ["0"]
    result = exp.compute_layer_importance(clean_x, corrupted_x, hook_points)
    assert isinstance(result, dict)
    assert "0" in result


# ---------------------------------------------------------------------------
# 11. Importance values are non-negative floats
# ---------------------------------------------------------------------------

def test_compute_layer_importance_values_non_negative():
    model = _make_model()
    config = PatchConfig()
    exp = PatchingExperiment(model, config)
    clean_x     = _make_input()
    corrupted_x = _make_input()
    hook_points = ["0", "1"]
    result = exp.compute_layer_importance(clean_x, corrupted_x, hook_points)
    for key, val in result.items():
        assert isinstance(val, float), f"Value for '{key}' is not float: {type(val)}"
        assert val >= 0.0, f"Importance for '{key}' is negative: {val}"


# ---------------------------------------------------------------------------
# 12. attention_pattern_similarity shape is (B, H)
# ---------------------------------------------------------------------------

def test_attention_pattern_similarity_shape():
    attn_a = torch.randn(B, H, T, T)
    attn_b = torch.randn(B, H, T, T)
    sim = attention_pattern_similarity(attn_a, attn_b)
    assert sim.shape == (B, H), f"Expected ({B},{H}), got {sim.shape}"


# ---------------------------------------------------------------------------
# 13. similarity of identical patterns = 1.0
# ---------------------------------------------------------------------------

def test_attention_pattern_similarity_identical_is_one():
    attn = torch.randn(B, H, T, T)
    sim = attention_pattern_similarity(attn, attn)
    assert torch.allclose(sim, torch.ones(B, H), atol=1e-5), (
        f"Identical attention patterns should have cosine similarity 1.0, got {sim}"
    )


# ---------------------------------------------------------------------------
# 14. patch_activations returns a new tensor (does not mutate target)
# ---------------------------------------------------------------------------

def test_patch_activations_does_not_mutate_target():
    target = torch.zeros(B, T, D)
    source = torch.ones(B, T, D)
    target_before = target.clone()
    _ = patch_activations(target, source, positions=None)
    assert torch.allclose(target, target_before), "patch_activations must not mutate target"


# ---------------------------------------------------------------------------
# 15. PatchingExperiment.run_patched_forward returns correct shape
# ---------------------------------------------------------------------------

def test_run_patched_forward_output_shape():
    model = _make_model()
    config = PatchConfig()
    exp = PatchingExperiment(model, config)
    clean_x = _make_input()
    acts = exp.capture_activations(clean_x, ["0"])
    corrupted_x = _make_input()
    out = exp.run_patched_forward(corrupted_x, acts, ["0"])
    assert out.shape == (B, T, V), f"Expected ({B},{T},{V}), got {out.shape}"


# ---------------------------------------------------------------------------
# 16. compute_patching_effect: full restoration yields effect ≈ 1
# ---------------------------------------------------------------------------

def test_compute_patching_effect_full_restoration():
    """When patched_logits == clean_logits, effect should be ≈ 1."""
    original = torch.zeros(B, T, V)
    clean    = torch.ones(B, T, V) * 2.0
    patched  = clean.clone()
    effect = compute_patching_effect(original, patched, clean)
    assert torch.allclose(effect, torch.ones_like(effect), atol=1e-5)


# ---------------------------------------------------------------------------
# 17. attention_pattern_similarity values are in [-1, 1]
# ---------------------------------------------------------------------------

def test_attention_pattern_similarity_range():
    attn_a = torch.randn(B, H, T, T)
    attn_b = torch.randn(B, H, T, T)
    sim = attention_pattern_similarity(attn_a, attn_b)
    assert (sim >= -1.0 - 1e-5).all() and (sim <= 1.0 + 1e-5).all(), (
        f"Cosine similarity must be in [-1, 1], got min={sim.min()}, max={sim.max()}"
    )
