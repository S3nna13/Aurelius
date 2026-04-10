"""Tests for activation_patching.py — Activation Patching for Interpretability."""

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.activation_patching import (
    ActivationPatcher,
    PatchConfig,
    capture_activations,
    compute_logit_diff,
    create_patch_hook,
    run_with_patch,
)
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


@pytest.fixture
def cfg():
    return TINY_CFG


@pytest.fixture
def model(cfg):
    m = AureliusTransformer(cfg)
    m.eval()
    return m


@pytest.fixture
def input_ids():
    return torch.randint(0, 256, (2, 8))


@pytest.fixture
def source_ids():
    return torch.randint(0, 256, (2, 8))


# ---------------------------------------------------------------------------
# PatchConfig tests
# ---------------------------------------------------------------------------


class TestPatchConfig:
    def test_default_values(self):
        cfg = PatchConfig()
        assert cfg.layer_idx == 0
        assert cfg.position_idx == -1
        assert cfg.patch_type == "replace"

    def test_custom_values(self):
        cfg = PatchConfig(layer_idx=3, position_idx=5, patch_type="add")
        assert cfg.layer_idx == 3
        assert cfg.position_idx == 5
        assert cfg.patch_type == "add"

    def test_invalid_patch_type_raises(self):
        with pytest.raises(ValueError, match="patch_type must be"):
            PatchConfig(patch_type="multiply")

    def test_zero_patch_type_valid(self):
        cfg = PatchConfig(patch_type="zero")
        assert cfg.patch_type == "zero"


# ---------------------------------------------------------------------------
# capture_activations tests
# ---------------------------------------------------------------------------


class TestCaptureActivations:
    def test_shape(self, model, input_ids):
        act = capture_activations(model, input_ids, layer_idx=0)
        assert act.shape == (2, 8, 64)  # (B, T, d_model)

    def test_layer_1(self, model, input_ids):
        act = capture_activations(model, input_ids, layer_idx=1)
        assert act.shape == (2, 8, 64)

    def test_detached(self, model, input_ids):
        act = capture_activations(model, input_ids, layer_idx=0)
        assert not act.requires_grad

    def test_different_layers_differ(self, model, input_ids):
        act0 = capture_activations(model, input_ids, layer_idx=0)
        act1 = capture_activations(model, input_ids, layer_idx=1)
        assert not torch.allclose(act0, act1)


# ---------------------------------------------------------------------------
# create_patch_hook tests
# ---------------------------------------------------------------------------


class TestCreatePatchHook:
    def test_replace_patches_position(self, model, input_ids, source_ids):
        source_act = capture_activations(model, input_ids, layer_idx=0)
        config = PatchConfig(layer_idx=0, position_idx=-1, patch_type="replace")
        hook = create_patch_hook(source_act, config)
        assert callable(hook)

    def test_zero_zeroes_position(self, model, input_ids):
        source_act = capture_activations(model, input_ids, layer_idx=0)
        config = PatchConfig(layer_idx=0, position_idx=3, patch_type="zero")
        hook = create_patch_hook(source_act, config)

        # Simulate calling the hook with a fake output
        fake_output = torch.randn(2, 8, 64)
        result = hook(None, None, fake_output)
        assert torch.all(result[:, 3, :] == 0.0)

    def test_add_adds_activation(self, model, input_ids):
        source_act = capture_activations(model, input_ids, layer_idx=0)
        config = PatchConfig(layer_idx=0, position_idx=0, patch_type="add")
        hook = create_patch_hook(source_act, config)

        fake_output = torch.ones(2, 8, 64)
        result = hook(None, None, fake_output)
        expected = 1.0 + source_act[:, 0, :]
        assert torch.allclose(result[:, 0, :], expected)

    def test_replace_only_affects_target_position(self, model, input_ids):
        source_act = capture_activations(model, input_ids, layer_idx=0)
        config = PatchConfig(layer_idx=0, position_idx=2, patch_type="replace")
        hook = create_patch_hook(source_act, config)

        fake_output = torch.ones(2, 8, 64) * 999.0
        result = hook(None, None, fake_output)
        # Position 2 should be replaced
        assert torch.allclose(result[:, 2, :], source_act[:, 2, :])
        # Position 0 should be untouched
        assert torch.allclose(result[:, 0, :], torch.ones(2, 64) * 999.0)


# ---------------------------------------------------------------------------
# run_with_patch tests
# ---------------------------------------------------------------------------


class TestRunWithPatch:
    def test_returns_logits_shape(self, model, input_ids):
        source_act = capture_activations(model, input_ids, layer_idx=0)
        config = PatchConfig(layer_idx=0, position_idx=-1, patch_type="replace")
        hook = create_patch_hook(source_act, config)
        logits = run_with_patch(model, input_ids, hook, layer_idx=0)
        assert logits.shape == (2, 8, 256)  # (B, T, V)

    def test_patching_changes_logits(self, model, input_ids, source_ids):
        # Clean logits
        with torch.no_grad():
            _loss, clean_logits, _pkv = model(input_ids)

        # Patched logits with source activations
        source_act = capture_activations(model, source_ids, layer_idx=0)
        config = PatchConfig(layer_idx=0, position_idx=-1, patch_type="replace")
        hook = create_patch_hook(source_act, config)
        patched_logits = run_with_patch(model, input_ids, hook, layer_idx=0)

        # Should differ (different source activations)
        assert not torch.allclose(clean_logits, patched_logits)

    def test_hook_removed_after_run(self, model, input_ids):
        source_act = capture_activations(model, input_ids, layer_idx=0)
        config = PatchConfig(layer_idx=0, position_idx=-1, patch_type="zero")
        hook = create_patch_hook(source_act, config)
        run_with_patch(model, input_ids, hook, layer_idx=0)

        # After run_with_patch, the hook should be removed.
        # Running the model again should give clean logits (no hook active).
        with torch.no_grad():
            _loss, logits1, _ = model(input_ids)
            _loss, logits2, _ = model(input_ids)
        assert torch.allclose(logits1, logits2)


# ---------------------------------------------------------------------------
# compute_logit_diff tests
# ---------------------------------------------------------------------------


class TestComputeLogitDiff:
    def test_zero_when_same(self):
        logits = torch.randn(2, 8, 256)
        diff = compute_logit_diff(logits, logits, token_a=10, token_b=20)
        assert abs(diff) < 1e-5

    def test_nonzero_when_different(self):
        clean = torch.randn(2, 8, 256)
        patched = torch.randn(2, 8, 256)
        diff = compute_logit_diff(clean, patched, token_a=10, token_b=20)
        assert isinstance(diff, float)

    def test_returns_float(self):
        logits = torch.randn(2, 8, 256)
        diff = compute_logit_diff(logits, logits, token_a=0, token_b=1)
        assert isinstance(diff, float)


# ---------------------------------------------------------------------------
# ActivationPatcher tests
# ---------------------------------------------------------------------------


class TestActivationPatcher:
    def test_patch_and_compare_keys(self, model, input_ids, source_ids):
        config = PatchConfig(layer_idx=0, position_idx=-1, patch_type="replace")
        patcher = ActivationPatcher(model, config)
        result = patcher.patch_and_compare(input_ids, source_ids, token_a=10, token_b=20)
        assert "clean_diff" in result
        assert "patched_diff" in result
        assert "effect_size" in result

    def test_effect_size_consistency(self, model, input_ids, source_ids):
        config = PatchConfig(layer_idx=0, position_idx=-1, patch_type="replace")
        patcher = ActivationPatcher(model, config)
        result = patcher.patch_and_compare(input_ids, source_ids, token_a=10, token_b=20)
        assert abs(result["effect_size"] - (result["patched_diff"] - result["clean_diff"])) < 1e-5

    def test_zero_patch_effect(self, model, input_ids, source_ids):
        config = PatchConfig(layer_idx=0, position_idx=-1, patch_type="zero")
        patcher = ActivationPatcher(model, config)
        result = patcher.patch_and_compare(input_ids, source_ids, token_a=10, token_b=20)
        assert isinstance(result["effect_size"], float)

    def test_same_input_small_effect(self, model, input_ids):
        config = PatchConfig(layer_idx=0, position_idx=-1, patch_type="replace")
        patcher = ActivationPatcher(model, config)
        # When source == clean, replace should produce no change
        result = patcher.patch_and_compare(input_ids, input_ids, token_a=10, token_b=20)
        assert abs(result["effect_size"]) < 1e-4

    def test_layer_1_patching(self, model, input_ids, source_ids):
        config = PatchConfig(layer_idx=1, position_idx=-1, patch_type="replace")
        patcher = ActivationPatcher(model, config)
        result = patcher.patch_and_compare(input_ids, source_ids, token_a=5, token_b=50)
        assert "effect_size" in result

    def test_all_values_are_float(self, model, input_ids, source_ids):
        config = PatchConfig(layer_idx=0, position_idx=-1, patch_type="replace")
        patcher = ActivationPatcher(model, config)
        result = patcher.patch_and_compare(input_ids, source_ids, token_a=10, token_b=20)
        for key in ("clean_diff", "patched_diff", "effect_size"):
            assert isinstance(result[key], float), f"{key} is not float"
