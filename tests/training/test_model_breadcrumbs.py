"""Tests for model_breadcrumbs.py -- Panda et al. 2023 selective unlearning."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.model_breadcrumbs import (
    BreadcrumbConfig,
    BreadcrumbResult,
    BreadcrumbUnlearner,
    apply_breadcrumb_perturbation,
    compute_weight_masks,
    measure_weight_change,
    select_weight_mask,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _tiny_model() -> AureliusTransformer:
    """Tiny AureliusTransformer: 2 layers, d_model=64, vocab=256, seq=32."""
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    return AureliusTransformer(cfg)


def _default_config(sparsity: float = 0.9) -> BreadcrumbConfig:
    """BreadcrumbConfig with 10% of weights changed for fast tests."""
    return BreadcrumbConfig(
        sparsity=sparsity,
        perturbation_scale=0.01,
        n_iterations=1,
        selection_method="magnitude",
        seed=42,
    )


# ---------------------------------------------------------------------------
# Tests: select_weight_mask
# ---------------------------------------------------------------------------


class TestSelectWeightMask:
    """Tests 1-3: mask shape, fraction, and methods."""

    def test_magnitude_returns_bool_same_shape(self):
        """Test 1: magnitude method returns bool tensor with same shape as param."""
        torch.manual_seed(0)
        param = torch.randn(64, 32)
        mask = select_weight_mask(param, sparsity=0.9, method="magnitude")

        assert mask.shape == param.shape, "Mask shape must match param shape"
        assert mask.dtype == torch.bool, "Mask dtype must be bool"

    def test_magnitude_fraction_correct(self):
        """Test 2: magnitude mask has approx (1 - sparsity) fraction True."""
        torch.manual_seed(1)
        param = torch.randn(1000)
        sparsity = 0.9
        mask = select_weight_mask(param, sparsity=sparsity, method="magnitude")

        fraction_true = mask.float().mean().item()
        expected = 1.0 - sparsity  # 0.10
        assert abs(fraction_true - expected) <= 0.05, (
            f"Expected ~{expected:.3f} fraction True, got {fraction_true:.3f}"
        )

    def test_random_fraction_correct(self):
        """Test 3: random method also returns approx correct True fraction."""
        torch.manual_seed(2)
        param = torch.randn(1000)
        sparsity = 0.9
        mask = select_weight_mask(param, sparsity=sparsity, method="random")

        assert mask.shape == param.shape
        assert mask.dtype == torch.bool
        fraction_true = mask.float().mean().item()
        expected = 1.0 - sparsity
        assert abs(fraction_true - expected) <= 0.05, (
            f"Expected ~{expected:.3f} fraction True, got {fraction_true:.3f}"
        )

    def test_magnitude_selects_smallest_abs(self):
        """Sanity: magnitude method should prefer weights near zero."""
        # Force a weight near zero so it is definitely selected
        param = torch.ones(100) * 10.0
        param[0] = 0.0001  # clearly smallest magnitude
        mask = select_weight_mask(param, sparsity=0.9, method="magnitude")
        assert mask[0].item() is True, "Smallest-magnitude weight must be selected"

    def test_gradient_method_basic(self):
        """Gradient method returns correct shape with a dummy gradient."""
        torch.manual_seed(3)
        param = torch.randn(64, 32)
        grad = torch.randn_like(param)
        mask = select_weight_mask(param, sparsity=0.9, method="gradient", grad=grad)

        assert mask.shape == param.shape
        assert mask.dtype == torch.bool

    def test_invalid_method_raises(self):
        """Invalid method string should raise ValueError."""
        param = torch.randn(32)
        with pytest.raises(ValueError, match="Unknown selection_method"):
            select_weight_mask(param, sparsity=0.9, method="bogus")


# ---------------------------------------------------------------------------
# Tests: apply_breadcrumb_perturbation
# ---------------------------------------------------------------------------


class TestApplyBreadcrumbPerturbation:
    """Tests 4-6: return type, weights changed, weights unchanged."""

    def _simple_model_and_masks(self, sparsity: float = 0.9):
        model = _tiny_model()
        config = _default_config(sparsity=sparsity)
        masks = compute_weight_masks(model, config)
        return model, masks

    def test_returns_nn_module(self):
        """Test 4: apply_breadcrumb_perturbation returns an nn.Module."""
        model, masks = self._simple_model_and_masks()
        result = apply_breadcrumb_perturbation(model, masks, scale=0.01, seed=42)
        assert isinstance(result, nn.Module), "Must return nn.Module"

    def test_changed_weights_are_different(self):
        """Test 5: weights at True mask positions differ from original."""
        model, masks = self._simple_model_and_masks(sparsity=0.9)
        new_model = apply_breadcrumb_perturbation(model, masks, scale=0.01, seed=42)

        orig_params = dict(model.named_parameters())
        new_params = dict(new_model.named_parameters())

        any_changed = False
        for name, mask in masks.items():
            if name not in orig_params or name not in new_params:
                continue
            # At least some True positions should have changed
            orig = orig_params[name].detach()
            new = new_params[name].detach()
            if mask.any():
                changed_vals = orig[mask] != new[mask]
                if changed_vals.any():
                    any_changed = True
                    break

        assert any_changed, "At least some masked weights should be different after perturbation"

    def test_unchanged_weights_are_identical(self):
        """Test 6: weights at False mask positions are bit-identical to original."""
        model, masks = self._simple_model_and_masks(sparsity=0.9)
        new_model = apply_breadcrumb_perturbation(model, masks, scale=0.01, seed=42)

        orig_params = dict(model.named_parameters())
        new_params = dict(new_model.named_parameters())

        for name, mask in masks.items():
            if name not in orig_params or name not in new_params:
                continue
            orig = orig_params[name].detach()
            new = new_params[name].detach()
            unchanged_mask = ~mask
            if unchanged_mask.any():
                assert torch.all(orig[unchanged_mask] == new[unchanged_mask]), (
                    f"Param {name!r}: mask=False positions must remain unchanged"
                )

    def test_original_model_not_modified(self):
        """Applying perturbation must not alter the source model in-place."""
        model, masks = self._simple_model_and_masks()
        orig_state = {n: p.detach().clone() for n, p in model.named_parameters()}
        apply_breadcrumb_perturbation(model, masks, scale=0.01, seed=42)

        for name, orig_p in orig_state.items():
            current_p = dict(model.named_parameters())[name].detach()
            assert torch.equal(orig_p, current_p), (
                f"Source model param {name!r} was modified in-place"
            )


# ---------------------------------------------------------------------------
# Tests: compute_weight_masks
# ---------------------------------------------------------------------------


class TestComputeWeightMasks:
    """Tests 7-8: correct param names, all masks are bool tensors."""

    def test_returns_correct_param_names(self):
        """Test 7: compute_weight_masks keys match model's trainable param names."""
        model = _tiny_model()
        config = _default_config()
        masks = compute_weight_masks(model, config)

        expected_names = {n for n, p in model.named_parameters() if p.requires_grad}
        assert set(masks.keys()) == expected_names, (
            f"Mask keys mismatch.\n  Expected: {expected_names}\n  Got: {set(masks.keys())}"
        )

    def test_all_masks_are_bool(self):
        """Test 8: every mask tensor has dtype=torch.bool."""
        model = _tiny_model()
        config = _default_config()
        masks = compute_weight_masks(model, config)

        for name, mask in masks.items():
            assert mask.dtype == torch.bool, (
                f"Mask for {name!r} has dtype {mask.dtype}, expected torch.bool"
            )

    def test_masks_match_param_shapes(self):
        """Each mask must have the same shape as its corresponding parameter."""
        model = _tiny_model()
        config = _default_config()
        masks = compute_weight_masks(model, config)

        param_dict = dict(model.named_parameters())
        for name, mask in masks.items():
            assert mask.shape == param_dict[name].shape, (
                f"Mask shape {mask.shape} != param shape {param_dict[name].shape} for {name!r}"
            )


# ---------------------------------------------------------------------------
# Tests: BreadcrumbUnlearner
# ---------------------------------------------------------------------------


class TestBreadcrumbUnlearner:
    """Tests 9-10: run returns correct types and fraction_changed is accurate."""

    def test_run_returns_model_and_result(self):
        """Test 9: BreadcrumbUnlearner.run returns (nn.Module, BreadcrumbResult)."""
        torch.manual_seed(0)
        model = _tiny_model()
        config = _default_config(sparsity=0.9)
        unlearner = BreadcrumbUnlearner(model, config)
        result_model, result = unlearner.run()

        assert isinstance(result_model, nn.Module), "First element must be nn.Module"
        assert isinstance(result, BreadcrumbResult), "Second element must be BreadcrumbResult"

    def test_fraction_changed_approx_correct(self):
        """Test 10: BreadcrumbResult.fraction_changed ~= (1 - sparsity) +/- 0.05."""
        torch.manual_seed(0)
        model = _tiny_model()
        sparsity = 0.9
        config = _default_config(sparsity=sparsity)
        unlearner = BreadcrumbUnlearner(model, config)
        _, result = unlearner.run()

        expected = 1.0 - sparsity  # 0.10
        assert abs(result.fraction_changed - expected) <= 0.05, (
            f"fraction_changed={result.fraction_changed:.4f} too far from {expected:.3f}"
        )

    def test_result_has_perturbation_norms(self):
        """BreadcrumbResult.perturbation_norms should be non-empty."""
        torch.manual_seed(0)
        model = _tiny_model()
        config = _default_config()
        unlearner = BreadcrumbUnlearner(model, config)
        _, result = unlearner.run()

        assert len(result.perturbation_norms) > 0, "perturbation_norms must not be empty"

    def test_n_params_changed_positive(self):
        """n_params_changed must be > 0 after a run."""
        torch.manual_seed(0)
        model = _tiny_model()
        config = _default_config()
        unlearner = BreadcrumbUnlearner(model, config)
        _, result = unlearner.run()

        assert result.n_params_changed > 0, "n_params_changed should be positive"

    def test_compute_masks_then_apply(self):
        """Calling compute_masks then apply separately should also work."""
        torch.manual_seed(0)
        model = _tiny_model()
        config = _default_config()
        unlearner = BreadcrumbUnlearner(model, config)
        masks = unlearner.compute_masks()
        new_model, result = unlearner.apply(masks)

        assert isinstance(new_model, nn.Module)
        assert isinstance(result, BreadcrumbResult)

    def test_random_selection_run(self):
        """BreadcrumbUnlearner works with selection_method='random'."""
        torch.manual_seed(0)
        model = _tiny_model()
        config = BreadcrumbConfig(
            sparsity=0.9,
            perturbation_scale=0.01,
            n_iterations=1,
            selection_method="random",
            seed=42,
        )
        unlearner = BreadcrumbUnlearner(model, config)
        new_model, result = unlearner.run()

        assert isinstance(new_model, nn.Module)
        assert result.n_params_changed > 0


# ---------------------------------------------------------------------------
# Tests: measure_weight_change
# ---------------------------------------------------------------------------


class TestMeasureWeightChange:
    """Tests 11-12: returns dict with 'total_change' > 0 after perturbation."""

    def test_returns_dict_with_total_change(self):
        """Test 11: measure_weight_change returns a dict containing 'total_change'."""
        model = _tiny_model()
        new_model = copy.deepcopy(model)
        changes = measure_weight_change(model, new_model)

        assert isinstance(changes, dict), "Must return a dict"
        assert "total_change" in changes, "Dict must contain 'total_change'"
        assert "fraction_changed" in changes, "Dict must contain 'fraction_changed'"

    def test_total_change_positive_after_perturbation(self):
        """Test 12: total_change > 0 after a breadcrumb perturbation run."""
        torch.manual_seed(0)
        model = _tiny_model()
        config = _default_config(sparsity=0.9)
        unlearner = BreadcrumbUnlearner(model, config)
        new_model, _ = unlearner.run()

        changes = measure_weight_change(model, new_model)
        assert changes["total_change"] > 0, (
            f"total_change should be > 0, got {changes['total_change']}"
        )

    def test_total_change_zero_for_identical_models(self):
        """total_change should be 0 when comparing a model to itself."""
        model = _tiny_model()
        same_model = copy.deepcopy(model)
        changes = measure_weight_change(model, same_model)

        assert changes["total_change"] == pytest.approx(0.0, abs=1e-9), (
            "Identical models should have zero total_change"
        )

    def test_per_layer_keys_present(self):
        """Each trainable parameter should have its own entry in the result dict."""
        model = _tiny_model()
        config = _default_config()
        unlearner = BreadcrumbUnlearner(model, config)
        new_model, _ = unlearner.run()

        changes = measure_weight_change(model, new_model)
        expected_names = {n for n, _ in model.named_parameters()}
        # Exclude the special summary keys
        param_keys = {k for k in changes if k not in {"total_change", "fraction_changed"}}
        assert param_keys == expected_names, (
            "measure_weight_change must include an entry per parameter"
        )
