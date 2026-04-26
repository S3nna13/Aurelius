from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.interpretability.activation_patcher import (
    ActivationPatcher,
    PatchResult,
    PatchSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )


def _input_ids(batch: int = 1, seq: int = 3, dim: int = 4) -> torch.Tensor:
    return torch.randn(batch, seq, dim)


def _layer_name(model: nn.Module, idx: int) -> str:
    names = [n for n, _ in model.named_modules() if n]
    return names[idx]


# ---------------------------------------------------------------------------
# PatchSpec
# ---------------------------------------------------------------------------


class TestPatchSpec:
    def test_default_source_value_is_none(self) -> None:
        spec = PatchSpec(layer_name="0", token_idx=1)
        assert spec.source_value is None

    def test_custom_source_value(self) -> None:
        val = torch.ones(4)
        spec = PatchSpec(layer_name="0", token_idx=0, source_value=val)
        assert spec.source_value is val

    def test_layer_name_stored(self) -> None:
        spec = PatchSpec(layer_name="transformer.h.3.mlp", token_idx=2)
        assert spec.layer_name == "transformer.h.3.mlp"

    def test_token_idx_stored(self) -> None:
        spec = PatchSpec(layer_name="0", token_idx=7)
        assert spec.token_idx == 7


# ---------------------------------------------------------------------------
# PatchResult
# ---------------------------------------------------------------------------


class TestPatchResult:
    def test_effect_score_is_float(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids()
        spec = PatchSpec(layer_name=_layer_name(model, 0), token_idx=0)
        result = patcher.patch(x, spec)
        assert isinstance(result.effect_score, float)

    def test_original_logits_shape(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        spec = PatchSpec(layer_name=_layer_name(model, 0), token_idx=0)
        result = patcher.patch(x, spec)
        assert result.original_logits.dim() == 1

    def test_patched_logits_shape_matches_original(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        spec = PatchSpec(layer_name=_layer_name(model, 0), token_idx=0)
        result = patcher.patch(x, spec)
        assert result.patched_logits.shape == result.original_logits.shape

    def test_patch_spec_stored(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        spec = PatchSpec(layer_name=_layer_name(model, 0), token_idx=1)
        result = patcher.patch(x, spec)
        assert result.patch_spec is spec


# ---------------------------------------------------------------------------
# ActivationPatcher.patch
# ---------------------------------------------------------------------------


class TestPatch:
    def test_returns_patch_result(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        spec = PatchSpec(layer_name=_layer_name(model, 0), token_idx=0)
        result = patcher.patch(x, spec)
        assert isinstance(result, PatchResult)

    def test_zero_ablation_changes_output(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        spec = PatchSpec(layer_name=_layer_name(model, 0), token_idx=0)
        result = patcher.patch(x, spec, target_token_idx=0)
        assert not torch.allclose(result.original_logits, result.patched_logits)

    def test_source_value_patch_applied(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        fill = torch.full((8,), 999.0)
        spec = PatchSpec(layer_name=_layer_name(model, 0), token_idx=1, source_value=fill)
        result = patcher.patch(x, spec)
        assert result.effect_score >= 0.0

    def test_invalid_layer_raises(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        spec = PatchSpec(layer_name="nonexistent.layer", token_idx=0)
        with pytest.raises(KeyError):
            patcher.patch(x, spec)

    def test_effect_score_non_negative(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        spec = PatchSpec(layer_name=_layer_name(model, 0), token_idx=0)
        result = patcher.patch(x, spec)
        assert result.effect_score >= 0.0

    def test_target_token_idx_last(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=5)
        spec = PatchSpec(layer_name=_layer_name(model, 0), token_idx=0)
        result = patcher.patch(x, spec, target_token_idx=-1)
        assert result.original_logits.dim() == 1

    def test_no_gradient_computed(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        spec = PatchSpec(layer_name=_layer_name(model, 0), token_idx=0)
        result = patcher.patch(x, spec)
        assert not result.original_logits.requires_grad
        assert not result.patched_logits.requires_grad


# ---------------------------------------------------------------------------
# ActivationPatcher.patch_batch
# ---------------------------------------------------------------------------


class TestPatchBatch:
    def test_returns_list(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        layer = _layer_name(model, 0)
        patches = [PatchSpec(layer_name=layer, token_idx=i) for i in range(2)]
        results = patcher.patch_batch(x, patches)
        assert isinstance(results, list)

    def test_length_matches_patches(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        layer = _layer_name(model, 0)
        patches = [PatchSpec(layer_name=layer, token_idx=i) for i in range(3)]
        results = patcher.patch_batch(x, patches)
        assert len(results) == 3

    def test_each_result_is_patch_result(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        layer = _layer_name(model, 0)
        patches = [PatchSpec(layer_name=layer, token_idx=i) for i in range(2)]
        for r in patcher.patch_batch(x, patches):
            assert isinstance(r, PatchResult)

    def test_empty_patches_returns_empty(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        assert patcher.patch_batch(x, []) == []


# ---------------------------------------------------------------------------
# ActivationPatcher.zero_ablate
# ---------------------------------------------------------------------------


class TestZeroAblate:
    def test_returns_patch_result(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        layer = _layer_name(model, 0)
        result = patcher.zero_ablate(x, layer, token_idx=0)
        assert isinstance(result, PatchResult)

    def test_source_value_is_none_in_spec(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        layer = _layer_name(model, 0)
        result = patcher.zero_ablate(x, layer, token_idx=1)
        assert result.patch_spec.source_value is None

    def test_effect_score_non_negative(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        x = _input_ids(seq=3)
        layer = _layer_name(model, 0)
        result = patcher.zero_ablate(x, layer, token_idx=0)
        assert result.effect_score >= 0.0
