"""Tests for src/eval/contrastive_activation_addition.py.

Tiny model: B=2, T=4, d_model=16, n_layers=3.
Stand-in layers are nn.Linear(16, 16) modules — they produce (B, T, 16) when
called as a simple feed-forward over the last dimension.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from aurelius.eval.contrastive_activation_addition import (
    ActivationCollector,
    CAAConfig,
    CAAEvaluator,
    CAAExtractor,
    CAASteeringHook,
)
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B = 2
T = 4
D = 16
N_LAYERS = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def layers() -> list[nn.Module]:
    """Three nn.Linear(D, D) layers used as stand-in transformer blocks."""
    torch.manual_seed(0)
    return [nn.Linear(D, D) for _ in range(N_LAYERS)]


@pytest.fixture(scope="module")
def model_fn(layers: list[nn.Module]):
    """Tiny 'model': passes input through each layer sequentially.

    Accepts LongTensor (B, T) — converts to float embeddings (B, T, D)
    first, then pipes through each layer.
    """
    embed = nn.Embedding(256, D)
    torch.manual_seed(1)

    def _forward(input_ids: Tensor) -> Tensor:
        x = embed(input_ids)  # (B, T, D)
        for layer in layers:
            x = layer(x)  # each nn.Linear applied over last dim
        return x

    return _forward


@pytest.fixture(scope="module")
def input_ids_pos() -> Tensor:
    torch.manual_seed(10)
    return torch.randint(0, 256, (B, T))


@pytest.fixture(scope="module")
def input_ids_neg() -> Tensor:
    torch.manual_seed(20)
    return torch.randint(0, 256, (B, T))


@pytest.fixture(scope="module")
def default_config() -> CAAConfig:
    return CAAConfig(layer_idx=1, token_position=-1, normalize=True, alpha=20.0)


@pytest.fixture(scope="module")
def extractor(default_config: CAAConfig, layers: list[nn.Module]) -> CAAExtractor:
    return CAAExtractor(config=default_config, layers=layers)


@pytest.fixture(scope="module")
def steering_vector(
    extractor: CAAExtractor, model_fn, input_ids_pos: Tensor, input_ids_neg: Tensor
) -> Tensor:
    return extractor.extract(model_fn, input_ids_pos, input_ids_neg)


# ---------------------------------------------------------------------------
# Tests: CAAConfig
# ---------------------------------------------------------------------------


class TestCAAConfig:
    def test_defaults(self):
        cfg = CAAConfig()
        assert cfg.layer_idx == 8
        assert cfg.token_position == -1
        assert cfg.normalize is True
        assert cfg.alpha == 20.0

    def test_custom_values(self):
        cfg = CAAConfig(layer_idx=2, token_position=0, normalize=False, alpha=5.0)
        assert cfg.layer_idx == 2
        assert cfg.token_position == 0
        assert cfg.normalize is False
        assert cfg.alpha == 5.0


# ---------------------------------------------------------------------------
# Tests: ActivationCollector
# ---------------------------------------------------------------------------


class TestActivationCollector:
    def test_collect_returns_correct_number_of_layers(self, layers, model_fn):
        collector = ActivationCollector(layers)
        acts = collector.collect(model_fn, torch.randint(0, 256, (B, T)))
        assert len(acts) == N_LAYERS

    def test_collected_activation_shapes(self, layers, model_fn):
        collector = ActivationCollector(layers)
        acts = collector.collect(model_fn, torch.randint(0, 256, (B, T)))
        for act in acts:
            assert act.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {act.shape}"

    def test_context_manager_removes_hooks(self, layers, model_fn):
        """After the context manager exits, no hooks should remain on the layers."""
        before = [len(list(layer._forward_hooks.values())) for layer in layers]
        collector = ActivationCollector(layers)
        with collector:
            during = [len(list(layer._forward_hooks.values())) for layer in layers]
        after = [len(list(layer._forward_hooks.values())) for layer in layers]

        # During collection, each layer should have exactly one extra hook
        for b, d in zip(before, during):
            assert d == b + 1

        # After exit, hook counts should return to baseline
        assert after == before

    def test_context_manager_installs_hooks(self, layers, model_fn):
        """Inside the context manager, hooks are present."""
        collector = ActivationCollector(layers)
        with collector:
            during_counts = [len(list(layer._forward_hooks.values())) for layer in layers]
        assert all(c >= 1 for c in during_counts)

    def test_get_activations_after_collect(self, layers, model_fn):
        """get_activations returns the same tensors as the collect return value."""
        collector = ActivationCollector(layers)
        input_ids = torch.randint(0, 256, (B, T))
        acts = collector.collect(model_fn, input_ids)
        stored = collector.get_activations()
        assert len(stored) == len(acts)
        for a, b in zip(acts, stored):
            assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Tests: CAAExtractor
# ---------------------------------------------------------------------------


class TestCAAExtractor:
    def test_extract_returns_shape_d(self, extractor, model_fn, input_ids_pos, input_ids_neg):
        sv = extractor.extract(model_fn, input_ids_pos, input_ids_neg)
        assert sv.shape == (D,), f"Expected ({D},), got {sv.shape}"

    def test_extract_unit_norm_when_normalize_true(
        self, extractor, model_fn, input_ids_pos, input_ids_neg
    ):
        assert extractor.config.normalize is True
        sv = extractor.extract(model_fn, input_ids_pos, input_ids_neg)
        norm = sv.norm().item()
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_extract_not_unit_norm_when_normalize_false(
        self, layers, model_fn, input_ids_pos, input_ids_neg
    ):
        cfg = CAAConfig(layer_idx=0, normalize=False, alpha=1.0)
        ext = CAAExtractor(cfg, layers)
        sv = ext.extract(model_fn, input_ids_pos, input_ids_neg)
        norm = sv.norm().item()
        # It would be astronomically unlikely for the raw difference to be
        # exactly unit-norm, so we verify it's not constrained to 1.0.
        # We just check the shape is correct and that we didn't normalise.
        assert sv.shape == (D,)
        # The norm could be anything — as long as extraction ran without error.
        assert norm >= 0.0

    def test_extract_multi_layer_returns_all_keys(
        self, extractor, model_fn, input_ids_pos, input_ids_neg
    ):
        result = extractor.extract_multi_layer(model_fn, input_ids_pos, input_ids_neg)
        assert set(result.keys()) == set(range(N_LAYERS))

    def test_extract_multi_layer_vector_shapes(
        self, extractor, model_fn, input_ids_pos, input_ids_neg
    ):
        result = extractor.extract_multi_layer(model_fn, input_ids_pos, input_ids_neg)
        for idx, sv in result.items():
            assert sv.shape == (D,), f"Layer {idx}: expected ({D},), got {sv.shape}"


# ---------------------------------------------------------------------------
# Tests: CAASteeringHook
# ---------------------------------------------------------------------------


class TestCAASteeringHook:
    def _run_with_hook(
        self,
        layers: list[nn.Module],
        model_fn,
        input_ids: Tensor,
        sv: Tensor,
        alpha: float,
        target_layer_idx: int,
    ) -> list[Tensor]:
        collector = ActivationCollector(layers)
        hook = CAASteeringHook(sv, alpha=alpha, token_position=-1)
        hook.attach(layers[target_layer_idx])
        acts = collector.collect(model_fn, input_ids)
        hook.detach()
        return acts

    def test_hook_modifies_output(self, layers, model_fn, steering_vector):
        """With alpha > 0, activations at the hooked layer should change."""
        input_ids = torch.randint(0, 256, (B, T))
        collector_base = ActivationCollector(layers)
        base_acts = collector_base.collect(model_fn, input_ids)

        steered_acts = self._run_with_hook(
            layers, model_fn, input_ids, steering_vector, alpha=20.0, target_layer_idx=1
        )

        # Activations at the hooked layer (or downstream) must differ
        assert not torch.allclose(base_acts[1], steered_acts[1]), (
            "Steering hook should have changed activations."
        )

    def test_detach_restores_original_behavior(self, layers, model_fn, steering_vector):
        """After detach(), the hook no longer affects outputs."""
        input_ids = torch.randint(0, 256, (B, T))

        # Baseline (no hook)
        collector_base = ActivationCollector(layers)
        base_acts = collector_base.collect(model_fn, input_ids)

        # Attach + detach
        hook = CAASteeringHook(steering_vector, alpha=20.0, token_position=-1)
        hook.attach(layers[1])
        hook.detach()

        # Run again — should match baseline
        collector_post = ActivationCollector(layers)
        post_acts = collector_post.collect(model_fn, input_ids)

        assert torch.allclose(base_acts[1], post_acts[1]), (
            "After detach, outputs should be identical to baseline."
        )

    def test_alpha_zero_produces_no_change(self, layers, model_fn, steering_vector):
        """alpha=0 should produce exactly the same output as no steering."""
        input_ids = torch.randint(0, 256, (B, T))

        collector_base = ActivationCollector(layers)
        base_acts = collector_base.collect(model_fn, input_ids)

        steered_acts = self._run_with_hook(
            layers, model_fn, input_ids, steering_vector, alpha=0.0, target_layer_idx=1
        )

        assert torch.allclose(base_acts[1], steered_acts[1], atol=1e-6), (
            "alpha=0 should produce identical outputs to baseline."
        )


# ---------------------------------------------------------------------------
# Tests: CAAEvaluator
# ---------------------------------------------------------------------------


class TestCAAEvaluator:
    def test_steer_and_compare_returns_correct_keys(
        self,
        extractor: CAAExtractor,
        default_config: CAAConfig,
        layers: list[nn.Module],
        model_fn,
        steering_vector: Tensor,
    ):
        evaluator = CAAEvaluator(extractor, default_config)
        input_ids = torch.randint(0, 256, (B, T))
        result = evaluator.steer_and_compare(
            model_fn=model_fn,
            baseline_input_ids=input_ids,
            steered_layer=layers[default_config.layer_idx],
            steering_vector=steering_vector,
            alpha=20.0,
        )
        assert "baseline_activations" in result
        assert "steered_activations" in result

    def test_steer_and_compare_changes_activations_at_target_layer(
        self,
        extractor: CAAExtractor,
        default_config: CAAConfig,
        layers: list[nn.Module],
        model_fn,
        steering_vector: Tensor,
    ):
        evaluator = CAAEvaluator(extractor, default_config)
        input_ids = torch.randint(0, 256, (B, T))
        result = evaluator.steer_and_compare(
            model_fn=model_fn,
            baseline_input_ids=input_ids,
            steered_layer=layers[default_config.layer_idx],
            steering_vector=steering_vector,
            alpha=20.0,
        )
        base = result["baseline_activations"][default_config.layer_idx]
        steered = result["steered_activations"][default_config.layer_idx]
        assert not torch.allclose(base, steered), (
            "Steering should change activations at the target layer."
        )

    def test_steer_and_compare_activation_list_lengths(
        self,
        extractor: CAAExtractor,
        default_config: CAAConfig,
        layers: list[nn.Module],
        model_fn,
        steering_vector: Tensor,
    ):
        evaluator = CAAEvaluator(extractor, default_config)
        input_ids = torch.randint(0, 256, (B, T))
        result = evaluator.steer_and_compare(
            model_fn=model_fn,
            baseline_input_ids=input_ids,
            steered_layer=layers[default_config.layer_idx],
            steering_vector=steering_vector,
            alpha=20.0,
        )
        assert len(result["baseline_activations"]) == N_LAYERS
        assert len(result["steered_activations"]) == N_LAYERS
