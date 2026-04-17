"""Tests for aurelius.training.gradient_checkpointing_v2."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from aurelius.training.gradient_checkpointing_v2 import (
    CheckpointConfig,
    CheckpointedLayer,
    CheckpointingBenchmark,
    MemoryStats,
    SelectiveCheckpointing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_linear(in_features: int = 8, out_features: int = 8) -> nn.Linear:
    """Return a small deterministic Linear layer."""
    layer = nn.Linear(in_features, out_features, bias=False)
    nn.init.eye_(layer.weight)
    return layer


def _make_input(batch: int = 2, features: int = 8) -> torch.Tensor:
    return torch.randn(batch, features, requires_grad=True)


# ---------------------------------------------------------------------------
# CheckpointConfig defaults
# ---------------------------------------------------------------------------


class TestCheckpointConfigDefaults:
    def test_defaults(self):
        cfg = CheckpointConfig()
        assert cfg.enabled is True
        assert cfg.checkpoint_every == 1
        assert cfg.use_reentrant is False

    def test_custom_values(self):
        cfg = CheckpointConfig(enabled=False, checkpoint_every=4, use_reentrant=True)
        assert cfg.enabled is False
        assert cfg.checkpoint_every == 4
        assert cfg.use_reentrant is True


# ---------------------------------------------------------------------------
# CheckpointedLayer — output correctness
# ---------------------------------------------------------------------------


class TestCheckpointedLayerOutput:
    def test_same_output_as_unwrapped(self):
        """Checkpointed forward must produce the same result as a direct call."""
        layer = _simple_linear()
        config = CheckpointConfig(enabled=True)
        cl = CheckpointedLayer(layer, config)
        cl.train()

        x = _make_input()
        with torch.no_grad():
            expected = layer(x)
            actual = cl(x)

        assert torch.allclose(actual, expected), "Checkpointed output differs"

    def test_wrapped_property_returns_inner_module(self):
        layer = _simple_linear()
        cl = CheckpointedLayer(layer, CheckpointConfig())
        assert cl.wrapped is layer


# ---------------------------------------------------------------------------
# CheckpointedLayer — gradient flow
# ---------------------------------------------------------------------------


class TestCheckpointedLayerGradients:
    def test_gradients_flow_through_checkpointed_layer(self):
        """Backward pass must produce gradients when checkpointing is enabled."""
        layer = _simple_linear()
        cl = CheckpointedLayer(layer, CheckpointConfig(enabled=True))
        cl.train()

        x = torch.randn(2, 8, requires_grad=True)
        out = cl(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient reached the input tensor"
        assert not torch.all(x.grad == 0), "Input gradient is all zeros"

    def test_weight_gradients_populated(self):
        """Layer parameters must also receive gradients."""
        layer = _simple_linear()
        cl = CheckpointedLayer(layer, CheckpointConfig(enabled=True))
        cl.train()

        x = torch.randn(2, 8)
        out = cl(x)
        out.sum().backward()

        assert layer.weight.grad is not None
        assert not torch.all(layer.weight.grad == 0)


# ---------------------------------------------------------------------------
# CheckpointedLayer — disabled behaviour
# ---------------------------------------------------------------------------


class TestCheckpointedLayerDisabled:
    def test_disabled_config_passes_through(self):
        """With enabled=False the module is called directly."""
        layer = _simple_linear()
        cl = CheckpointedLayer(layer, CheckpointConfig(enabled=False))
        cl.train()

        x = _make_input()
        with torch.no_grad():
            out = cl(x)
            expected = layer(x)

        assert torch.allclose(out, expected)

    def test_inference_mode_passes_through(self):
        """In inference (non-training) mode the module should be called directly."""
        layer = _simple_linear()
        cl = CheckpointedLayer(layer, CheckpointConfig(enabled=True))
        cl.train(False)  # switch to non-training mode

        x = _make_input()
        with torch.no_grad():
            out = cl(x)
            expected = layer(x)

        assert torch.allclose(out, expected)

    def test_non_training_mode_skips_checkpoint(self):
        """Confirm the non-training branch is taken via identical output."""
        layer = nn.Linear(4, 4)
        cl = CheckpointedLayer(layer, CheckpointConfig(enabled=True))
        cl.train(False)

        x = torch.randn(3, 4)
        with torch.no_grad():
            assert torch.allclose(cl(x), layer(x))


# ---------------------------------------------------------------------------
# SelectiveCheckpointing
# ---------------------------------------------------------------------------


class TestSelectiveCheckpointing:
    def test_wrap_returns_all_layers(self):
        layers = [_simple_linear() for _ in range(6)]
        sc = SelectiveCheckpointing(layers, CheckpointConfig(checkpoint_every=1))
        wrapped = sc.wrap()
        assert len(wrapped) == 6

    def test_wrap_returns_checkpointed_layer_instances(self):
        layers = [_simple_linear() for _ in range(4)]
        sc = SelectiveCheckpointing(layers, CheckpointConfig())
        wrapped = sc.wrap()
        assert all(isinstance(w, CheckpointedLayer) for w in wrapped)

    def test_checkpoint_every_2_wraps_correct_layers(self):
        """With checkpoint_every=2 only even-indexed layers should be checkpointed."""
        layers = [_simple_linear() for _ in range(6)]
        sc = SelectiveCheckpointing(layers, CheckpointConfig(checkpoint_every=2))
        wrapped = sc.wrap()

        for i, cl in enumerate(wrapped):
            if i % 2 == 0:
                assert cl.config.enabled, f"Layer {i} should be checkpointed"
            else:
                assert not cl.config.enabled, f"Layer {i} should NOT be checkpointed"

    def test_unwrap_returns_original_modules(self):
        layers = [_simple_linear() for _ in range(4)]
        sc = SelectiveCheckpointing(layers, CheckpointConfig())
        wrapped = sc.wrap()
        unwrapped = SelectiveCheckpointing.unwrap(wrapped)

        assert len(unwrapped) == len(layers)
        for original, recovered in zip(layers, unwrapped):
            assert original is recovered, "Unwrapped module is not the original object"

    def test_unwrap_length_matches(self):
        layers = [nn.ReLU() for _ in range(5)]
        sc = SelectiveCheckpointing(layers, CheckpointConfig(checkpoint_every=3))
        wrapped = sc.wrap()
        assert len(SelectiveCheckpointing.unwrap(wrapped)) == 5


# ---------------------------------------------------------------------------
# MemoryStats
# ---------------------------------------------------------------------------


class TestMemoryStats:
    def test_has_correct_fields(self):
        ms = MemoryStats(
            peak_allocated_mb=100.0,
            current_allocated_mb=50.0,
            reserved_mb=200.0,
        )
        assert ms.peak_allocated_mb == 100.0
        assert ms.current_allocated_mb == 50.0
        assert ms.reserved_mb == 200.0

    def test_capture_returns_memory_stats_instance(self):
        ms = MemoryStats.capture()
        assert isinstance(ms, MemoryStats)

    def test_capture_returns_non_negative_values(self):
        ms = MemoryStats.capture()
        assert ms.peak_allocated_mb >= 0.0
        assert ms.current_allocated_mb >= 0.0
        assert ms.reserved_mb >= 0.0


# ---------------------------------------------------------------------------
# CheckpointingBenchmark
# ---------------------------------------------------------------------------


class TestCheckpointingBenchmark:
    def _make_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
        )

    def test_estimate_savings_every_2(self):
        model = self._make_model()
        bench = CheckpointingBenchmark(model, CheckpointConfig())
        savings = bench.estimate_savings(original_layers=4, checkpointed_every=2)
        assert abs(savings - 0.5) < 1e-9, f"Expected 0.5, got {savings}"

    def test_estimate_savings_every_4(self):
        model = self._make_model()
        bench = CheckpointingBenchmark(model, CheckpointConfig())
        savings = bench.estimate_savings(original_layers=8, checkpointed_every=4)
        assert abs(savings - 0.75) < 1e-9, f"Expected 0.75, got {savings}"

    def test_estimate_savings_every_1(self):
        """checkpoint_every=1 — all layers recompute, no net savings."""
        model = self._make_model()
        bench = CheckpointingBenchmark(model, CheckpointConfig())
        savings = bench.estimate_savings(original_layers=4, checkpointed_every=1)
        assert abs(savings - 0.0) < 1e-9

    def test_run_forward_returns_expected_keys(self):
        model = self._make_model()
        bench = CheckpointingBenchmark(model, CheckpointConfig(enabled=False))
        x = torch.randn(2, 8)
        result = bench.run_forward(x, n_iterations=2)
        assert "forward_ms" in result
        assert "backward_ms" in result
        assert "peak_memory_mb" in result

    def test_run_forward_returns_positive_times(self):
        model = self._make_model()
        bench = CheckpointingBenchmark(model, CheckpointConfig(enabled=False))
        x = torch.randn(2, 8)
        result = bench.run_forward(x, n_iterations=2)
        assert result["forward_ms"] >= 0.0
        assert result["backward_ms"] >= 0.0
