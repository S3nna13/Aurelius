from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.interpretability.activation_patcher import ActivationPatcher
from src.interpretability.causal_tracer import (
    CausalTrace,
    CausalTracer,
    TraceConfig,
    TraceResult,
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


def _layer_names(model: nn.Module) -> list[str]:
    return [n for n, _ in model.named_modules() if n]


def _ids(seq: int = 4, dim: int = 4) -> torch.Tensor:
    return torch.randn(1, seq, dim)


# ---------------------------------------------------------------------------
# TraceConfig
# ---------------------------------------------------------------------------

class TestTraceConfig:
    def test_default_metric(self) -> None:
        cfg = TraceConfig(layers=["0"], token_range=(0, 3))
        assert cfg.metric == "logit_diff"

    def test_custom_metric(self) -> None:
        cfg = TraceConfig(layers=["0"], token_range=(0, 2), metric="prob_diff")
        assert cfg.metric == "prob_diff"

    def test_layers_stored(self) -> None:
        cfg = TraceConfig(layers=["a", "b"], token_range=(0, 1))
        assert cfg.layers == ["a", "b"]


# ---------------------------------------------------------------------------
# CausalTracer construction
# ---------------------------------------------------------------------------

class TestCausalTracerConstruction:
    def test_default_patcher_created(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        assert tracer._patcher is not None

    def test_explicit_patcher_used(self) -> None:
        model = _tiny_model()
        patcher = ActivationPatcher(model)
        tracer = CausalTracer(model, patcher=patcher)
        assert tracer._patcher is patcher


# ---------------------------------------------------------------------------
# CausalTracer.trace
# ---------------------------------------------------------------------------

class TestTrace:
    def test_returns_trace_result(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        layers = _layer_names(model)
        cfg = TraceConfig(layers=layers, token_range=(0, 2))
        result = tracer.trace(_ids(), _ids(), cfg)
        assert isinstance(result, TraceResult)

    def test_invalid_layers_skipped(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        cfg = TraceConfig(layers=["bogus.layer.xyz"], token_range=(0, 2))
        result = tracer.trace(_ids(), _ids(), cfg)
        assert result.traces == []

    def test_trace_count_equals_layers_times_tokens(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        layers = _layer_names(model)
        cfg = TraceConfig(layers=layers, token_range=(0, 3))
        result = tracer.trace(_ids(), _ids(), cfg)
        assert len(result.traces) == len(layers) * 3

    def test_total_effect_matches_sum(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        layers = _layer_names(model)
        cfg = TraceConfig(layers=layers, token_range=(0, 2))
        result = tracer.trace(_ids(), _ids(), cfg)
        expected = sum(t.effect for t in result.traces)
        assert abs(result.total_effect - expected) < 1e-6

    def test_top_layers_is_sorted_descending(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        layers = _layer_names(model)
        cfg = TraceConfig(layers=layers, token_range=(0, 3))
        result = tracer.trace(_ids(), _ids(), cfg)
        means = {}
        for t in result.traces:
            means.setdefault(t.layer_name, []).append(t.effect)
        layer_means = {k: sum(v) / len(v) for k, v in means.items()}
        expected_order = sorted(layer_means, key=lambda k: layer_means[k], reverse=True)
        assert result.top_layers == expected_order

    def test_effects_between_0_and_1(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        layers = _layer_names(model)
        cfg = TraceConfig(layers=layers, token_range=(0, 2))
        result = tracer.trace(_ids(), _ids(), cfg)
        for t in result.traces:
            assert 0.0 <= t.effect <= 1.0

    def test_each_trace_has_layer_and_token(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        layers = _layer_names(model)[:1]
        cfg = TraceConfig(layers=layers, token_range=(0, 2))
        result = tracer.trace(_ids(), _ids(), cfg)
        for t in result.traces:
            assert isinstance(t.layer_name, str)
            assert isinstance(t.token_idx, int)


# ---------------------------------------------------------------------------
# CausalTracer.top_k_layers
# ---------------------------------------------------------------------------

class TestTopKLayers:
    def test_returns_list_of_causal_trace(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        layers = _layer_names(model)
        cfg = TraceConfig(layers=layers, token_range=(0, 3))
        result = tracer.trace(_ids(), _ids(), cfg)
        top = tracer.top_k_layers(result, k=2)
        assert all(isinstance(t, CausalTrace) for t in top)

    def test_length_capped_at_k(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        layers = _layer_names(model)
        cfg = TraceConfig(layers=layers, token_range=(0, 3))
        result = tracer.trace(_ids(), _ids(), cfg)
        assert len(tracer.top_k_layers(result, k=2)) <= 2

    def test_sorted_descending_by_effect(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        layers = _layer_names(model)
        cfg = TraceConfig(layers=layers, token_range=(0, 4))
        result = tracer.trace(_ids(), _ids(), cfg)
        top = tracer.top_k_layers(result, k=5)
        effects = [t.effect for t in top]
        assert effects == sorted(effects, reverse=True)


# ---------------------------------------------------------------------------
# CausalTracer.heatmap_data
# ---------------------------------------------------------------------------

class TestHeatmapData:
    def test_returns_nested_dict(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        layers = _layer_names(model)
        cfg = TraceConfig(layers=layers, token_range=(0, 2))
        result = tracer.trace(_ids(), _ids(), cfg)
        hm = tracer.heatmap_data(result)
        assert isinstance(hm, dict)
        for v in hm.values():
            assert isinstance(v, dict)

    def test_all_layers_present(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        layers = _layer_names(model)
        cfg = TraceConfig(layers=layers, token_range=(0, 2))
        result = tracer.trace(_ids(), _ids(), cfg)
        hm = tracer.heatmap_data(result)
        for t in result.traces:
            assert t.layer_name in hm
            assert t.token_idx in hm[t.layer_name]

    def test_values_are_floats(self) -> None:
        model = _tiny_model()
        tracer = CausalTracer(model)
        layers = _layer_names(model)
        cfg = TraceConfig(layers=layers, token_range=(0, 2))
        result = tracer.trace(_ids(), _ids(), cfg)
        hm = tracer.heatmap_data(result)
        for layer_dict in hm.values():
            for v in layer_dict.values():
                assert isinstance(v, float)
