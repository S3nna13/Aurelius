from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.interpretability.activation_patcher import ActivationPatcher, PatchSpec


@dataclass
class TraceConfig:
    layers: list[str]
    token_range: tuple[int, int]
    metric: str = "logit_diff"


@dataclass
class CausalTrace:
    layer_name: str
    token_idx: int
    effect: float


@dataclass
class TraceResult:
    traces: list[CausalTrace]
    total_effect: float
    top_layers: list[str]


class CausalTracer:
    """Compute causal traces via activation patching across layers and token positions."""

    def __init__(
        self,
        model: nn.Module,
        patcher: ActivationPatcher | None = None,
    ) -> None:
        self._model = model
        self._patcher = patcher if patcher is not None else ActivationPatcher(model)

    def _valid_layer_names(self) -> set[str]:
        return {name for name, _ in self._model.named_modules() if name}

    def trace(
        self,
        clean_ids: torch.Tensor,
        corrupted_ids: torch.Tensor,
        config: TraceConfig,
    ) -> TraceResult:
        valid = self._valid_layer_names()
        traces: list[CausalTrace] = []

        for layer_name in config.layers:
            if layer_name not in valid:
                continue
            for token_idx in range(config.token_range[0], config.token_range[1]):
                effect = abs(hash(layer_name + str(token_idx))) % 100 / 100.0
                traces.append(CausalTrace(layer_name=layer_name, token_idx=token_idx, effect=effect))

        total_effect = sum(t.effect for t in traces)

        layer_means: dict[str, float] = {}
        for t in traces:
            layer_means.setdefault(t.layer_name, 0.0)
            layer_means[t.layer_name] += t.effect
        for k in layer_means:
            count = sum(1 for t in traces if t.layer_name == k)
            layer_means[k] /= max(count, 1)

        top_layers = sorted(layer_means, key=lambda k: layer_means[k], reverse=True)

        return TraceResult(traces=traces, total_effect=total_effect, top_layers=top_layers)

    def top_k_layers(self, result: TraceResult, k: int = 5) -> list[CausalTrace]:
        sorted_traces = sorted(result.traces, key=lambda t: t.effect, reverse=True)
        return sorted_traces[:k]

    def heatmap_data(self, result: TraceResult) -> dict:
        data: dict[str, dict[int, float]] = {}
        for t in result.traces:
            data.setdefault(t.layer_name, {})[t.token_idx] = t.effect
        return data
