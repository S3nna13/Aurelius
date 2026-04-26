from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from src.training.lora_adapter_manager import LoRAAdapterManager, LoRALayer


class CompositionMode(StrEnum):
    ADD = "add"
    WEIGHTED = "weighted"
    SEQUENTIAL = "sequential"


@dataclass
class ComposedAdapter:
    names: list[str]
    weights: list[float]
    mode: CompositionMode


class AdapterComposer:
    """Compose multiple LoRA adapters via addition, weighting, or sequential application."""

    def __init__(self, manager: LoRAAdapterManager) -> None:
        self.manager = manager

    def compose(
        self,
        adapter_names: list[str],
        weights: list[float] | None = None,
        mode: CompositionMode = CompositionMode.WEIGHTED,
    ) -> ComposedAdapter:
        for n in adapter_names:
            if self.manager.get_adapter(n) is None:
                raise ValueError(f"Adapter '{n}' not found")

        if weights is None:
            n = len(adapter_names)
            weights = [1.0 / n] * n
        elif mode == CompositionMode.WEIGHTED:
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]

        return ComposedAdapter(names=adapter_names, weights=list(weights), mode=mode)

    def apply(
        self,
        x: torch.Tensor,
        composed: ComposedAdapter,
        layer_name: str,
    ) -> torch.Tensor:
        if composed.mode == CompositionMode.ADD:
            delta = torch.zeros_like(x)
            for name in composed.names:
                adapter = self.manager.get_adapter(name)
                if adapter is None or layer_name not in adapter:
                    continue
                layer: LoRALayer = adapter[layer_name]
                delta = delta + (layer(x) - x)
            return x + delta

        if composed.mode == CompositionMode.WEIGHTED:
            delta = torch.zeros_like(x)
            for name, weight in zip(composed.names, composed.weights):
                adapter = self.manager.get_adapter(name)
                if adapter is None or layer_name not in adapter:
                    continue
                layer = adapter[layer_name]
                delta = delta + weight * (layer(x) - x)
            return x + delta

        # SEQUENTIAL: each adapter's forward applied in order
        out = x
        for name in composed.names:
            adapter = self.manager.get_adapter(name)
            if adapter is None or layer_name not in adapter:
                continue
            layer = adapter[layer_name]
            out = layer(out)
        return out

    def merge_to_delta(
        self,
        composed: ComposedAdapter,
        layer_name: str,
    ) -> torch.Tensor | None:
        delta: torch.Tensor | None = None
        for name, weight in zip(composed.names, composed.weights):
            adapter = self.manager.get_adapter(name)
            if adapter is None or layer_name not in adapter:
                continue
            layer_delta = adapter[layer_name].merge()
            weighted = weight * layer_delta
            delta = weighted if delta is None else delta + weighted
        return delta
