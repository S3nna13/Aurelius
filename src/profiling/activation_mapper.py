from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ActivationStats:
    module_name: str
    shape: tuple[int, ...]
    mean: float
    std: float
    min_val: float
    max_val: float
    has_nan: bool
    has_inf: bool


class ActivationMapper:
    def __init__(self):
        self._stats: list[ActivationStats] = []

    def register_hooks(self, model: nn.Module) -> list:
        handles = []
        for name, module in model.named_modules():
            handle = module.register_forward_hook(self._make_hook(name))
            handles.append(handle)
        return handles

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                return
            t = output.detach().float()
            self._stats.append(
                ActivationStats(
                    module_name=name,
                    shape=tuple(t.shape),
                    mean=float(t.mean()),
                    std=float(t.std()) if t.numel() > 1 else 0.0,
                    min_val=float(t.min()),
                    max_val=float(t.max()),
                    has_nan=bool(torch.isnan(t).any()),
                    has_inf=bool(torch.isinf(t).any()),
                )
            )

        return hook

    def remove_hooks(self, handles: list):
        for handle in handles:
            handle.remove()

    def get_stats(self) -> list[ActivationStats]:
        return list(self._stats)

    def detect_anomalies(self) -> list[str]:
        anomalies = []
        for s in self._stats:
            parts = []
            if s.has_nan:
                parts.append("has NaN")
            if s.has_inf:
                parts.append("has Inf")
            if parts:
                anomalies.append(f"{s.module_name}: {', '.join(parts)}")
        return anomalies

    def clear(self):
        self._stats.clear()


ACTIVATION_MAPPER_REGISTRY: dict[str, type[ActivationMapper]] = {"default": ActivationMapper}
