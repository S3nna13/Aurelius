from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch


ModelDelta = dict[str, torch.Tensor]


def aggregate_with_clipping(deltas: list[ModelDelta], clip_norm: float = 1.0) -> ModelDelta:
    if not deltas:
        return {}
    clipped: list[ModelDelta] = []
    for delta in deltas:
        total_norm = sum(d.norm().item() ** 2 for d in delta.values()) ** 0.5
        scale = min(1.0, clip_norm / max(total_norm, 1e-8))
        clipped.append({k: v * scale for k, v in delta.items()})

    result: ModelDelta = {}
    for key in clipped[0]:
        stacked = torch.stack([c[key] for c in clipped])
        result[key] = stacked.mean(dim=0)
    return result


def add_clip_noise(x: torch.Tensor, noise_scale: float = 0.0, seed: int | None = None) -> torch.Tensor:
    if noise_scale <= 0.0:
        return x.clone()
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(x) * noise_scale
    return x + noise


@dataclass
class SecureAggregatorV2:
    min_clients: int = 2
    clip_norm: float = 1.0
    noise_scale: float = 0.0
    _total_aggregations: int = 0
    _skipped: int = 0

    def aggregate(self, deltas: list[ModelDelta]) -> ModelDelta | None:
        if len(deltas) < self.min_clients:
            self._skipped += 1
            return None
        result = aggregate_with_clipping(deltas, clip_norm=self.clip_norm)
        if self.noise_scale > 0.0:
            result = {k: add_clip_noise(v, self.noise_scale) for k, v in result.items()}
        self._total_aggregations += 1
        return result

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_aggregations": self._total_aggregations,
            "skipped_low_clients": self._skipped,
            "min_clients": self.min_clients,
        }


SECURE_AGGREGATOR_V2 = SecureAggregatorV2()
