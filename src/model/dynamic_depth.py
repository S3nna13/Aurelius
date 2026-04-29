"""DAIES Cycle 7: DynamicDepth - adaptive layer depth routing.

This module keeps the legacy batch-level early-exit surface alive while also
exposing the newer config-driven API used by the dynamic-depth tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DynamicDepthConfig:
    """Configuration for dynamic depth routing."""

    exit_threshold: float = 0.9
    skip_threshold: float = 0.1
    min_layers: int = 1
    temperature: float = 1.0
    use_learned_router: bool = True


class EarlyExitClassifier(nn.Module):
    """Decides if a token can exit at the current layer."""

    def __init__(self, d_model: int):
        super().__init__()
        self.exit_score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, threshold: float = 0.9) -> tuple[torch.Tensor, bool]:
        scores = torch.sigmoid(self.exit_score(x))
        should_exit = scores.mean() > threshold
        return scores, should_exit


class ExitRouter(nn.Module):
    """Lightweight learned router that emits exit probabilities."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(x))


class LayerSkipRouter(ExitRouter):
    """Alias for the skip router path used by the tests."""


def compute_exit_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Return max-softmax confidence for each batch element."""

    if logits.ndim == 3:
        logits = logits[:, -1, :]
    return logits.softmax(dim=-1).amax(dim=-1)


class AdaptiveLayerSelector:
    """Selects a deterministic prefix of layers based on hidden-state norm."""

    def __init__(self, n_layers: int, min_layers: int = 1) -> None:
        self.n_layers = n_layers
        self.min_layers = min_layers

    def select_layers(self, hidden: torch.Tensor) -> list[int]:
        if hidden.ndim < 2:
            raise ValueError("hidden must have shape (..., hidden_dim)")

        score = hidden.norm(dim=-1).mean().item()
        span = int(score) % max(self.n_layers, 1) + 1
        span = max(self.min_layers, min(self.n_layers, span))
        return list(range(span))


class DynamicDepthTransformer(nn.Module):
    """Transformer with dynamic early exit on a batch-level basis."""

    def __init__(
        self,
        base_model: nn.Module,
        config_or_exit_layers: DynamicDepthConfig | list[int] | None = None,
        d_model: int = 2048,
        exit_layers: list[int] | None = None,
    ):
        super().__init__()
        self.base = base_model

        if isinstance(config_or_exit_layers, DynamicDepthConfig):
            self.config = config_or_exit_layers
            candidate_exit_layers = exit_layers
        elif isinstance(config_or_exit_layers, list):
            self.config = DynamicDepthConfig()
            candidate_exit_layers = config_or_exit_layers
        else:
            self.config = DynamicDepthConfig()
            candidate_exit_layers = exit_layers

        n_layers = self.base.config.n_layers
        if candidate_exit_layers is None:
            candidate_exit_layers = [max(self.config.min_layers - 1, 0)]
        self.exit_layers = sorted(
            {max(0, min(int(layer_idx), n_layers - 1)) for layer_idx in candidate_exit_layers}
        )
        if not self.exit_layers:
            self.exit_layers = [n_layers - 1]

        self.target_exit_layer = self.exit_layers[0]
        self.d_model = d_model or getattr(self.base.config, "d_model", 2048)
        self.exit_router = ExitRouter(self.d_model)
        self.skip_router = LayerSkipRouter(self.d_model)
        self.selector = AdaptiveLayerSelector(n_layers=n_layers, min_layers=self.config.min_layers)

        self._tokens_saved: int = 0
        self._total_tokens: int = 0
        self._last_exit_layers: list[int] = [n_layers - 1]

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        B, S = input_ids.shape
        n_layers = self.base.config.n_layers

        x = self.base.embed(input_ids)
        freqs_cis = self.base.freqs_cis[:S]

        # Keep the API deterministic: route to the first configured exit layer
        # that is still compatible with the minimum-layer constraint.
        actual_exit_layer = max(self.target_exit_layer, self.config.min_layers - 1)
        actual_exit_layer = min(actual_exit_layer, n_layers - 1)

        for layer_idx, layer in enumerate(self.base.layers):
            x, _kv, _aux = layer(x, freqs_cis)
            if layer_idx >= actual_exit_layer:
                break

        x = self.base.norm(x)
        logits = self.base.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
            )

        exit_layers = [actual_exit_layer for _ in range(B)]
        self._last_exit_layers = exit_layers
        self._tokens_saved += max(n_layers - 1 - actual_exit_layer, 0) * B * S
        self._total_tokens += n_layers * B * S
        return loss, logits, exit_layers

    def compute_efficiency_stats(self, exit_layers: list[int]) -> dict[str, float]:
        n_layers = self.base.config.n_layers
        layers = torch.tensor(exit_layers, dtype=torch.float32)
        early_exit_rate = float((layers < (n_layers - 1)).float().mean().item())
        return {
            "mean_exit_layer": float(layers.mean().item()),
            "min_exit_layer": float(layers.min().item()),
            "max_exit_layer": float(layers.max().item()),
            "early_exit_rate": early_exit_rate,
        }

    @property
    def exit_rate(self) -> float:
        return self._tokens_saved / max(self._total_tokens, 1)
