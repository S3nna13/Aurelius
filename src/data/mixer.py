"""Online data mixing: blend multiple DataLoaders with configurable weights."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class MixerConfig:
    weights: list[float]  # mixing weight for each source (will be normalized)
    temperature: float = 1.0  # temperature scaling: w_i -> w_i^(1/T) before normalizing
    # T < 1: sharper (more concentrated on high-weight sources)
    # T > 1: flatter (more uniform)


class DataMixer:
    """Infinite iterator that samples from multiple DataLoaders with mixing weights.

    Each call to __next__() samples one source according to mixing weights,
    then returns a batch from that source's iterator.

    When a source is exhausted, its iterator is reset (restarts from beginning).

    Usage:
        mixer = DataMixer([loader_a, loader_b], MixerConfig(weights=[0.7, 0.3]))
        for i, batch in enumerate(mixer):
            if i >= total_steps:
                break
            train(batch)
    """

    def __init__(self, loaders: list[DataLoader], cfg: MixerConfig) -> None:
        if len(loaders) != len(cfg.weights):
            raise ValueError(
                f"Number of loaders ({len(loaders)}) must match "
                f"number of weights ({len(cfg.weights)})"
            )
        self.loaders = loaders
        self.cfg = cfg
        self._iters: list[Iterator] = [iter(loader) for loader in loaders]
        self._probs = self._compute_probs(cfg.weights)

    def _compute_probs(self, weights: list[float]) -> list[float]:
        """Apply temperature scaling and normalize weights to probabilities."""
        w = torch.tensor(weights, dtype=torch.float)
        w = w ** (1.0 / max(self.cfg.temperature, 1e-8))
        w = w / w.sum()
        return w.tolist()

    def __iter__(self) -> DataMixer:
        return self

    def __next__(self):
        """Sample a source and return its next batch."""
        # Sample source index according to mixing probabilities
        source_idx = torch.multinomial(torch.tensor(self._probs), 1).item()

        # Try to get batch, reset iterator if exhausted
        try:
            batch = next(self._iters[source_idx])
        except StopIteration:
            self._iters[source_idx] = iter(self.loaders[source_idx])
            batch = next(self._iters[source_idx])

        return batch

    def update_weights(self, new_weights: list[float]) -> None:
        """Update mixing weights dynamically (e.g., based on per-source loss)."""
        self._probs = self._compute_probs(new_weights)

    @property
    def current_weights(self) -> list[float]:
        """Return current normalized mixing probabilities."""
        return list(self._probs)


class LossAdaptiveMixer(DataMixer):
    """DataMixer that adjusts weights based on per-source training loss.

    Sources with higher loss get upweighted (focus on harder data).
    Uses exponential moving average of per-source loss.
    """

    def __init__(
        self,
        loaders: list[DataLoader],
        cfg: MixerConfig,
        ema_alpha: float = 0.1,
    ) -> None:
        super().__init__(loaders, cfg)
        self.ema_alpha = ema_alpha
        self._ema_losses: list[float] = [1.0] * len(loaders)

    def record_loss(self, source_idx: int, loss: float) -> None:
        """Update EMA loss for source_idx and reweight accordingly."""
        self._ema_losses[source_idx] = (
            self.ema_alpha * loss + (1 - self.ema_alpha) * self._ema_losses[source_idx]
        )
        # Reweight: higher loss -> higher weight
        self.update_weights(self._ema_losses)
