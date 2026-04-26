from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RewardSoupConfig:
    n_models: int = 3
    aggregation: str = "mean"
    weights: list[float] | None = None


class RewardSoupEnsemble:
    """Ensemble of reward models via model weight interpolation (Reward Soup)."""

    def __init__(self, config: RewardSoupConfig) -> None:
        self.config = config
        self._score_buffer: list[list[float]] = []

    def add_model_scores(self, scores: list[float]) -> None:
        self._score_buffer.append(list(scores))

    def aggregate(self) -> list[float]:
        if not self._score_buffer:
            return []

        n_models = len(self._score_buffer)
        n_samples = len(self._score_buffer[0])
        mode = self.config.aggregation

        if mode == "mean":
            result = [
                sum(self._score_buffer[m][s] for m in range(n_models)) / n_models
                for s in range(n_samples)
            ]
        elif mode == "weighted":
            weights = self.config.weights
            if weights is None:
                weights = [1.0 / n_models] * n_models
            result = [
                sum(weights[m] * self._score_buffer[m][s] for m in range(n_models))
                for s in range(n_samples)
            ]
        elif mode == "min":
            result = [
                min(self._score_buffer[m][s] for m in range(n_models)) for s in range(n_samples)
            ]
        elif mode == "max":
            result = [
                max(self._score_buffer[m][s] for m in range(n_models)) for s in range(n_samples)
            ]
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")

        self._score_buffer = []
        return result

    def interpolate_weights(
        self,
        state_dicts: list[dict[str, torch.Tensor]],
        weights: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        if not state_dicts:
            raise ValueError("state_dicts must not be empty")

        n = len(state_dicts)
        if weights is None:
            weights = [1.0 / n] * n

        merged: dict[str, torch.Tensor] = {}
        for key in state_dicts[0]:
            merged[key] = sum(weights[i] * state_dicts[i][key].float() for i in range(n))
        return merged

    def score_batch(
        self,
        scores_per_model: list[list[float]],
    ) -> list[float]:
        for model_scores in scores_per_model:
            self.add_model_scores(model_scores)
        return self.aggregate()
