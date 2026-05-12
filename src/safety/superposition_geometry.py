"""Feature Superposition Geometry for Misalignment Analysis.

Implements the Feature Superposition Hypothesis from arXiv:2605.00842:
fine-tuning amplifies target features but leaks into geometrically adjacent
features in the superposed feature space, unintentionally reinforcing harmful
features.

Gradient-Level Derivation:
    Δh ≈ α·d_insecure causes spillover Δf_j ≈ α·<d_j, d_insecure>
    onto nearby features in the SAE feature space.

Geometry-Aware Data Filtering:
    SAE-based filtering removing training samples closest to toxic features;
    achieves 34.5% misalignment reduction.

Reference: arXiv:2605.00842 "Understanding Emergent Misalignment via Feature Superposition Geometry"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class SuperpositionGeometryConfig:
    """Configuration for feature superposition geometry analysis.

    Attributes:
        sae_window: Number of most active features to consider per activation
        toxic_similarity_threshold: Cosine similarity threshold for toxic features
        misalignment_risk_threshold: Risk threshold for triggering warnings
        spillover_coefficient: Base coefficient for gradient spillover estimation
        filter_remove_ratio: Fraction of training samples to remove (0.0-1.0)
        layer_sample_interval: Sample every N layers for layer-wise analysis
        min_feature_correlation: Minimum correlation to track a feature pair
        accumulation_steps: Steps to accumulate gradient statistics
        device: Compute device ('cuda' or 'cpu')
    """

    sae_window: int = 512
    toxic_similarity_threshold: float = 0.7
    misalignment_risk_threshold: float = 0.5
    spillover_coefficient: float = 0.1
    filter_remove_ratio: float = 0.15
    layer_sample_interval: int = 4
    min_feature_correlation: float = 0.3
    accumulation_steps: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    feature_dim: int | None = None
    num_layers: int | None = None


class FeatureSuperpositionAnalyzer:
    """Analyze feature geometry in transformer representations using SAEs.

    Uses sparse autoencoders to identify features and computes cosine similarity
    between misalignment-inducing features and toxic features to quantify
    superposition risk.

    Attributes:
        config: SuperpositionGeometryConfig instance
        device: Compute device
    """

    def __init__(self, config: SuperpositionGeometryConfig):
        """Initialize the analyzer with configuration."""
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)

        self._toxic_feature_indices: Tensor | None = None
        self._misalignment_feature_indices: Tensor | None = None
        self._feature_directions: Tensor | None = None
        self._layer_similarities: dict[int, Tensor] = {}
        self._similarity_history: list[dict[str, Any]] = []

    def fit(
        self,
        sae_weights: Tensor,
        toxic_feature_indices: Tensor,
        misalignment_feature_indices: Tensor,
    ) -> None:
        """Fit the analyzer with SAE weights and identified feature indices.

        Args:
            sae_weights: (d_model, n_features) SAE decode matrix
            toxic_feature_indices: (n_toxic,) indices of toxic features
            misalignment_feature_indices: (n_misalign,) indices of misalignment features
        """
        self._validate_tensor(sae_weights, "sae_weights", dim=2)
        self._validate_tensor(toxic_feature_indices, "toxic_feature_indices", dim=1)
        self._validate_tensor(misalignment_feature_indices, "misalignment_feature_indices", dim=1)

        self._feature_directions = sae_weights.to(self.device)
        self._toxic_feature_indices = toxic_feature_indices.to(self.device)
        self._misalignment_feature_indices = misalignment_feature_indices.to(self.device)

        self.logger.info(
            "Fitted FeatureSuperpositionAnalyzer with %d features, %d toxic, %d misalignment",
            sae_weights.shape[1],
            len(toxic_feature_indices),
            len(misalignment_feature_indices),
        )

    def compute_feature_similarities(self, layer: int, activations: Tensor | None = None) -> Tensor:
        """Compute cosine similarities between misalignment and toxic features.

        Args:
            layer: Layer index for tracking
            activations: Optional (seq_len, d_model) activations for context

        Returns:
            (n_misalign, n_toxic) cosine similarity matrix
        """
        if self._feature_directions is None:
            raise RuntimeError("Analyzer must be fit before computing similarities")

        toxic_dirs = self._feature_directions[:, self._toxic_feature_indices]
        misalign_dirs = self._feature_directions[:, self._misalignment_feature_indices]

        toxic_norm = toxic_dirs / (toxic_dirs.norm(dim=0, keepdim=True) + 1e-8)
        misalign_norm = misalign_dirs / (misalign_dirs.norm(dim=0, keepdim=True) + 1e-8)

        similarities = torch.mm(misalign_norm.T, toxic_norm)

        self._layer_similarities[layer] = similarities.detach().cpu()

        return similarities

    def compute_layer_wise_toxic_similarity(
        self, layer_activations: list[tuple[int, Tensor]]
    ) -> dict[int, float]:
        """Compute average similarity to toxic features across layers.

        Args:
            layer_activations: List of (layer_idx, activations) tuples

        Returns:
            Dictionary mapping layer_idx to average toxic similarity
        """
        results = {}
        for layer_idx, activations in layer_activations:
            if activations is None or activations.numel() == 0:
                results[layer_idx] = 0.0
                continue

            flat_activations = activations.flatten()
            if self._feature_directions is None:
                continue

            projected = self._feature_directions.T @ flat_activations
            toxic_projection = projected[self._toxic_feature_indices]

            toxic_sim = torch.mean(torch.abs(toxic_projection)).item()
            results[layer_idx] = toxic_sim

        return results

    def identify_high_risk_features(self, threshold: float | None = None) -> Tensor:
        """Identify features with highest risk of spillover.

        Args:
            threshold: Optional override for similarity threshold

        Returns:
            (n_risk,) indices of high-risk features
        """
        effective_threshold = threshold or self.config.toxic_similarity_threshold

        high_risk = []
        for layer, sim_matrix in self._layer_similarities.items():
            max_toxic_sim = sim_matrix.max(dim=1).values
            risk_features = (max_toxic_sim > effective_threshold).nonzero(as_tuple=True)[0]
            high_risk.extend(risk_features.tolist())

        return torch.tensor(list(set(high_risk)), device=self.device)

    def compute_alignment_score(self, activations: Tensor) -> float:
        """Compute overall alignment score based on feature geometry.

        Args:
            activations: (d_model,) or (seq_len, d_model) activation vector

        Returns:
            Alignment score between 0.0 (aligned) and 1.0 (misaligned)
        """
        flat = activations.flatten()
        if self._feature_directions is None:
            return 0.5

        projected = self._feature_directions.T @ flat
        toxic_projection = projected[self._toxic_feature_indices]
        misalign_projection = projected[self._misalignment_feature_indices]

        toxic_score = torch.mean(torch.abs(toxic_projection)).item()
        misalign_score = torch.mean(torch.abs(misalign_projection)).item()

        if toxic_score + misalign_score < 1e-6:
            return 0.5

        risk = toxic_score / (toxic_score + misalign_score + 1e-8)
        return min(1.0, max(0.0, risk))

    def get_superposition_gap(self, target_feature: int) -> float:
        """Compute the superposition gap for a target feature.

        The gap measures how much the feature has been amplified relative
        to geometrically adjacent toxic features.

        Args:
            target_feature: Index of target feature

        Returns:
            Superposition gap score
        """
        if self._feature_directions is None:
            return 0.0

        target_dir = self._feature_directions[:, target_feature]
        target_norm = target_dir / (target_dir.norm() + 1e-8)

        toxic_dirs = self._feature_directions[:, self._toxic_feature_indices]
        toxic_norm = toxic_dirs / (toxic_dirs.norm(dim=0, keepdim=True) + 1e-8)

        similarities = torch.mm(target_norm.unsqueeze(0), toxic_norm).squeeze(0)
        max_toxic_sim = similarities.max().item()

        alignment = 1.0 - max_toxic_sim
        return alignment

    def _validate_tensor(self, tensor: Tensor, name: str, dim: int) -> None:
        """Validate tensor dimensions."""
        if tensor.dim() != dim:
            raise ValueError(f"{name} must have {dim} dimensions, got {tensor.dim()}")


class GradientSpilloverModel:
    """Model gradient spillover from activation updates to feature updates.

    Implements the derivation: Δf_j ≈ α·<d_j, d_insecure>

    Tracks how fine-tuning activation updates cause unintended spillover
    onto geometrically nearby features in the superposed space.
    """

    def __init__(self, config: SuperpositionGeometryConfig):
        """Initialize the spillover model.

        Args:
            config: SuperpositionGeometryConfig instance
        """
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)

        self._insecure_direction: Tensor | None = None
        self._feature_directions: Tensor | None = None
        self._spillover_history: list[dict[str, Tensor]] = []
        self._accumulated_spillover: Tensor | None = None
        self._step_count = 0

    def set_insecure_direction(self, d_insecure: Tensor) -> None:
        """Set the insecure gradient direction.

        Args:
            d_insecure: (d_model,) or (d_model, 1) insecure gradient direction
        """
        flat = d_insecure.flatten()
        self._insecure_direction = flat / (flat.norm() + 1e-8)
        self._insecure_direction = self._insecure_direction.to(self.device)

    def set_feature_directions(self, feature_dirs: Tensor) -> None:
        """Set the feature directions from SAE decode matrix.

        Args:
            feature_dirs: (d_model, n_features) feature direction matrix
        """
        self._feature_directions = feature_dirs.to(self.device)

    def compute_spillover(
        self,
        activation_update: Tensor,
        alpha: float = 1.0,
    ) -> Tensor:
        """Compute spillover onto nearby features from activation update.

        Implements: Δf_j ≈ α·<d_j, d_insecure>

        Args:
            activation_update: (d_model,) activation update Δh
            alpha: Learning rate scaling factor

        Returns:
            (n_features,) spillover values for each feature
        """
        if self._feature_directions is None or self._insecure_direction is None:
            raise RuntimeError(
                "Must set feature_directions and insecure_direction before computing spillover"
            )

        d_insecure = self._insecure_direction
        if activation_update.numel() == d_insecure.numel():
            d_insecure = activation_update / (activation_update.norm() + 1e-8)

        feature_norm = self._feature_directions / (
            self._feature_directions.norm(dim=0, keepdim=True) + 1e-8
        )

        spillover = alpha * (feature_norm.T @ d_insecure)

        return spillover.detach()

    def compute_layer_spillover(
        self,
        layer_updates: dict[int, Tensor],
        layer_feature_map: dict[int, Tensor],
        alpha: float = 1.0,
    ) -> dict[int, Tensor]:
        """Compute spillover per layer with layer-specific feature spaces.

        Args:
            layer_updates: Dict mapping layer_idx to (d_model,) activation updates
            layer_feature_map: Dict mapping layer_idx to (d_model, n_features) SAE weights
            alpha: Learning rate scaling factor

        Returns:
            Dict mapping layer_idx to (n_features,) spillover values
        """
        results = {}
        for layer_idx, update in layer_updates.items():
            if layer_idx not in layer_feature_map:
                continue

            self.set_feature_directions(layer_feature_map[layer_idx])
            self.set_insecure_direction(update)
            results[layer_idx] = self.compute_spillover(update, alpha)

        return results

    def accumulate_spillover(self, spillover: Tensor) -> None:
        """Accumulate spillover over multiple steps for statistical analysis.

        Args:
            spillover: (n_features,) spillover values from one step
        """
        if self._accumulated_spillover is None:
            self._accumulated_spillover = spillover.detach().to(self.device)
        else:
            self._accumulated_spillover += spillover.detach().to(self.device)

        self._step_count += 1

        if self._step_count >= self.config.accumulation_steps:
            self._spillover_history.append(
                {"step": self._step_count, "spillover": self._accumulated_spillover.clone()}
            )
            self._accumulated_spillover = None
            self._step_count = 0

    def get_spillover_statistics(self) -> dict[str, float]:
        """Compute statistics over accumulated spillover.

        Returns:
            Dictionary with mean, std, max, min spillover values
        """
        if len(self._spillover_history) == 0:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "steps": 0}

        all_spillover = torch.stack([h["spillover"] for h in self._spillover_history])

        stats = {
            "mean": all_spillover.mean().item(),
            "std": all_spillover.std().item(),
            "max": all_spillover.max().item(),
            "min": all_spillover.min().item(),
            "steps": len(self._spillover_history),
        }
        return stats

    def predict_misalignment_risk(self, feature_idx: int) -> float:
        """Predict misalignment risk for a feature based on spillover history.

        Args:
            feature_idx: Feature index

        Returns:
            Risk score between 0.0 (safe) and 1.0 (high risk)
        """
        if len(self._spillover_history) == 0:
            return 0.0

        all_spillover = torch.stack([h["spillover"] for h in self._spillover_history])
        feature_spillover = all_spillover[:, feature_idx]

        mean_spillover = feature_spillover.mean().item()
        std_spillover = feature_spillover.std().item()

        risk = min(1.0, abs(mean_spillover) + self.config.spillover_coefficient * std_spillover)
        return risk


class GeometryAwareFilter:
    """Filter training data by removing samples closest to toxic features.

    Uses SAE activations to identify and score training samples based on
    their proximity to toxic features in the superposition space.
    Achieves 34.5% misalignment reduction per arXiv:2605.00842.
    """

    def __init__(self, config: SuperpositionGeometryConfig):
        """Initialize the geometry-aware filter.

        Args:
            config: SuperpositionGeometryConfig instance
        """
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)

        self._feature_directions: Tensor | None = None
        self._toxic_feature_indices: Tensor | None = None
        self._sample_scores: Tensor | None = None
        self._filtered_indices: Tensor | None = None

    def fit(self, feature_directions: Tensor, toxic_indices: Tensor) -> None:
        """Fit the filter with feature directions and toxic feature indices.

        Args:
            feature_directions: (d_model, n_features) SAE decode matrix
            toxic_indices: (n_toxic,) indices of toxic features
        """
        self._feature_directions = feature_directions.to(self.device)
        self._toxic_feature_indices = toxic_indices.to(self.device)

        self.logger.info(
            "Fitted GeometryAwareFilter with %d features, %d toxic",
            feature_directions.shape[1],
            len(toxic_indices),
        )

    def score_samples(self, activations: Tensor) -> Tensor:
        """Score training samples by their proximity to toxic features.

        Args:
            activations: (n_samples, seq_len, d_model) or (n_samples, d_model)
                         activation tensor

        Returns:
            (n_samples,) toxicity scores per sample
        """
        if self._feature_directions is None:
            raise RuntimeError("Filter must be fit before scoring samples")

        if activations.dim() == 3:
            activations = activations.mean(dim=1)

        activations = activations.to(self.device)

        projected = self._feature_directions.T @ activations.T
        toxic_projection = projected[:, self._toxic_feature_indices]

        scores = torch.norm(toxic_projection, dim=0)

        self._sample_scores = scores.detach().cpu()
        return scores

    def filter_samples(
        self,
        activations: Tensor,
        num_remove: int | None = None,
        remove_ratio: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Filter training samples by removing those closest to toxic features.

        Args:
            activations: (n_samples, seq_len, d_model) or (n_samples, d_model)
            num_remove: Exact number of samples to remove
            remove_ratio: Fraction of samples to remove (0.0-1.0)

        Returns:
            Tuple of (kept_indices, removed_indices)
        """
        scores = self.score_samples(activations)

        n_samples = scores.shape[0]
        if num_remove is None:
            if remove_ratio is not None:
                num_remove = int(n_samples * remove_ratio)
            else:
                num_remove = int(n_samples * self.config.filter_remove_ratio)

        num_remove = min(num_remove, n_samples - 1)

        sorted_indices = torch.argsort(scores, descending=True)
        removed_indices = sorted_indices[:num_remove]
        kept_indices = sorted_indices[num_remove:]

        self._filtered_indices = kept_indices

        self.logger.info(
            "Filtered %d samples (removed %d, kept %d)",
            n_samples,
            num_remove,
            n_samples - num_remove,
        )

        return kept_indices, removed_indices

    def get_filtered_activations(
        self,
        activations: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Get filtered activation tensor.

        Args:
            activations: (n_samples, seq_len, d_model) or (n_samples, d_model)
            mask: Optional boolean mask for custom filtering

        Returns:
            Filtered activation tensor
        """
        if mask is not None:
            return activations[mask]

        if self._filtered_indices is None:
            raise RuntimeError("Must run filter_samples before get_filtered_activations")

        return activations[self._filtered_indices]

    def compute_toxic_distance(self, activations: Tensor, aggregate: str = "mean") -> float:
        """Compute distance to toxic feature subspace.

        Args:
            activations: (d_model,) or (n_samples, d_model) activation tensor
            aggregate: Aggregation method ('mean', 'max', 'min')

        Returns:
            Aggregated toxic distance
        """
        if self._feature_directions is None:
            raise RuntimeError("Filter must be fit before computing distance")

        if activations.dim() == 2:
            activations = activations.mean(dim=0)

        activations = activations.to(self.device)

        projected = self._feature_directions.T @ activations
        toxic_projection = projected[self._toxic_feature_indices]

        if aggregate == "mean":
            return torch.mean(torch.abs(toxic_projection)).item()
        elif aggregate == "max":
            return torch.max(torch.abs(toxic_projection)).item()
        elif aggregate == "min":
            return torch.min(torch.abs(toxic_projection)).item()
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")


class MisalignmentGeometryMonitor:
    """Monitor alignment during fine-tuning with feature similarity dynamics.

    Tracks feature similarity dynamics over training steps and predicts
    misalignment risk based on geometric relationships between alignment
    and toxic features.

    Per arXiv:2605.00842:
    - Misalignment-inducing features have higher similarity to toxic features
      across all layers, especially earlier layers
    - Similarity grows temporally during fine-tuning alongside misaligned output counts
    """

    def __init__(self, config: SuperpositionGeometryConfig):
        """Initialize the misalignment monitor.

        Args:
            config: SuperpositionGeometryConfig instance
        """
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)

        self._analyzer = FeatureSuperpositionAnalyzer(config)
        self._spillover_model = GradientSpilloverModel(config)

        self._history: list[dict[str, Any]] = []
        self._layer_trajectories: dict[int, list[float]] = {}
        self._current_risk: float = 0.0
        self._step: int = 0

    def register_features(
        self,
        sae_weights: Tensor,
        toxic_indices: Tensor,
        misalignment_indices: Tensor,
    ) -> None:
        """Register feature indices for monitoring.

        Args:
            sae_weights: (d_model, n_features) SAE decode matrix
            toxic_indices: (n_toxic,) indices of toxic features
            misalignment_indices: (n_misalign,) indices of misalignment features
        """
        self._analyzer.fit(sae_weights, toxic_indices, misalignment_indices)
        self._spillover_model.set_feature_directions(sae_weights)

    def update(
        self,
        activations: Tensor,
        layer_idx: int,
        gradient_update: Tensor | None = None,
    ) -> dict[str, Any]:
        """Update monitor with new activation and gradient information.

        Args:
            activations: Current layer activations
            layer_idx: Current layer index
            gradient_update: Optional activation update for spillover computation

        Returns:
            Dictionary with monitoring metrics
        """
        if layer_idx % self.config.layer_sample_interval != 0:
            return {}

        metrics = {}

        if gradient_update is not None:
            self._spillover_model.set_insecure_direction(gradient_update)
            spillover = self._spillover_model.compute_spillover(gradient_update)
            self._spillover_model.accumulate_spillover(spillover)

        similarities = self._analyzer.compute_feature_similarities(layer_idx, activations)

        mean_toxic_sim = similarities.max(dim=1).values.mean().item()
        max_toxic_sim = similarities.max().item()

        if layer_idx not in self._layer_trajectories:
            self._layer_trajectories[layer_idx] = []
        self._layer_trajectories[layer_idx].append(mean_toxic_sim)

        alignment_score = self._analyzer.compute_alignment_score(activations)

        self._current_risk = alignment_score
        self._step += 1

        metrics.update(
            {
                "step": self._step,
                "layer": layer_idx,
                "mean_toxic_similarity": mean_toxic_sim,
                "max_toxic_similarity": max_toxic_sim,
                "alignment_score": alignment_score,
                "risk": self._current_risk,
            }
        )

        self._history.append(metrics)

        return metrics

    def check_misalignment(self) -> bool:
        """Check if misalignment risk exceeds threshold.

        Returns:
            True if misalignment detected
        """
        if self._current_risk > self.config.misalignment_risk_threshold:
            self.logger.warning(
                "MISALIGNMENT DETECTED: risk=%.4f, threshold=%.4f",
                self._current_risk,
                self.config.misalignment_risk_threshold,
            )
            return True
        return False

    def get_risk_score(self) -> float:
        """Get current misalignment risk score.

        Returns:
            Risk score between 0.0 (aligned) and 1.0 (misaligned)
        """
        return self._current_risk

    def get_layer_dynamics(self) -> dict[int, list[float]]:
        """Get similarity trajectories per layer.

        Returns:
            Dictionary mapping layer_idx to list of similarity values over time
        """
        return self._layer_trajectories.copy()

    def get_temporal_trend(self) -> dict[str, float]:
        """Compute temporal trend of misalignment across training steps.

        Returns:
            Dictionary with trend metrics (slope, current_value, initial_value)
        """
        if len(self._history) < 2:
            return {"slope": 0.0, "current": 0.0, "initial": 0.0}

        steps = [h["step"] for h in self._history]
        risks = [h["alignment_score"] for h in self._history]

        steps_tensor = torch.tensor(steps, dtype=torch.float)
        risks_tensor = torch.tensor(risks, dtype=torch.float)

        centered_steps = steps_tensor - steps_tensor.mean()
        centered_risks = risks_tensor - risks_tensor.mean()

        slope = (
            torch.dot(centered_steps, centered_risks) / (centered_steps.norm() ** 2 + 1e-8)
        ).item()

        return {
            "slope": slope,
            "current": risks[-1],
            "initial": risks[0],
            "total_steps": len(self._history),
        }

    def predict_future_risk(self, n_steps: int = 100) -> float:
        """Predict misalignment risk after n additional steps.

        Args:
            n_steps: Number of steps to predict ahead

        Returns:
            Predicted risk score
        """
        trend = self.get_temporal_trend()

        current_risk = trend["current"]
        slope = trend["slope"]

        predicted = current_risk + slope * n_steps
        predicted = min(1.0, max(0.0, predicted))

        return predicted

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive monitoring summary.

        Returns:
            Dictionary with all monitoring metrics and trajectories
        """
        spillover_stats = self._spillover_model.get_spillover_statistics()
        temporal_trend = self.get_temporal_trend()

        return {
            "current_risk": self._current_risk,
            "total_steps": self._step,
            "spillover_statistics": spillover_stats,
            "temporal_trend": temporal_trend,
            "layer_trajectories": {
                k: {
                    "initial": v[0] if len(v) > 0 else 0.0,
                    "current": v[-1] if len(v) > 0 else 0.0,
                    "delta": (v[-1] - v[0]) if len(v) > 1 else 0.0,
                    "length": len(v),
                }
                for k, v in self._layer_trajectories.items()
            },
            "misalignment_detected": self.check_misalignment(),
        }

    def reset(self) -> None:
        """Reset monitoring state."""
        self._history.clear()
        self._layer_trajectories.clear()
        self._current_risk = 0.0
        self._step = 0


__all__ = [
    "SuperpositionGeometryConfig",
    "FeatureSuperpositionAnalyzer",
    "GradientSpilloverModel",
    "GeometryAwareFilter",
    "MisalignmentGeometryMonitor",
]
