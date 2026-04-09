"""Model soups: weight averaging of fine-tuned variants for improved generalization (Wortsman et al. 2022)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class SoupConfig:
    """Configuration for model soup averaging."""

    averaging_method: str = "uniform"  # "uniform" | "fisher_weighted" | "greedy"
    n_models: int = 4
    held_out_fraction: float = 0.1  # for greedy soup validation
    temperature: float = 1.0  # for softmax weighting of Fisher scores


def uniform_soup(state_dicts: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Simple arithmetic mean of all state dicts.

    All models must have identical architecture (same keys and shapes).

    Args:
        state_dicts: List of model state dicts to average.

    Returns:
        Averaged state dict.
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")

    n = len(state_dicts)
    avg = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        avg[key] = stacked.mean(dim=0).to(state_dicts[0][key].dtype)
    return avg


def weighted_soup(
    state_dicts: list[dict[str, Tensor]],
    weights: list[float],
) -> dict[str, Tensor]:
    """Weighted average of state dicts with normalized weights.

    Args:
        state_dicts: List of model state dicts.
        weights: Raw weights (will be normalized to sum to 1).

    Returns:
        Weighted-average state dict.
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights must have the same length")

    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    norm_weights = [w / total for w in weights]

    blended = {}
    for key in state_dicts[0]:
        acc = torch.zeros_like(state_dicts[0][key], dtype=torch.float)
        for sd, w in zip(state_dicts, norm_weights):
            acc += w * sd[key].float()
        blended[key] = acc.to(state_dicts[0][key].dtype)
    return blended


def fisher_weighted_soup(
    state_dicts: list[dict[str, Tensor]],
    fisher_scores: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Fisher-weighted model averaging.

    For each parameter: p = sum(F_i * p_i) / sum(F_i).
    Falls back to uniform averaging if Fisher info not available for a param.

    Args:
        state_dicts: List of model state dicts.
        fisher_scores: List of Fisher importance dicts (same keys/shapes as state dicts).

    Returns:
        Merged state dict.
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")
    if len(state_dicts) != len(fisher_scores):
        raise ValueError("state_dicts and fisher_scores must have the same length")

    merged = {}
    for key in state_dicts[0]:
        param_dtype = state_dicts[0][key].dtype
        # Check all models have Fisher info for this key
        has_fisher = all(key in fs for fs in fisher_scores)
        if has_fisher:
            numerator = torch.zeros_like(state_dicts[0][key], dtype=torch.float)
            denominator = torch.zeros_like(state_dicts[0][key], dtype=torch.float)
            for sd, fs in zip(state_dicts, fisher_scores):
                fi = fs[key].float().abs()
                numerator += fi * sd[key].float()
                denominator += fi
            # Avoid division by zero: fall back to uniform where denominator is 0
            safe_denom = denominator.clamp(min=1e-10)
            merged[key] = (numerator / safe_denom).to(param_dtype)
        else:
            # Fall back to uniform averaging
            stacked = torch.stack([sd[key].float() for sd in state_dicts])
            merged[key] = stacked.mean(dim=0).to(param_dtype)
    return merged


def greedy_soup(
    state_dicts: list[dict[str, Tensor]],
    eval_fn: Callable[[dict], float],
    base_state_dict: dict[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Greedy soup: iteratively add models if they improve performance.

    Algorithm (Wortsman et al. 2022):
    1. Rank all models by eval_fn score.
    2. Start with the best single model.
    3. Try adding each remaining model; keep if eval_fn improves on the running average.

    Args:
        state_dicts: List of model state dicts (fine-tuned variants).
        eval_fn: Callable that takes a state dict and returns a scalar score (higher = better).
        base_state_dict: Optional; unused (kept for API compatibility).

    Returns:
        Best found averaged state dict.
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")

    # Score all individual models
    scores = [eval_fn(sd) for sd in state_dicts]

    # Sort by descending score
    order = sorted(range(len(state_dicts)), key=lambda i: scores[i], reverse=True)

    # Start with the best model
    best_idx = order[0]
    soup_dicts = [state_dicts[best_idx]]
    best_score = scores[best_idx]
    best_soup = state_dicts[best_idx]

    for idx in order[1:]:
        candidate_soup = uniform_soup(soup_dicts + [state_dicts[idx]])
        candidate_score = eval_fn(candidate_soup)
        if candidate_score >= best_score:
            soup_dicts.append(state_dicts[idx])
            best_score = candidate_score
            best_soup = candidate_soup

    return best_soup


def compute_weight_variance(state_dicts: list[dict[str, Tensor]]) -> dict[str, float]:
    """Compute per-parameter variance across models.

    Args:
        state_dicts: List of model state dicts.

    Returns:
        Dict mapping parameter name to variance scalar (mean variance over elements).
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")

    variances: dict[str, float] = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        # Variance per element, then take mean for a single scalar
        var = stacked.var(dim=0).mean().item()
        variances[key] = var
    return variances


def interpolate_models(
    state_dict_a: dict[str, Tensor],
    state_dict_b: dict[str, Tensor],
    alpha: float = 0.5,
) -> dict[str, Tensor]:
    """Linear interpolation between two model state dicts.

    Result = (1 - alpha) * A + alpha * B.

    Args:
        state_dict_a: First model state dict.
        state_dict_b: Second model state dict.
        alpha: Interpolation factor (0 = model A, 1 = model B).

    Returns:
        Interpolated state dict.
    """
    interpolated = {}
    for key in state_dict_a:
        a = state_dict_a[key].float()
        b = state_dict_b[key].float()
        result = (1.0 - alpha) * a + alpha * b
        interpolated[key] = result.to(state_dict_a[key].dtype)
    return interpolated


class ModelSoup:
    """High-level model soup manager.

    Collects fine-tuned model state dicts and blends them using the
    configured averaging method (uniform, fisher_weighted, or greedy).
    """

    def __init__(self, config: SoupConfig) -> None:
        self.config = config
        self._state_dicts: list[dict[str, Tensor]] = []
        self._scores: list[float] = []

    def add_model(self, state_dict: dict[str, Tensor], score: float | None = None) -> None:
        """Add a model state dict to the soup.

        Args:
            state_dict: Model state dict to add.
            score: Optional evaluation score for this model (used by greedy/fisher methods).
        """
        self._state_dicts.append({k: v.clone() for k, v in state_dict.items()})
        self._scores.append(score if score is not None else float("nan"))

    def blend(self, eval_fn: Callable | None = None) -> dict[str, Tensor]:
        """Blend added models using the configured averaging method.

        Args:
            eval_fn: Required for "greedy" method. Callable that takes a state dict
                     and returns a scalar score (higher = better).

        Returns:
            Blended state dict.
        """
        if not self._state_dicts:
            raise ValueError("No models added to the soup. Call add_model() first.")

        method = self.config.averaging_method

        if method == "uniform":
            return uniform_soup(self._state_dicts)

        elif method == "fisher_weighted":
            # Use scores as a proxy Fisher scalar per model (broadcast to param shape)
            # If scores are NaN, fall back to uniform
            valid_scores = [s for s in self._scores if not (s != s)]  # filter NaN
            if len(valid_scores) == len(self._scores):
                # Build scalar fisher dicts using scores as proxy
                fisher_scores = []
                for i, sd in enumerate(self._state_dicts):
                    fs = {k: torch.full_like(v, self._scores[i], dtype=torch.float)
                          for k, v in sd.items()}
                    fisher_scores.append(fs)
                return fisher_weighted_soup(self._state_dicts, fisher_scores)
            else:
                logger.warning("Some scores are NaN; falling back to uniform soup.")
                return uniform_soup(self._state_dicts)

        elif method == "greedy":
            if eval_fn is None:
                raise ValueError("eval_fn is required for greedy soup method.")
            return greedy_soup(self._state_dicts, eval_fn)

        else:
            raise ValueError(
                f"Unknown averaging_method: {method!r}. "
                "Choose from 'uniform', 'fisher_weighted', 'greedy'."
            )

    def diversity_score(self) -> float:
        """Mean pairwise L2 distance between all model parameters (flattened).

        Returns:
            Mean pairwise L2 distance as a float.
        """
        if len(self._state_dicts) < 2:
            return 0.0

        # Flatten each model's parameters into a single vector
        flat_params = []
        for sd in self._state_dicts:
            flat = torch.cat([v.float().flatten() for v in sd.values()])
            flat_params.append(flat)

        # Compute mean pairwise L2 distance
        n = len(flat_params)
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = (flat_params[i] - flat_params[j]).norm().item()
                total_dist += dist
                count += 1

        return total_dist / count if count > 0 else 0.0
