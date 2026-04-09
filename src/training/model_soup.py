"""Model soups and weight-space ensembling (Wortsman et al. 2022).

Complementary to model_merge_advanced.py (DARE/TIES/DARE-TIES).
This module focuses on:
  - Uniform and weighted model soup averaging
  - Greedy model soup construction
  - Weight divergence analysis between models
  - Linear interpolation and loss barrier computation
  - Learned mixing weights via gradient descent
  - High-level ModelSoupEnsemble orchestrator
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelSoupConfig:
    """Configuration for model soup / weight-space ensembling.

    Attributes:
        method: Averaging strategy — "uniform" | "greedy" | "learned".
        n_models: Expected number of models to ensemble.
        eval_metric: Metric used by greedy soup — "loss" (lower=better)
                     or "acc" (higher=better).
        lr: Learning rate for the learned mixer optimisation.
        n_mix_iter: Number of gradient steps for learned mixing weights.
    """

    method: str = "uniform"
    n_models: int = 3
    eval_metric: str = "loss"
    lr: float = 0.01
    n_mix_iter: int = 50


# ---------------------------------------------------------------------------
# Core soup functions (operate on nn.Module)
# ---------------------------------------------------------------------------


def uniform_soup(models: list[nn.Module]) -> dict[str, Tensor]:
    """Average all model state dicts uniformly.

    Args:
        models: List of models with identical architectures.

    Returns:
        Averaged state dict (float32 intermediate, cast back to original dtype).
    """
    if not models:
        raise ValueError("models list must be non-empty")

    state_dicts = [m.state_dict() for m in models]
    n = len(state_dicts)
    averaged: dict[str, Tensor] = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        averaged[key] = stacked.mean(dim=0).to(state_dicts[0][key].dtype)
    return averaged


def weighted_soup(
    models: list[nn.Module],
    weights: list[float],
) -> dict[str, Tensor]:
    """Weighted average of model state dicts.

    Weights are normalised internally so they sum to 1.

    Args:
        models: List of models with identical architectures.
        weights: Per-model mixing weights (need not sum to 1 beforehand).

    Returns:
        Weighted-averaged state dict.

    Raises:
        ValueError: If lengths mismatch or weights sum to zero/negative.
    """
    if not models:
        raise ValueError("models list must be non-empty")
    if len(models) != len(weights):
        raise ValueError("models and weights must have the same length")

    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    norm_weights = [w / total for w in weights]

    state_dicts = [m.state_dict() for m in models]
    blended: dict[str, Tensor] = {}
    for key in state_dicts[0]:
        acc = torch.zeros_like(state_dicts[0][key], dtype=torch.float)
        for sd, w in zip(state_dicts, norm_weights):
            acc = acc + w * sd[key].float()
        blended[key] = acc.to(state_dicts[0][key].dtype)
    return blended


def greedy_soup(
    models: list[nn.Module],
    eval_fn: Callable[[nn.Module], float],
    maximize: bool = False,
) -> dict[str, Tensor]:
    """Greedy model soup: iteratively add models if they improve the ensemble.

    Algorithm:
        1. Start with models[0] as the initial soup.
        2. For each subsequent model: tentatively add it (uniform average of
           current soup + new model).
        3. Evaluate with eval_fn; keep the addition if the metric improves
           (lower when not maximize, higher when maximize).
        4. Return the final soup state dict.

    Args:
        models: Ordered list of candidate models.
        eval_fn: Callable that receives an nn.Module and returns a scalar.
        maximize: If True, higher eval_fn is better; otherwise lower is better.

    Returns:
        State dict of the final greedy soup.
    """
    if not models:
        raise ValueError("models list must be non-empty")

    # Helper: load a state dict into a deepcopy of the first model
    def _load_state(state_dict: dict[str, Tensor]) -> nn.Module:
        m = copy.deepcopy(models[0])
        m.load_state_dict(state_dict)
        return m

    # Start with the first model
    soup_state = {k: v.clone() for k, v in models[0].state_dict().items()}
    soup_model = _load_state(soup_state)
    best_score = eval_fn(soup_model)
    soup_members: list[dict[str, Tensor]] = [soup_state]

    for candidate in models[1:]:
        candidate_state = candidate.state_dict()
        # Tentative: average current soup members + new candidate
        tentative_states = soup_members + [candidate_state]
        tentative_avg: dict[str, Tensor] = {}
        n = len(tentative_states)
        for key in soup_state:
            stacked = torch.stack([sd[key].float() for sd in tentative_states])
            tentative_avg[key] = stacked.mean(dim=0).to(soup_state[key].dtype)

        tentative_model = _load_state(tentative_avg)
        score = eval_fn(tentative_model)

        improved = (score > best_score) if maximize else (score < best_score)
        if improved:
            soup_members.append({k: v.clone() for k, v in candidate_state.items()})
            soup_state = tentative_avg
            best_score = score

    return soup_state


# ---------------------------------------------------------------------------
# Weight divergence
# ---------------------------------------------------------------------------


def compute_weight_divergence(
    model_a: nn.Module,
    model_b: nn.Module,
) -> dict[str, float]:
    """Compute per-layer weight divergence between two models.

    Args:
        model_a: First model.
        model_b: Second model (same architecture).

    Returns:
        Dict with keys:
            "mean_l2": mean L2 distance per parameter tensor.
            "max_l2": maximum L2 distance across all parameter tensors.
            "mean_cosine_sim": mean cosine similarity per parameter tensor
                               (higher = more similar, range [-1, 1]).
    """
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    l2_distances: list[float] = []
    cosine_sims: list[float] = []

    for key in sd_a:
        if key not in sd_b:
            continue
        a = sd_a[key].float().flatten()
        b = sd_b[key].float().flatten()

        l2 = (a - b).norm().item()
        l2_distances.append(l2)

        # Cosine similarity (handle zero vectors)
        norm_a = a.norm()
        norm_b = b.norm()
        if norm_a > 0 and norm_b > 0:
            cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        else:
            cos_sim = 1.0 if torch.allclose(a, b) else 0.0
        cosine_sims.append(cos_sim)

    if not l2_distances:
        return {"mean_l2": 0.0, "max_l2": 0.0, "mean_cosine_sim": 1.0}

    return {
        "mean_l2": float(sum(l2_distances) / len(l2_distances)),
        "max_l2": float(max(l2_distances)),
        "mean_cosine_sim": float(sum(cosine_sims) / len(cosine_sims)),
    }


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def interpolate_models(
    model_a: nn.Module,
    model_b: nn.Module,
    alpha: float,
) -> dict[str, Tensor]:
    """Linear interpolation: (1 - alpha) * model_a + alpha * model_b.

    Args:
        model_a: Source model (alpha=0 endpoint).
        model_b: Target model (alpha=1 endpoint).
        alpha: Interpolation coefficient in [0, 1].

    Returns:
        Interpolated state dict.
    """
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    interpolated: dict[str, Tensor] = {}
    for key in sd_a:
        a = sd_a[key].float()
        b = sd_b[key].float() if key in sd_b else a
        interpolated[key] = ((1.0 - alpha) * a + alpha * b).to(sd_a[key].dtype)
    return interpolated


# ---------------------------------------------------------------------------
# Loss barrier
# ---------------------------------------------------------------------------


def compute_loss_barrier(
    model_a: nn.Module,
    model_b: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    n_points: int = 5,
) -> dict[str, float]:
    """Compute loss landscape along the linear interpolation path.

    Evaluates at n_points evenly spaced alpha values from 0 to 1 (inclusive).

    Args:
        model_a: First endpoint model.
        model_b: Second endpoint model.
        eval_fn: Callable receiving an nn.Module and returning a scalar loss.
        n_points: Number of points to evaluate (including endpoints).

    Returns:
        Dict with:
            "max_barrier": max(interpolated_losses) - mean(endpoint_losses).
            "mean_loss": mean loss across all evaluated points.
            "barrier_location_alpha": alpha value where the maximum loss occurs.
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    alphas = [i / (n_points - 1) for i in range(n_points)]
    losses: list[float] = []

    for alpha in alphas:
        interp_state = interpolate_models(model_a, model_b, alpha)
        interp_model = copy.deepcopy(model_a)
        interp_model.load_state_dict(interp_state)
        loss = eval_fn(interp_model)
        losses.append(loss)

    endpoint_mean = (losses[0] + losses[-1]) / 2.0
    max_loss = max(losses)
    max_idx = losses.index(max_loss)
    barrier_alpha = alphas[max_idx]

    return {
        "max_barrier": max_loss - endpoint_mean,
        "mean_loss": float(sum(losses) / len(losses)),
        "barrier_location_alpha": float(barrier_alpha),
    }


# ---------------------------------------------------------------------------
# Learned soup mixer
# ---------------------------------------------------------------------------


class LearnedSoupMixer(nn.Module):
    """Learn optimal mixing weights via gradient descent.

    Maintains a set of logits over n_models; mixing weights are the softmax
    of those logits (always sum to 1, always positive).

    Args:
        n_models: Number of models to mix.
    """

    def __init__(self, n_models: int) -> None:
        super().__init__()
        self.mixing_logits = nn.Parameter(torch.zeros(n_models))

    def get_weights(self) -> Tensor:
        """Return softmax-normalised mixing weights of shape (n_models,)."""
        return F.softmax(self.mixing_logits, dim=0)

    def forward(self, model_outputs: list[Tensor]) -> Tensor:
        """Weighted average of model output tensors.

        Args:
            model_outputs: List of (B, T, V) tensors, one per model.

        Returns:
            (B, T, V) weighted average.
        """
        weights = self.get_weights()  # (n_models,)
        # Stack → (n_models, B, T, V), then weighted sum
        stacked = torch.stack(model_outputs, dim=0)  # (n_models, B, T, V)
        # weights shape: (n_models,) → (n_models, 1, 1, 1)
        w = weights.view(-1, *([1] * (stacked.dim() - 1)))
        return (stacked * w).sum(dim=0)


# ---------------------------------------------------------------------------
# High-level ensemble
# ---------------------------------------------------------------------------


class ModelSoupEnsemble:
    """Build and evaluate model soup ensembles.

    Args:
        base_model: Reference model whose architecture is used for all copies.
        cfg: :class:`ModelSoupConfig` controlling the ensemble strategy.
    """

    def __init__(self, base_model: nn.Module, cfg: ModelSoupConfig) -> None:
        self.base_model = base_model
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _make_model_from_state(self, state_dict: dict[str, Tensor]) -> nn.Module:
        """Create a deepcopy of base_model and load the given state dict."""
        m = copy.deepcopy(self.base_model)
        m.load_state_dict(state_dict)
        return m

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_soup(
        self,
        models: list[nn.Module],
        eval_fn: Callable[[nn.Module], float] | None = None,
    ) -> nn.Module:
        """Create a soup using the configured method.

        Args:
            models: Candidate models to average.
            eval_fn: Required when method == "greedy".

        Returns:
            A new nn.Module with the soup state dict loaded.

        Raises:
            ValueError: If method == "greedy" and eval_fn is None.
        """
        method = self.cfg.method

        if method == "uniform":
            state_dict = uniform_soup(models)

        elif method == "greedy":
            if eval_fn is None:
                raise ValueError("eval_fn is required for greedy soup method")
            maximize = self.cfg.eval_metric != "loss"
            state_dict = greedy_soup(models, eval_fn, maximize=maximize)

        elif method == "learned":
            # For "learned" we fall back to uniform (learning requires external loop)
            logger.warning(
                "create_soup with method='learned' uses uniform soup as initialisation. "
                "Use LearnedSoupMixer directly for gradient-based weight optimisation."
            )
            state_dict = uniform_soup(models)

        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from 'uniform', 'greedy', 'learned'."
            )

        return self._make_model_from_state(state_dict)

    def ensemble_predict(
        self,
        models: list[nn.Module],
        input_ids: Tensor,
    ) -> Tensor:
        """Average logit predictions from multiple models.

        Each model is called as ``loss, logits, pkv = model(input_ids)`` and
        the logits are averaged.

        Args:
            models: List of models.
            input_ids: (B, T) integer token tensor.

        Returns:
            (B, T, V) averaged logits.
        """
        all_logits: list[Tensor] = []
        for model in models:
            _loss, logits, _pkv = model(input_ids)
            all_logits.append(logits)
        stacked = torch.stack(all_logits, dim=0)  # (n_models, B, T, V)
        return stacked.mean(dim=0)

    def find_best_interpolation(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        eval_fn: Callable[[nn.Module], float],
        n_trials: int = 5,
    ) -> tuple[float, float]:
        """Grid search over alpha in [0, 1] to find the best interpolation.

        Args:
            model_a: First endpoint model.
            model_b: Second endpoint model.
            eval_fn: Callable receiving an nn.Module and returning a scalar.
            n_trials: Number of evenly-spaced alpha values to try.

        Returns:
            (best_alpha, best_score) tuple.
        """
        maximize = self.cfg.eval_metric != "loss"
        alphas = [i / max(n_trials - 1, 1) for i in range(n_trials)]

        best_alpha = alphas[0]
        best_score: float | None = None

        for alpha in alphas:
            interp_state = interpolate_models(model_a, model_b, alpha)
            interp_model = self._make_model_from_state(interp_state)
            score = eval_fn(interp_model)

            if best_score is None:
                best_score = score
                best_alpha = alpha
            elif maximize and score > best_score:
                best_score = score
                best_alpha = alpha
            elif (not maximize) and score < best_score:
                best_score = score
                best_alpha = alpha

        return float(best_alpha), float(best_score)  # type: ignore[arg-type]
