"""Model Soup: checkpoint averaging for improved generalization.

Implements uniform, greedy, and learned weight averaging of multiple
fine-tuned checkpoints (all from the same pre-trained base).

References:
    Wortsman et al. 2022, "Model soups: averaging weights of multiple fine-tuned
    models improves accuracy without increasing inference time"
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SoupConfig:
    """Configuration for ModelSoup.

    Attributes:
        method: One of "uniform", "greedy", or "learned".
        max_models: Maximum number of checkpoints to store.
    """

    method: str = "uniform"
    max_models: int = 10


# ---------------------------------------------------------------------------
# Core averaging functions
# ---------------------------------------------------------------------------


def uniform_soup(state_dicts: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Simple uniform average of all models' state dicts.

    Args:
        state_dicts: Non-empty list of state dicts from identically architected models.

    Returns:
        Averaged state dict.

    Raises:
        ValueError: If state_dicts is empty.
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")
    averaged: dict[str, Tensor] = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        averaged[key] = stacked.mean(dim=0).to(state_dicts[0][key].dtype)
    return averaged


def greedy_soup(
    state_dicts: list[dict[str, Tensor]],
    eval_fn: Callable[[dict[str, Tensor]], float],
    higher_is_better: bool = True,
) -> dict[str, Tensor]:
    """Greedy model soup: add each model only if it improves the soup's eval score.

    The first candidate always seeds the soup. Subsequent candidates are
    tentatively averaged in; the addition is kept only if eval_fn strictly improves.

    Args:
        state_dicts: Ordered list of candidate state dicts.
        eval_fn: Callable that receives a state dict and returns a scalar.
        higher_is_better: True means higher eval_fn score is better.

    Returns:
        State dict of the best greedy soup found.

    Raises:
        ValueError: If state_dicts is empty.
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")

    soup_members: list[dict[str, Tensor]] = [{k: v.clone() for k, v in state_dicts[0].items()}]
    soup_state = uniform_soup(soup_members)
    best_score = eval_fn(soup_state)

    for candidate in state_dicts[1:]:
        tentative_members = soup_members + [{k: v.clone() for k, v in candidate.items()}]
        tentative_avg = uniform_soup(tentative_members)
        score = eval_fn(tentative_avg)

        improved = (score > best_score) if higher_is_better else (score < best_score)
        if improved:
            soup_members.append({k: v.clone() for k, v in candidate.items()})
            soup_state = tentative_avg
            best_score = score

    return soup_state


def learned_soup(
    state_dicts: list[dict[str, Tensor]],
    weights: Tensor | None = None,
) -> dict[str, Tensor]:
    """Weighted average where weights is a (n_models,) mixing-coefficient tensor.

    Weights are normalised to sum to 1 before blending. When weights is None,
    falls back to uniform averaging.

    Args:
        state_dicts: Non-empty list of candidate state dicts.
        weights: Optional (n_models,) float tensor of mixing coefficients.

    Returns:
        Weighted-averaged state dict.

    Raises:
        ValueError: If state_dicts is empty or weights shape is wrong.
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")
    n = len(state_dicts)

    if weights is None:
        w = torch.ones(n, dtype=torch.float32) / n
    else:
        if weights.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {tuple(weights.shape)}")
        total = weights.float().sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        w = weights.float() / total

    averaged: dict[str, Tensor] = {}
    for key in state_dicts[0]:
        acc = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
        for sd, wi in zip(state_dicts, w.tolist()):
            acc = acc + wi * sd[key].float()
        averaged[key] = acc.to(state_dicts[0][key].dtype)
    return averaged


def interpolate_models(
    state_dict_a: dict[str, Tensor],
    state_dict_b: dict[str, Tensor],
    alpha: float,
) -> dict[str, Tensor]:
    """Linear interpolation between two state dicts: (1-alpha)*A + alpha*B.

    Args:
        state_dict_a: First state dict (alpha=0 endpoint).
        state_dict_b: Second state dict (alpha=1 endpoint).
        alpha: Interpolation coefficient in [0, 1].

    Returns:
        Interpolated state dict.
    """
    interpolated: dict[str, Tensor] = {}
    for key in state_dict_a:
        a = state_dict_a[key].float()
        b = state_dict_b[key].float() if key in state_dict_b else a
        interpolated[key] = ((1.0 - alpha) * a + alpha * b).to(state_dict_a[key].dtype)
    return interpolated


def compute_weight_distance(sd_a: dict[str, Tensor], sd_b: dict[str, Tensor]) -> float:
    """Mean L2 distance across all parameter tensors between two state dicts.

    Args:
        sd_a: First state dict.
        sd_b: Second state dict.

    Returns:
        Mean per-tensor L2 distance as a Python float.
    """
    distances: list[float] = []
    for key in sd_a:
        if key in sd_b:
            diff = sd_a[key].float() - sd_b[key].float()
            distances.append(diff.norm().item())
    if not distances:
        return 0.0
    return float(sum(distances) / len(distances))


# ---------------------------------------------------------------------------
# ModelSoup orchestrator
# ---------------------------------------------------------------------------


class ModelSoup:
    """High-level orchestrator for model soup construction.

    Stores checkpoints as state dicts and dispatches to uniform_soup,
    greedy_soup, or learned_soup based on config.

    Args:
        config: SoupConfig controlling the soup strategy and pool size.
    """

    def __init__(self, config: SoupConfig) -> None:
        self.config = config
        self._checkpoints: list[dict[str, Tensor]] = []

    def add_checkpoint(self, state_dict: dict[str, Tensor]) -> None:
        """Store a checkpoint state dict (up to config.max_models, FIFO eviction)."""
        if len(self._checkpoints) >= self.config.max_models:
            self._checkpoints.pop(0)
        self._checkpoints.append({k: v.clone() for k, v in state_dict.items()})

    def cook(
        self,
        eval_fn: Callable[[dict[str, Tensor]], float] | None = None,
        weights: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Build the soup using the method specified in config.

        Args:
            eval_fn: Required when config.method == "greedy".
            weights: Optional mixing coefficients for config.method == "learned".

        Returns:
            Resulting soup state dict.

        Raises:
            ValueError: If no checkpoints added, or greedy method without eval_fn.
        """
        if not self._checkpoints:
            raise ValueError("No checkpoints added; call add_checkpoint() first")

        method = self.config.method
        if method == "uniform":
            return uniform_soup(self._checkpoints)
        elif method == "greedy":
            if eval_fn is None:
                raise ValueError("eval_fn is required for greedy soup method")
            return greedy_soup(self._checkpoints, eval_fn=eval_fn)
        elif method == "learned":
            return learned_soup(self._checkpoints, weights=weights)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from 'uniform', 'greedy', 'learned'."
            )

    def get_soup_stats(
        self,
        base_state_dict: dict[str, Tensor],
        soup_state_dict: dict[str, Tensor],
    ) -> dict[str, float]:
        """Compute weight-distance statistics between a base and soup state dict.

        Returns:
            Dict with keys: mean_l2_distance, max_l2_distance, n_params.
        """
        distances: list[float] = []
        n_params = 0
        for key in base_state_dict:
            if key in soup_state_dict:
                a = base_state_dict[key].float()
                b = soup_state_dict[key].float()
                distances.append((a - b).norm().item())
                n_params += a.numel()
        mean_l2 = float(sum(distances) / len(distances)) if distances else 0.0
        max_l2 = float(max(distances)) if distances else 0.0
        return {
            "mean_l2_distance": mean_l2,
            "max_l2_distance": max_l2,
            "n_params": float(n_params),
        }
