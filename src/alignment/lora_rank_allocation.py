"""LoRA rank allocation: sensitivity-based budget allocation and adaptive rank pruning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class RankAllocationConfig:
    """Configuration for LoRA rank allocation."""

    total_rank_budget: int = 64  # total rank across all adapters
    min_rank: int = 1  # minimum rank per adapter
    max_rank: int = 16  # maximum rank per adapter
    sensitivity_method: str = "gradient"  # "gradient" | "singular_value" | "uniform"
    prune_threshold: float = 0.01  # prune singular values below this


class LoRAAdapter(nn.Module):
    """Standard LoRA adapter with A/B decomposition.

    W_delta = (lora_B @ lora_A) * scaling

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: Low-rank decomposition rank.
        alpha: LoRA scaling factor; effective scale = alpha / rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: Tensor) -> Tensor:
        """Compute adapter output: x @ lora_A.T @ lora_B.T * scaling."""
        return x @ self.lora_A.T @ self.lora_B.T * self.scaling

    def effective_weight(self) -> Tensor:
        """Return effective weight matrix (out_features, in_features)."""
        return (self.lora_B @ self.lora_A) * self.scaling


def compute_gradient_sensitivity(adapter: LoRAAdapter) -> float:
    """Compute gradient-based sensitivity as mean absolute gradient.

    Args:
        adapter: LoRAAdapter whose parameters may have gradients.

    Returns:
        Mean of |grad| over all adapter parameters, or 0.0 if no gradients.
    """
    total_grad = 0.0
    total_elements = 0

    for param in [adapter.lora_A, adapter.lora_B]:
        if param.grad is not None:
            total_grad += param.grad.detach().abs().sum().item()
            total_elements += param.grad.numel()

    if total_elements == 0:
        return 0.0
    return total_grad / total_elements


def compute_singular_value_sensitivity(adapter: LoRAAdapter) -> float:
    """Compute singular-value-based sensitivity as rank concentration.

    Args:
        adapter: LoRAAdapter to analyze.

    Returns:
        Ratio of top singular value to sum of all singular values (in [0, 1]).
    """
    W = adapter.effective_weight().detach()
    # Use float32 for SVD stability
    W = W.float()
    try:
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
    except Exception:
        return 0.0

    total = S.sum().item()
    if total <= 0.0:
        return 0.0
    return S[0].item() / total


def allocate_ranks_uniform(n_adapters: int, config: RankAllocationConfig) -> list[int]:
    """Distribute total_rank_budget evenly across n_adapters.

    Each adapter receives at least min_rank and at most max_rank.
    The sum of returned ranks is <= total_rank_budget.

    Args:
        n_adapters: Number of adapters to allocate ranks to.
        config: RankAllocationConfig with budget and bounds.

    Returns:
        List of n_adapters integer ranks.
    """
    if n_adapters == 0:
        return []

    base = config.total_rank_budget // n_adapters
    base = max(config.min_rank, min(config.max_rank, base))

    ranks = [base] * n_adapters

    # Distribute remainder while respecting max_rank
    remainder = config.total_rank_budget - sum(ranks)
    for i in range(n_adapters):
        if remainder <= 0:
            break
        can_add = config.max_rank - ranks[i]
        add = min(can_add, remainder)
        ranks[i] += add
        remainder -= add

    return ranks


def allocate_ranks_by_sensitivity(
    sensitivities: list[float],
    config: RankAllocationConfig,
) -> list[int]:
    """Allocate ranks proportionally to sensitivity scores.

    Args:
        sensitivities: Sensitivity score for each adapter (higher = more important).
        config: RankAllocationConfig with budget and bounds.

    Returns:
        List of integer ranks, one per adapter.
    """
    n = len(sensitivities)
    if n == 0:
        return []

    total_sens = sum(sensitivities)

    if total_sens <= 0.0:
        # Fall back to uniform if all sensitivities are zero
        return allocate_ranks_uniform(n, config)

    # Normalize and scale by budget
    fractions = [s / total_sens for s in sensitivities]
    raw_ranks = [f * config.total_rank_budget for f in fractions]

    # Round and enforce bounds
    ranks = [max(config.min_rank, min(config.max_rank, round(r))) for r in raw_ranks]

    # Trim total if it exceeds budget
    while sum(ranks) > config.total_rank_budget:
        # Reduce the adapter with highest rank that is above min_rank
        max_idx = max(
            range(n),
            key=lambda i: (ranks[i], -sensitivities[i]),
        )
        if ranks[max_idx] <= config.min_rank:
            break
        ranks[max_idx] -= 1

    return ranks


def prune_low_rank_components(adapter: LoRAAdapter, threshold: float) -> int:
    """Prune singular values of effective_weight below threshold.

    Reconstructs lora_A and lora_B from the kept SVD components and
    updates the adapter parameters in-place.

    Args:
        adapter: LoRAAdapter to prune.
        threshold: Singular values below this are discarded.

    Returns:
        Number of components pruned (original_rank - kept_rank).
    """
    W = adapter.effective_weight().detach().float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    # Keep components with singular value >= threshold (at least 1)
    mask = S >= threshold
    if not mask.any():
        mask[0] = True  # Always keep at least 1 component

    U_kept = U[:, mask]  # (out_features, k)
    S_kept = S[mask]  # (k,)
    Vh_kept = Vh[mask, :]  # (k, in_features)

    k = S_kept.shape[0]
    original_rank = S.shape[0]
    n_pruned = original_rank - k

    # Reconstruct: W = U_kept @ diag(S_kept) @ Vh_kept
    # Split scaling evenly: A = sqrt(s) * Vh, B = sqrt(s) * U (then scaling applied externally)
    # Since effective_weight = (lora_B @ lora_A) * scaling, we absorb 1/scaling into the factors.
    scaling = adapter.scaling
    if scaling == 0.0:
        scaling = 1.0

    sqrt_S = S_kept.sqrt()  # (k,)
    # new_A shape: (k, in_features), new_B shape: (out_features, k)
    new_A = (sqrt_S.unsqueeze(1) * Vh_kept) / (scaling**0.5)
    new_B = (U_kept * sqrt_S.unsqueeze(0)) / (scaling**0.5)

    # Cast back to original dtype
    dtype = adapter.lora_A.dtype
    device = adapter.lora_A.device

    with torch.no_grad():
        adapter.lora_A.data = new_A.to(dtype=dtype, device=device)
        adapter.lora_B.data = new_B.to(dtype=dtype, device=device)
        adapter.rank = k

    return n_pruned


class RankAllocator:
    """Manages sensitivity-based rank allocation across multiple LoRA adapters.

    Args:
        config: RankAllocationConfig controlling allocation behavior.
    """

    def __init__(self, config: RankAllocationConfig) -> None:
        self.config = config
        self._adapters: dict[str, LoRAAdapter] = {}

    def register_adapter(self, name: str, adapter: LoRAAdapter) -> None:
        """Register a named LoRA adapter for tracking.

        Args:
            name: Unique name for the adapter.
            adapter: LoRAAdapter instance to register.
        """
        self._adapters[name] = adapter

    def compute_sensitivities(self) -> dict[str, float]:
        """Compute sensitivity scores for all registered adapters.

        Uses the method specified in config.sensitivity_method.

        Returns:
            Dict mapping adapter name to sensitivity score.
        """
        method = self.config.sensitivity_method
        result: dict[str, float] = {}

        for name, adapter in self._adapters.items():
            if method == "gradient":
                result[name] = compute_gradient_sensitivity(adapter)
            elif method == "singular_value":
                result[name] = compute_singular_value_sensitivity(adapter)
            else:  # "uniform" or unknown
                result[name] = 1.0

        return result

    def reallocate(self) -> dict[str, int]:
        """Compute sensitivities and return a new rank allocation plan.

        Does not resize adapters in-place; just returns the allocation plan.

        Returns:
            Dict mapping adapter name to proposed new rank.
        """
        sensitivities = self.compute_sensitivities()
        names = list(sensitivities.keys())
        sens_values = [sensitivities[n] for n in names]

        ranks = allocate_ranks_by_sensitivity(sens_values, self.config)
        return dict(zip(names, ranks))
