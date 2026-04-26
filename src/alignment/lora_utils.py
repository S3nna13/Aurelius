"""LoRA utilities: rank selection, adapter merging, and multi-adapter composition."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# 1. Intrinsic rank estimation via SVD
# ---------------------------------------------------------------------------


def estimate_intrinsic_rank(weight: Tensor, threshold: float = 0.99) -> int:
    """Estimate the intrinsic rank of a weight matrix via SVD.

    Computes the minimum rank r such that the top-r singular values capture
    `threshold` fraction of total variance (sum of squared singular values).

    Args:
        weight: 2-D weight tensor (out_features, in_features).
        threshold: Fraction of variance to capture (default 0.99).

    Returns:
        Minimum rank r satisfying the threshold.
    """
    with torch.no_grad():
        # torch.linalg.svdvals returns singular values in descending order
        sigma = torch.linalg.svdvals(weight.float())
        variance = sigma**2
        total_variance = variance.sum()
        cumulative = torch.cumsum(variance, dim=0)
        # Find smallest r where cumulative variance >= threshold * total
        passing = (cumulative / total_variance) >= threshold
        # passing is a bool tensor; first True index + 1 = rank
        r = int(passing.nonzero(as_tuple=False)[0].item()) + 1
    return r


# ---------------------------------------------------------------------------
# 2. LoRAAdapterInfo dataclass
# ---------------------------------------------------------------------------


@dataclass
class LoRAAdapterInfo:
    """Metadata and weight tensors for a single LoRA adapter."""

    name: str
    rank: int
    alpha: float
    A: Tensor  # shape (rank, in_features)
    B: Tensor  # shape (out_features, rank)

    @property
    def scale(self) -> float:
        """Scaling factor alpha / rank."""
        return self.alpha / self.rank

    @property
    def delta_weight(self) -> Tensor:
        """Effective weight update: scale * (B @ A), shape (out_features, in_features)."""
        return self.scale * (self.B @ self.A)


# ---------------------------------------------------------------------------
# 3. Adapter merging
# ---------------------------------------------------------------------------


def merge_lora_adapters(
    adapters: list[LoRAAdapterInfo],
    weights: list[float] | None = None,
) -> Tensor:
    """Weighted sum of LoRA adapter delta_weights.

    Args:
        adapters: List of LoRAAdapterInfo objects. All must share the same
                  (out_features, in_features) shape.
        weights: Per-adapter scalar weights. If None, uniform 1/n is used.

    Returns:
        Merged delta weight tensor of shape (out_features, in_features).
    """
    n = len(adapters)
    if n == 0:
        raise ValueError("adapters list must not be empty")

    if weights is None:
        weights = [1.0 / n] * n

    if len(weights) != n:
        raise ValueError(f"len(weights)={len(weights)} must equal len(adapters)={n}")

    merged = sum(w * adapter.delta_weight for w, adapter in zip(weights, adapters))
    return merged  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 4. LoRALinear
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    The base weight is frozen; only lora_A and lora_B are trainable.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: LoRA rank.
        alpha: LoRA scaling factor; scale = alpha / rank.
        dropout: Dropout probability applied to input before LoRA path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Frozen base weight
        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        base_out = F.linear(x, self.weight)
        lora_out = self.scale * F.linear(self.dropout(x), self.lora_B @ self.lora_A)
        return base_out + lora_out

    def merge_weights(self) -> Tensor:
        """Return base weight + LoRA update, suitable for inference without overhead."""
        return self.weight + self.scale * (self.lora_B @ self.lora_A)

    def get_adapter_info(self, name: str = "") -> LoRAAdapterInfo:
        """Snapshot current adapter state as a LoRAAdapterInfo."""
        return LoRAAdapterInfo(
            name=name,
            rank=self.rank,
            alpha=self.alpha,
            A=self.lora_A.detach(),
            B=self.lora_B.detach(),
        )


# ---------------------------------------------------------------------------
# 5. MultiLoRALinear
# ---------------------------------------------------------------------------


class _LoRASlot(nn.Module):
    """A single (lora_A, lora_B) adapter slot stored as a Module for nn.ModuleList."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))


class MultiLoRALinear(nn.Module):
    """Linear layer supporting multiple switchable LoRA adapter slots.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: LoRA rank shared across all slots.
        alpha: LoRA scaling factor; scale = alpha / rank.
        n_adapters: Number of adapter slots.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        n_adapters: int = 2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / rank
        self.active_adapter: int = 0

        # Frozen base weight
        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)

        # n_adapters slots, each with its own (lora_A, lora_B)
        self.adapters = nn.ModuleList(
            [_LoRASlot(in_features, out_features, rank) for _ in range(n_adapters)]
        )

    def forward(self, x: Tensor) -> Tensor:
        slot: _LoRASlot = self.adapters[self.active_adapter]  # type: ignore[assignment]
        base_out = F.linear(x, self.weight)
        lora_out = self.scale * F.linear(x, slot.lora_B @ slot.lora_A)
        return base_out + lora_out

    def switch_adapter(self, idx: int) -> None:
        """Switch the active adapter slot to idx."""
        if idx < 0 or idx >= len(self.adapters):
            raise IndexError(f"adapter index {idx} out of range [0, {len(self.adapters) - 1}]")
        self.active_adapter = idx


# ---------------------------------------------------------------------------
# 6. Rank distribution analysis
# ---------------------------------------------------------------------------


def analyze_lora_rank_distribution(
    model: nn.Module,
    threshold: float = 0.99,
) -> dict[str, int]:
    """Estimate intrinsic rank for every nn.Linear layer in a model.

    Walks all named modules, finds nn.Linear instances, and calls
    estimate_intrinsic_rank on each weight matrix.

    Args:
        model: The PyTorch model to analyze.
        threshold: Variance threshold forwarded to estimate_intrinsic_rank.

    Returns:
        Dict mapping module name -> estimated intrinsic rank.
    """
    result: dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            result[name] = estimate_intrinsic_rank(module.weight, threshold)
    return result
