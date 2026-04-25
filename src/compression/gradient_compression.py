"""Gradient compression: TOP_K, RANDOM_K, THRESHOLD, SIGN_SGD strategies."""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import torch


class GradCompressMethod(str, Enum):
    TOP_K = "top_k"
    RANDOM_K = "random_k"
    THRESHOLD = "threshold"
    SIGN_SGD = "sign_sgd"


@dataclass
class GradCompressionConfig:
    method: GradCompressMethod = GradCompressMethod.TOP_K
    sparsity: float = 0.99   # fraction of weights to zero out (0.99 → keep 1%)
    threshold: float = 0.01  # for THRESHOLD method


class GradientCompressor:
    """Compress and decompress gradient tensors.

    ``compress`` returns ``(values, indices)`` for the surviving elements;
    the rest are treated as zero.  ``decompress`` reconstructs a dense
    tensor of the original shape.
    """

    def __init__(self, config: GradCompressionConfig | None = None) -> None:
        self.config = config or GradCompressionConfig()
        self._rng = random.Random(42)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self, grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (values, flat_indices) of non-zeroed gradient elements."""
        flat = grad.reshape(-1)
        method = self.config.method

        if method == GradCompressMethod.TOP_K:
            return self._top_k(flat)

        elif method == GradCompressMethod.RANDOM_K:
            return self._random_k(flat)

        elif method == GradCompressMethod.THRESHOLD:
            return self._threshold(flat)

        elif method == GradCompressMethod.SIGN_SGD:
            return self._sign_sgd(flat)

        else:
            indices = torch.arange(flat.numel())
            return flat.clone(), indices

    def decompress(
        self,
        values: torch.Tensor,
        indices: torch.Tensor,
        original_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Reconstruct a dense gradient tensor from (values, indices)."""
        n = 1
        for d in original_shape:
            n *= d
        out = torch.zeros(n, dtype=values.dtype)
        if indices.numel() > 0:
            out[indices] = values
        return out.reshape(original_shape)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _keep_count(self, numel: int) -> int:
        """Number of elements to keep given the sparsity config."""
        return max(1, int(round(numel * (1.0 - self.config.sparsity))))

    def _top_k(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = self._keep_count(flat.numel())
        abs_vals = flat.abs()
        # topk returns sorted=True by default
        _, indices = torch.topk(abs_vals, k)
        values = flat[indices]
        return values, indices

    def _random_k(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = self._keep_count(flat.numel())
        perm = torch.randperm(flat.numel())
        indices = perm[:k]
        values = flat[indices]
        return values, indices

    def _threshold(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = flat.abs() > self.config.threshold
        indices = mask.nonzero(as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            # Fall back: keep the single largest element
            indices = torch.tensor([flat.abs().argmax().item()])
        values = flat[indices]
        return values, indices

    def _sign_sgd(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        signs = flat.sign()
        indices = torch.arange(flat.numel())
        return signs, indices
