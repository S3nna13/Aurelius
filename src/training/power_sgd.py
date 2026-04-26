"""
PowerSGD: Low-rank gradient compression via power iteration.
Reference: Vogels et al. 2019, "PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization"
Pure PyTorch implementation — no external dependencies.
"""  # noqa: E501

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class LowRankApproximation:
    """Low-rank matrix decomposition via randomized SVD."""

    def __init__(self, rank: int = 4) -> None:
        self.rank = rank

    def decompose(self, G: Tensor) -> tuple[Tensor, Tensor]:
        """Decompose G (m, n) into P (m, rank) and Q (n, rank) where G ≈ P @ Q.T."""
        m, n = G.shape
        rank = min(self.rank, m, n)
        # Randomized SVD: sketch with random projection
        omega = torch.randn(m, rank, dtype=G.dtype, device=G.device)
        Q, _ = torch.linalg.qr(G.T @ omega)  # Q: (n, rank)
        P = G @ Q  # P: (m, rank)
        return P, Q

    def reconstruct(self, P: Tensor, Q: Tensor) -> Tensor:
        """Reconstruct approximation: G_approx = P @ Q.T."""
        return P @ Q.T

    def compression_ratio(self, original_shape: tuple[int, int]) -> float:
        """Return (m*n) / (m*rank + n*rank)."""
        m, n = original_shape
        rank = self.rank
        return (m * n) / (m * rank + n * rank)


class PowerIteration:
    """Core power iteration for dominant subspace extraction."""

    def __init__(self, rank: int = 4, n_iter: int = 2) -> None:
        self.rank = rank
        self.n_iter = n_iter

    def run(self, G: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute low-rank factors via power iteration.
        Returns P (m, rank), Q (n, rank).
        """
        m, n = G.shape
        rank = min(self.rank, m, n)

        # Step 0: initialize Q with random normal, orthogonalize
        Q = torch.randn(n, rank, dtype=G.dtype, device=G.device)
        Q, _ = torch.linalg.qr(Q)  # Q: (n, rank)

        for _ in range(self.n_iter):
            P = G @ Q  # (m, rank)
            P, _ = torch.linalg.qr(P)
            Q = G.T @ P  # (n, rank)
            Q, _ = torch.linalg.qr(Q)

        # Final P without re-orthogonalization for efficiency
        P = G @ Q  # (m, rank)
        return P, Q

    def reconstruction_error(self, G: Tensor, P: Tensor, Q: Tensor) -> float:
        """Relative Frobenius reconstruction error: ||G - P@Q.T||_F / ||G||_F."""
        G_approx = P @ Q.T
        numerator = torch.norm(G - G_approx, p="fro")
        denominator = torch.norm(G, p="fro")
        if denominator.item() == 0.0:
            return 0.0
        return (numerator / denominator).item()


class GradientCompressor:
    """Compress gradient tensors using PowerIteration low-rank approximation."""

    def __init__(
        self,
        rank: int = 4,
        min_compression_ratio: float = 2.0,
        n_iter: int = 2,
    ) -> None:
        self.rank = rank
        self.min_compression_ratio = min_compression_ratio
        self.n_iter = n_iter
        self._power_iter = PowerIteration(rank=rank, n_iter=n_iter)
        self._lra = LowRankApproximation(rank=rank)

    def compress(self, grad: Tensor) -> tuple[Any, dict]:
        """
        Compress a gradient tensor.
        1D or low compression-ratio tensors are returned unchanged.
        Returns (compressed, metadata).
        """
        if grad.dim() < 2:
            return grad, {"compressed": False}

        original_shape = grad.shape
        # Reshape to 2D
        G = grad.reshape(original_shape[0], -1)
        m, n = G.shape

        ratio = self._lra.compression_ratio((m, n))
        if ratio < self.min_compression_ratio:
            return grad, {"compressed": False}

        P, Q = self._power_iter.run(G)
        error = self._power_iter.reconstruction_error(G, P, Q)

        compressed = {"P": P, "Q": Q}
        metadata = {
            "compressed": True,
            "original_shape": original_shape,
            "rank": self.rank,
            "error": error,
        }
        return compressed, metadata

    def decompress(self, compressed: Any, metadata: dict) -> Tensor:
        """Reconstruct gradient tensor from compressed representation."""
        if not metadata.get("compressed", False):
            return compressed  # already a raw tensor

        P = compressed["P"]
        Q = compressed["Q"]
        original_shape = metadata["original_shape"]

        G_approx = P @ Q.T  # (m, n_flat)
        return G_approx.reshape(original_shape)


class PowerSGDOptimizer:
    """Wraps any optimizer with PowerSGD gradient compression."""

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        compressor: GradientCompressor,
        start_iter: int = 10,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.compressor = compressor
        self.start_iter = start_iter
        self.n_steps: int = 0

    @property
    def state(self) -> dict:
        return self.base_optimizer.state

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def step(self) -> None:
        """Compress gradients (after warmup), decompress, assign back, then step."""
        self.n_steps += 1

        if self.n_steps > self.start_iter:
            for group in self.base_optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    compressed, meta = self.compressor.compress(param.grad.data)
                    approx_grad = self.compressor.decompress(compressed, meta)
                    param.grad.data.copy_(approx_grad)

        self.base_optimizer.step()

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Delegate zero_grad to base optimizer."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


class CompressionStats:
    """Track compression statistics across training steps."""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def record(self, name: str, metadata: dict) -> None:
        """Accumulate per-param stats."""
        self._records.append({"name": name, "metadata": metadata})

    def summary(self) -> dict:
        """Return aggregate statistics."""
        total_compressed = 0
        total_skipped = 0
        compression_ratios: list[float] = []
        reconstruction_errors: list[float] = []

        for rec in self._records:
            meta = rec["metadata"]
            if meta.get("compressed", False):
                total_compressed += 1
                if "rank" in meta:
                    # We don't have original_shape here, but we can use error
                    pass
                if "error" in meta:
                    reconstruction_errors.append(meta["error"])
            else:
                total_skipped += 1

        mean_ratio = (
            sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0.0
        )
        mean_error = (
            sum(reconstruction_errors) / len(reconstruction_errors)
            if reconstruction_errors
            else 0.0
        )

        return {
            "total_params_compressed": total_compressed,
            "total_params_skipped": total_skipped,
            "mean_compression_ratio": mean_ratio,
            "mean_reconstruction_error": mean_error,
        }

    def reset(self) -> None:
        """Clear all recorded stats."""
        self._records = []
