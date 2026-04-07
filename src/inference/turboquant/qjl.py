"""QJL: Quantized JL sketching for KV cache attention (arXiv:2406.03482).

Stage 2 of TurboQuant: compresses the PolarQuant residual into binary signs
and per-vector norms. Enables approximate inner product estimation without
full decompression.

Key: compressed to (signs, norms)
Query: projected full-precision

Estimator: norms * sqrt(pi/2) / m * (signs.float() @ q_proj)
where m = sketch_dim, S ~ N(0,1)

E[estimate] = <key_residual, query>  (unbiased)
"""
from __future__ import annotations

import math

import torch


class QJLSketch:
    """QJL asymmetric inner product sketch.

    Compresses key residuals to binary signs + norms.
    Queries are projected full-precision for asymmetric estimation.

    Args:
        dim: Input feature dimension (must match residual dim).
        sketch_dim: Sketch dimension (higher = more accurate, more memory).
        seed: Random seed for the Gaussian projection matrix S.
    """

    def __init__(self, dim: int, sketch_dim: int, seed: int = 42) -> None:
        self.dim = dim
        self.sketch_dim = sketch_dim

        # S ~ N(0,1) — CRITICAL: Gaussian, not ±1
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.S = torch.randn(sketch_dim, dim, generator=gen)  # (sketch_dim, dim)

    def compress_keys(
        self,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress key residuals to (signs, norms).

        Args:
            residual: Key residual tensor of shape (..., dim).

        Returns:
            signs: Binary sketch (..., sketch_dim), dtype=int8
            norms: Per-vector L2 norms (...,), dtype=float32
        """
        # Project: (..., dim) @ (dim, sketch_dim) = (..., sketch_dim)
        proj = residual.float() @ self.S.to(residual.device).T
        signs = proj.sign().to(torch.int8)                    # binary quantization
        norms = residual.float().norm(dim=-1)                 # (...,) L2 norms
        return signs, norms

    def estimate_attention(
        self,
        signs: torch.Tensor,
        norms: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate key-query inner products without decompressing.

        Asymmetric estimator (arXiv:2406.03482 Theorem 1):
            E[estimate] = <key_residual, query>

        Args:
            signs: Compressed keys (..., sketch_dim), dtype=int8.
            norms: Key L2 norms (...,), dtype=float32.
            query: Query tensor (..._q, dim) — may have different leading dims.

        Returns:
            Inner product estimates of shape (batch, n_q_heads, S_keys)
            or compatible broadcast shape.
        """
        S = self.S.to(query.device)
        m = self.sketch_dim

        # Project query full-precision: (..._q, sketch_dim)
        q_proj = query.float() @ S.T

        # Estimator: norms * sqrt(pi/2) / m * sum(signs * q_proj, dim=-1)
        # signs shape: (..._k, sketch_dim), q_proj shape: (..._q, sketch_dim)
        # dot product over sketch_dim
        dot = (signs.float() * q_proj).sum(dim=-1)            # (..., )
        estimate = norms * math.sqrt(math.pi / 2.0) / m * dot
        return estimate
