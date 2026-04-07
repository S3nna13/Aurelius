# src/inference/turboquant/compressor.py
"""TurboQuant two-stage KV cache compressor.

Stage 1 (PolarQuant): Random rotation + Lloyd-Max quantization.
Stage 2 (QJL): Gaussian sketch of the quantization residual.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .polar_quant import PolarQuant, PolarQuantState
from .qjl import QJLSketch


@dataclass
class CompressedKV:
    """Two-stage compressed representation of a single key or value tensor."""
    # Stage 1: PolarQuant
    polar_state: PolarQuantState
    # Stage 2: QJL on residual
    residual_signs: torch.Tensor   # (..., sketch_dim), int8
    residual_norms: torch.Tensor   # (...), float32


class TurboQuantCompressor:
    """Two-stage KV compressor: PolarQuant + QJL.

    Args:
        dim: Head dimension (head_dim from model config).
        n_codes: Lloyd-Max codebook size (default 256 = 8-bit).
        sketch_dim: QJL sketch dimension.
        seed: Seed for reproducible rotation and sketch matrices.
    """

    def __init__(
        self,
        dim: int,
        n_codes: int = 256,
        sketch_dim: int = 64,
        seed: int = 42,
    ) -> None:
        self.polar = PolarQuant(dim=dim, n_codes=n_codes, seed=seed)
        self.qjl = QJLSketch(dim=dim, sketch_dim=sketch_dim, seed=seed + 1)

    def compress(self, x: torch.Tensor) -> CompressedKV:
        """Compress a KV tensor through both stages.

        Args:
            x: Tensor of shape (..., dim).

        Returns:
            CompressedKV with polar_state and residual sketch.
        """
        polar_state, residual = self.polar.compress(x)
        signs, norms = self.qjl.compress_keys(residual)
        return CompressedKV(
            polar_state=polar_state,
            residual_signs=signs,
            residual_norms=norms,
        )

    def decompress(self, ckv: CompressedKV) -> torch.Tensor:
        """Approximate reconstruction from compressed state.

        Note: Reconstruction is approximate (lossy compression).
        For attention computation, use estimate_attention() instead.

        Returns:
            Reconstructed tensor of shape (..., dim).
        """
        # Stage 1 reconstruction (the main approximation)
        x_hat = self.polar.decompress(ckv.polar_state)
        # Residual from Stage 2 is stored as sketch — cannot fully recover
        # Return Stage 1 reconstruction only
        return x_hat
