"""PolarQuant: Stage 1 of TurboQuant KV cache compression (arXiv:2504.19874).

Applies a random orthogonal rotation, per-vector min-max normalization,
and Lloyd-Max quantization to KV cache vectors. Returns quantized state
and residual for Stage 2 (QJL).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .lloyd_max import compute_lloyd_max_codebook


@dataclass
class PolarQuantState:
    """Compressed representation from PolarQuant.

    All tensors stored for dequantization:
      codes:  integer indices into the Lloyd-Max codebook
      mins:   per-vector minimum values (before normalization)
      maxs:   per-vector maximum values (before normalization)
      Q:      the random orthogonal rotation matrix (shared across all vectors)
    """

    codes: torch.Tensor  # shape (..., d), dtype=int64 — codebook indices
    mins: torch.Tensor  # shape (..., 1) — per-vector mins
    maxs: torch.Tensor  # shape (..., 1) — per-vector maxs
    Q: torch.Tensor  # shape (d, d) — rotation matrix


class PolarQuant:
    """PolarQuant compressor for a fixed vector dimension.

    Args:
        dim: Feature dimension of the vectors to compress.
        n_codes: Number of Lloyd-Max quantization levels (default 256 = 8-bit).
        seed: Random seed for the rotation matrix.
    """

    def __init__(self, dim: int, n_codes: int = 256, seed: int = 42) -> None:
        self.dim = dim
        self.n_codes = n_codes

        # Precompute the random orthogonal rotation matrix via QR decomposition
        gen = torch.Generator()
        gen.manual_seed(seed)
        raw = torch.randn(dim, dim, generator=gen)
        Q, _ = torch.linalg.qr(raw)
        self.Q = Q  # (dim, dim), orthogonal

        # Lloyd-Max codebook
        self.codebook = compute_lloyd_max_codebook(n_codes)  # (n_codes,)

    def compress(self, x: torch.Tensor) -> tuple[PolarQuantState, torch.Tensor]:
        """Compress KV cache vectors.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            (state, residual) where residual = x - decompress(state).
        """
        orig_dtype = x.dtype
        x_f = x.float()

        # Step 1: Apply random orthogonal rotation
        # x_rot[..., j] = sum_k x[..., k] * Q[k, j]
        x_rot = x_f @ self.Q.to(x_f.device)  # (..., dim)

        # Step 2: Per-vector min-max normalization to [0, 1]
        mins = x_rot.min(dim=-1, keepdim=True).values  # (..., 1)
        maxs = x_rot.max(dim=-1, keepdim=True).values  # (..., 1)
        scale = (maxs - mins).clamp(min=1e-8)
        x_norm = (x_rot - mins) / scale  # (..., dim) in [0, 1]

        # Step 3: Nearest-centroid quantization
        # codes[i] = argmin_j |x_norm[i] - codebook[j]|
        cb = self.codebook.to(x_norm.device)  # (n_codes,)
        # Broadcast: (..., dim, 1) vs (n_codes,) -> (..., dim, n_codes)
        dists = (x_norm.unsqueeze(-1) - cb).abs()
        codes = dists.argmin(dim=-1)  # (..., dim), int64

        state = PolarQuantState(codes=codes, mins=mins, maxs=maxs, Q=self.Q)

        # Step 4: Compute residual in original (unrotated) space
        x_reconstructed = self.decompress(state).to(orig_dtype)
        residual = x.to(orig_dtype) - x_reconstructed

        return state, residual

    def decompress(self, state: PolarQuantState) -> torch.Tensor:
        """Reconstruct approximate x from compressed state.

        Args:
            state: PolarQuantState from compress().

        Returns:
            Approximate x of the same shape as the original input.
        """
        cb = self.codebook.to(state.codes.device)

        # Look up centroids for each code
        x_hat_norm = cb[state.codes].float()  # (..., dim) in [0, 1]

        # Denormalize: undo min-max scaling
        scale = (state.maxs - state.mins).clamp(min=1e-8)
        x_hat_rot = x_hat_norm * scale + state.mins  # (..., dim) in original rotated space

        # Undo rotation: x_hat = x_hat_rot @ Q^T  (Q is orthogonal so Q^-1 = Q^T)
        Q = state.Q.to(x_hat_rot.device)
        x_hat = x_hat_rot @ Q.T  # (..., dim)

        return x_hat
