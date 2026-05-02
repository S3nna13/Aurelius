"""Argmax / top-k beam selection from verifier scores.

Used when auxiliary **verifier** models or heuristics assign a scalar score to
each partial hypothesis during beam search (Li et al., "Verify-and-Edit",
style decoders).  This module is intentionally tiny: it only normalises tensor
shapes and applies ``argmax`` / ``topk`` without silent coercion.
"""

from __future__ import annotations

import torch


class BeamVerifierSelector:
    """Select the best beam index from a score tensor."""

    @staticmethod
    def select_best(scores: torch.Tensor) -> torch.Tensor:
        """Return ``argmax`` over the last dimension.

        ``scores`` may be ``[K]`` or ``[B, K]``.  Returns shape ``[]`` or ``[B]``.
        """
        if scores.dim() not in (1, 2):
            raise ValueError(f"scores must be [K] or [B,K], got {tuple(scores.shape)}")
        if not torch.isfinite(scores).all():
            raise ValueError("scores must be finite (no NaN/Inf)")
        return torch.argmax(scores, dim=-1)

    @staticmethod
    def select_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
        """Return indices of the top-``k`` beams (last dimension, descending)."""
        if k < 1:
            raise ValueError("k must be >= 1")
        if scores.dim() not in (1, 2):
            raise ValueError(f"scores must be [K] or [B,K], got {tuple(scores.shape)}")
        if not torch.isfinite(scores).all():
            raise ValueError("scores must be finite (no NaN/Inf)")
        if scores.dim() == 1:
            if k > scores.shape[0]:
                raise ValueError("k cannot exceed number of beams")
            return torch.topk(scores, k=k, dim=-1).indices
        if k > scores.shape[-1]:
            raise ValueError("k cannot exceed number of beams")
        return torch.topk(scores, k=k, dim=-1).indices


__all__ = ["BeamVerifierSelector"]
