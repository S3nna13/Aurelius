"""Alignment helpers between draft and target token streams."""

from __future__ import annotations

import torch


def token_alignment(draft: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Boolean alignment over the overlapping prefix of two 1D token streams."""
    if draft.dim() != 1 or target.dim() != 1:
        raise ValueError("draft and target must be 1D")
    length = min(draft.numel(), target.numel())
    return draft[:length] == target[:length]


def alignment_rate(draft: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fraction of aligned tokens over the overlap region."""
    aligned = token_alignment(draft, target)
    if aligned.numel() == 0:
        return torch.tensor(0.0)
    return aligned.float().mean()


def first_misalignment(draft: torch.Tensor, target: torch.Tensor) -> int | None:
    """Index of first misaligned token, or None if fully aligned over overlap."""
    aligned = token_alignment(draft, target)
    mismatches = (~aligned).nonzero(as_tuple=False)
    if mismatches.numel() == 0:
        return None
    return int(mismatches[0].item())

