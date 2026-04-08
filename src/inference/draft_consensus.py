"""Consensus helpers across multiple draft decoders."""

from __future__ import annotations

from collections import Counter

import torch


def majority_token(tokens: torch.Tensor) -> int:
    """Return the modal token id from a 1D tensor of proposals."""
    if tokens.dim() != 1 or tokens.numel() == 0:
        raise ValueError("tokens must be a non-empty 1D tensor")
    return Counter(tokens.tolist()).most_common(1)[0][0]


def consensus_token(tokens: torch.Tensor) -> int | None:
    """Return the unanimous token id if all proposals agree."""
    if tokens.dim() != 1 or tokens.numel() == 0:
        raise ValueError("tokens must be a non-empty 1D tensor")
    first = int(tokens[0].item())
    return first if torch.all(tokens == first) else None


def consensus_rate(proposals: torch.Tensor) -> torch.Tensor:
    """Fraction of time steps with full draft consensus.

    `proposals` has shape `(n_drafts, seq_len)`.
    """
    if proposals.dim() != 2:
        raise ValueError("proposals must be 2D [drafts, seq_len]")
    if proposals.size(0) == 0 or proposals.size(1) == 0:
        return torch.tensor(0.0)
    unanimous = (proposals == proposals[:1]).all(dim=0)
    return unanimous.float().mean()
