"""Acceptance heuristics for speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AcceptanceStats:
    accepted: int
    proposed: int

    @property
    def rate(self) -> float:
        return 0.0 if self.proposed == 0 else self.accepted / self.proposed


def accept_prefix(draft_tokens: torch.Tensor, target_tokens: torch.Tensor) -> int:
    """Return the longest accepted speculative prefix length."""
    if draft_tokens.dim() != 1 or target_tokens.dim() != 1:
        raise ValueError("draft_tokens and target_tokens must be 1D")
    accepted = 0
    for draft, target in zip(draft_tokens.tolist(), target_tokens.tolist()):
        if draft != target:
            break
        accepted += 1
    return accepted


def acceptance_stats(drafts: list[torch.Tensor], targets: list[torch.Tensor]) -> AcceptanceStats:
    """Aggregate speculative acceptance over multiple sequences."""
    if len(drafts) != len(targets):
        raise ValueError("drafts and targets must have the same length")
    accepted = 0
    proposed = 0
    for draft, target in zip(drafts, targets):
        accepted += accept_prefix(draft, target)
        proposed += draft.numel()
    return AcceptanceStats(accepted=accepted, proposed=proposed)


def acceptance_mask(draft_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask over accepted draft positions."""
    accepted = accept_prefix(draft_tokens, target_tokens)
    mask = torch.zeros_like(draft_tokens, dtype=torch.bool)
    mask[:accepted] = True
    return mask
