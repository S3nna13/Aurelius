"""Verification helpers for Medusa-style draft decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VerificationResult:
    accepted_prefix: torch.Tensor
    accepted_length: int
    rejected_at: int | None


def verify_draft_tokens(
    base_tokens: torch.Tensor, draft_tokens: torch.Tensor
) -> VerificationResult:
    """Accept the longest matching prefix between base and draft proposals."""
    if base_tokens.dim() != 1 or draft_tokens.dim() != 1:
        raise ValueError("base_tokens and draft_tokens must be 1D")
    max_len = min(base_tokens.numel(), draft_tokens.numel())
    accepted = 0
    for index in range(max_len):
        if base_tokens[index].item() != draft_tokens[index].item():
            return VerificationResult(
                accepted_prefix=draft_tokens[:index],
                accepted_length=index,
                rejected_at=index,
            )
        accepted += 1
    return VerificationResult(
        accepted_prefix=draft_tokens[:accepted],
        accepted_length=accepted,
        rejected_at=None if accepted == draft_tokens.numel() else accepted,
    )


def acceptance_rate(results: list[VerificationResult]) -> float:
    """Mean accepted fraction over a batch of verification results."""
    if not results:
        return 0.0
    numer = sum(result.accepted_length for result in results)
    denom = sum(
        max(result.accepted_prefix.numel(), result.accepted_length, 1) for result in results
    )
    return numer / denom


def truncate_after_rejection(tokens: torch.Tensor, result: VerificationResult) -> torch.Tensor:
    """Keep only the accepted Medusa prefix."""
    if tokens.dim() != 1:
        raise ValueError("tokens must be 1D")
    return tokens[: result.accepted_length]
