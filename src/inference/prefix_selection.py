"""Helpers for selecting accepted prefixes in draft decoding."""

from __future__ import annotations

import torch


def longest_common_prefix(a: torch.Tensor, b: torch.Tensor) -> int:
    """Length of the common token prefix between two 1D sequences."""
    if a.dim() != 1 or b.dim() != 1:
        raise ValueError("a and b must be 1D")
    matched = 0
    for left, right in zip(a.tolist(), b.tolist()):
        if left != right:
            break
        matched += 1
    return matched


def select_prefix(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Keep only the prefix of `candidate` that matches `reference`."""
    return candidate[: longest_common_prefix(candidate, reference)]


def prefix_match_mask(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Boolean mask over matching prefix positions."""
    prefix_len = longest_common_prefix(candidate, reference)
    mask = torch.zeros_like(candidate, dtype=torch.bool)
    mask[:prefix_len] = True
    return mask
