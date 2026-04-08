"""Acceptance metrics over sets of draft paths."""

from __future__ import annotations

import torch


def path_acceptance_lengths(draft_paths: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Accepted prefix length for each draft path against one target path."""
    lengths = []
    for path in draft_paths:
        matched = 0
        for draft_token, target_token in zip(path.tolist(), target.tolist()):
            if draft_token != target_token:
                break
            matched += 1
        lengths.append(matched)
    return torch.tensor(lengths, dtype=torch.long)


def best_accepting_path(draft_paths: list[torch.Tensor], target: torch.Tensor) -> int:
    """Index of the path with the longest accepted prefix."""
    if not draft_paths:
        raise ValueError("draft_paths must be non-empty")
    lengths = path_acceptance_lengths(draft_paths, target)
    return int(torch.argmax(lengths).item())


def mean_acceptance_length(draft_paths: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Mean accepted prefix length across all draft paths."""
    if not draft_paths:
        return torch.tensor(0.0)
    return path_acceptance_lengths(draft_paths, target).float().mean()
