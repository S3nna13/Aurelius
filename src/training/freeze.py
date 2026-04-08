"""Layer freezing utilities for selective parameter training."""

from __future__ import annotations

from fnmatch import fnmatch

import torch.nn as nn


def _matches_any(name: str, patterns: list[str]) -> bool:
    """Return True if *name* matches any of the fnmatch *patterns*."""
    return any(fnmatch(name, p) for p in patterns)


def freeze_layers(model: nn.Module, patterns: list[str]) -> int:
    """Freeze parameters whose names match ANY of the given fnmatch patterns.

    Args:
        model: The PyTorch module to modify.
        patterns: List of fnmatch-style patterns (e.g. ``"layers.0.*"``).

    Returns:
        Count of frozen parameter tensors.
    """
    count = 0
    for name, param in model.named_parameters():
        if _matches_any(name, patterns):
            param.requires_grad = False
            count += 1
    return count


def unfreeze_layers(model: nn.Module, patterns: list[str]) -> int:
    """Unfreeze (set ``requires_grad=True``) parameters matching patterns.

    Args:
        model: The PyTorch module to modify.
        patterns: List of fnmatch-style patterns.

    Returns:
        Count of unfrozen parameter tensors.
    """
    count = 0
    for name, param in model.named_parameters():
        if _matches_any(name, patterns):
            param.requires_grad = True
            count += 1
    return count


def freeze_except(model: nn.Module, patterns: list[str]) -> int:
    """Freeze ALL parameters EXCEPT those matching any pattern.

    Args:
        model: The PyTorch module to modify.
        patterns: List of fnmatch-style patterns for parameters to keep trainable.

    Returns:
        Count of frozen parameter tensors.
    """
    count = 0
    for name, param in model.named_parameters():
        if _matches_any(name, patterns):
            param.requires_grad = True
        else:
            param.requires_grad = False
            count += 1
    return count


def get_trainable_param_count(model: nn.Module) -> dict[str, int]:
    """Return tensor counts for trainable, frozen, and total parameters.

    Returns:
        Dictionary with keys ``"trainable"``, ``"frozen"``, and ``"total"``,
        each holding the number of parameter *tensors* (not elements).
    """
    trainable = 0
    frozen = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable += 1
        else:
            frozen += 1
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
