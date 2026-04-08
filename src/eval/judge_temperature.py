"""Temperature-scaling helpers for judge score distributions."""

from __future__ import annotations

import torch


def scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by a positive temperature."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return logits / temperature


def tempered_distribution(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax distribution after temperature scaling."""
    return torch.softmax(scale_logits(logits, temperature), dim=-1)


def distribution_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Entropy of the tempered distribution."""
    probs = tempered_distribution(logits, temperature)
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()


def calibrated_top1(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Top-1 class after temperature scaling."""
    return tempered_distribution(logits, temperature).argmax(dim=-1)
