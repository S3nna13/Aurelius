"""Temperature control helpers for MoE router logits."""

from __future__ import annotations

import torch


def apply_router_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale router logits by temperature."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return logits / temperature


def router_probabilities(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute router probabilities under a temperature."""
    return torch.softmax(apply_router_temperature(logits, temperature), dim=-1)


def router_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute mean token-level routing entropy."""
    probs = router_probabilities(logits, temperature)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return entropy.mean()


def sharpened_top1(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Return top-1 expert indices after temperature scaling."""
    probs = router_probabilities(logits, temperature)
    return probs.argmax(dim=-1)
