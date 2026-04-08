"""Curriculum-style corpus mixing utilities for pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CorpusSource:
    name: str
    weight: float
    temperature: float = 1.0
    max_fraction: float = 1.0


def normalize_mix_weights(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize non-negative mixing weights."""
    if weights.dim() != 1:
        raise ValueError("weights must be 1D")
    if torch.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = weights.sum()
    if total <= eps:
        raise ValueError("weights must have positive sum")
    return weights / total


def temperature_mix(weights: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature to a weight vector before normalization."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    adjusted = weights.pow(1.0 / temperature)
    return normalize_mix_weights(adjusted)


def source_probabilities(sources: list[CorpusSource]) -> dict[str, float]:
    """Compute normalized source probabilities with per-source temperatures."""
    if not sources:
        return {}
    raw = torch.tensor([source.weight for source in sources], dtype=torch.float32)
    temps = torch.tensor([source.temperature for source in sources], dtype=torch.float32)
    if torch.any(temps <= 0):
        raise ValueError("All source temperatures must be positive")
    adjusted = raw.pow(1.0 / temps)
    probs = normalize_mix_weights(adjusted)
    capped = torch.minimum(probs, torch.tensor([source.max_fraction for source in sources]))
    capped = normalize_mix_weights(capped)
    return {source.name: prob.item() for source, prob in zip(sources, capped)}


def allocate_source_tokens(sources: list[CorpusSource], total_tokens: int) -> dict[str, int]:
    """Allocate integer token budgets across sources."""
    if total_tokens < 0:
        raise ValueError(f"total_tokens must be non-negative, got {total_tokens}")
    probs = source_probabilities(sources)
    names = [source.name for source in sources]
    allocation: dict[str, int] = {}
    remaining = total_tokens
    for index, name in enumerate(names):
        if index == len(names) - 1:
            allocation[name] = remaining
        else:
            share = int(round(total_tokens * probs[name]))
            share = min(share, remaining)
            allocation[name] = share
            remaining -= share
    return allocation
