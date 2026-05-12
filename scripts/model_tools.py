"""Model context length helpers."""

from __future__ import annotations

MODEL_CONTEXT_LENGTHS: dict[str, int] = {
    "aurelius-1b": 4096,
    "aurelius-3b": 8192,
    "aurelius-7b": 8192,
    "aurelius-14b": 16384,
    "aurelius-32b": 32768,
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars ≈ 1 token for English text)."""
    return max(1, len(text) // 4)


def context_length(model_name: str) -> int:
    """Return the context length for a given model."""
    return MODEL_CONTEXT_LENGTHS.get(model_name, 4096)
