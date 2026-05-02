"""LLM watermarking via green/red token list bias (Kirchenbauer et al. 2023).

Embeds detectable signals into generated text by biasing token selection
toward a deterministically-chosen "green list" based on the previous token.
Detection works by computing a z-score on the fraction of green tokens.
"""

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class WatermarkConfig:
    """Configuration for watermark generation and detection."""

    vocab_size: int = 50257
    delta: float = 2.0
    gamma: float = 0.25
    seeding_scheme: str = "hash"
    key: int = 42


def get_green_list(
    prev_token: int,
    vocab_size: int,
    gamma: float,
    key: int,
) -> list[int]:
    """Deterministically partition vocabulary into green and red lists.

    Uses a hash of (prev_token, key) to seed a PRNG, then randomly permutes
    the vocabulary and returns the first gamma fraction as the green list.

    Args:
        prev_token: The previous token id used as context for hashing.
        vocab_size: Total vocabulary size.
        gamma: Fraction of vocabulary to assign to the green list.
        key: Secret integer key mixed into the hash.

    Returns:
        Sorted list of green token ids (length ≈ gamma * vocab_size).
    """
    raw = f"{key}:{prev_token}".encode()
    hash_int = int(hashlib.sha256(raw).hexdigest(), 16)

    rng = torch.Generator()
    rng.manual_seed(hash_int % (2**63))
    perm = torch.randperm(vocab_size, generator=rng)

    n_green = max(1, int(vocab_size * gamma))
    return sorted(perm[:n_green].tolist())


def apply_watermark_bias(
    logits: Tensor,
    green_list: list[int],
    delta: float,
) -> Tensor:
    """Add delta to logits of green-list tokens.

    Args:
        logits: 1-D logit tensor of shape (vocab_size,).
        green_list: List of token ids that belong to the green list.
        delta: Additive bias applied to green-list logits.

    Returns:
        Modified logits tensor with same shape as input.
    """
    out = logits.clone()
    if green_list:
        indices = torch.tensor(green_list, dtype=torch.long, device=logits.device)
        out[indices] += delta
    return out


def detect_watermark_score(
    token_ids: list[int],
    vocab_size: int,
    gamma: float,
    key: int,
) -> tuple[float, float]:
    """Compute z-score for watermark detection over a token sequence.

    For each consecutive pair (token_ids[i-1], token_ids[i]) the function
    checks whether token_ids[i] falls in the green list seeded by token_ids[i-1].
    The z-score measures how far the observed green count deviates from the
    null-hypothesis expectation under uniform sampling.

    Args:
        token_ids: Sequence of token ids to analyse.
        vocab_size: Total vocabulary size.
        gamma: Expected green fraction under the null hypothesis.
        key: Secret key used for green list generation.

    Returns:
        Tuple of (z_score, green_fraction).
    """
    T = len(token_ids) - 1  # number of testable positions
    if T <= 0:
        return (0.0, 0.0)

    green_count = 0
    for i in range(1, len(token_ids)):
        green_set = set(get_green_list(token_ids[i - 1], vocab_size, gamma, key))
        if token_ids[i] in green_set:
            green_count += 1

    green_fraction = green_count / T
    expected = T * gamma
    std = math.sqrt(T * gamma * (1.0 - gamma))
    z_score = (green_count - expected) / (std + 1e-8)

    return (z_score, green_fraction)


class WatermarkLogitProcessor:
    """Logit processor that applies watermark bias during generation."""

    def __init__(self, config: WatermarkConfig) -> None:
        self.config = config

    def __call__(self, logits: Tensor, prev_token: int) -> Tensor:
        """Apply green-list bias to logits.

        Args:
            logits: 1-D tensor of shape (vocab_size,).
            prev_token: The previous generated token id.

        Returns:
            Modified logits with delta added to green-list positions.
        """
        green_list = get_green_list(
            prev_token,
            self.config.vocab_size,
            self.config.gamma,
            self.config.key,
        )
        return apply_watermark_bias(logits, green_list, self.config.delta)


class WatermarkDetector:
    """Detects watermarked text using z-score on green token fraction."""

    def __init__(self, config: WatermarkConfig, z_threshold: float = 4.0) -> None:
        self.config = config
        self.z_threshold = z_threshold

    def detect(self, token_ids: list[int]) -> dict[str, float]:
        """Compute watermark detection statistics for a single sequence.

        Args:
            token_ids: List of token ids comprising the sequence.

        Returns:
            Dict with keys "z_score", "green_fraction", "is_watermarked".
        """
        z_score, green_fraction = detect_watermark_score(
            token_ids,
            self.config.vocab_size,
            self.config.gamma,
            self.config.key,
        )
        return {
            "z_score": z_score,
            "green_fraction": green_fraction,
            "is_watermarked": float(z_score > self.z_threshold),
        }

    def batch_detect(self, sequences: list[list[int]]) -> list[dict[str, float]]:
        """Run detection on a batch of sequences.

        Args:
            sequences: List of token-id sequences.

        Returns:
            List of detection dicts, one per sequence.
        """
        return [self.detect(seq) for seq in sequences]


def compute_watermark_strength(
    original_logits: Tensor,
    watermarked_logits: Tensor,
) -> float:
    """Measure the mean absolute bias introduced by watermarking.

    Args:
        original_logits: Logits before watermark bias, shape (..., vocab_size).
        watermarked_logits: Logits after watermark bias, same shape.

    Returns:
        Mean absolute difference as a Python float.
    """
    return (watermarked_logits - original_logits).abs().mean().item()
