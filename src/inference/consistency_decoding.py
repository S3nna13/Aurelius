"""Consistency decoding (Wang et al. 2023 "Self-Consistency").

Generate N independent completions with temperature sampling, then aggregate
per-token via majority vote to produce a single consensus output.  An agreement
score measures how consistently the samples agree across token positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ConsistencyConfig:
    """Configuration for consistency decoding."""

    n_samples: int = 8
    temperature: float = 0.7
    top_k: int = 50
    aggregation: str = "majority_vote"  # "majority_vote" | "weighted_vote"


# ---------------------------------------------------------------------------
# Standalone sampling helpers
# ---------------------------------------------------------------------------


def sample_with_temperature(
    logits: Tensor,
    temperature: float,
    top_k: int = 0,
) -> Tensor:
    """Sample next token ids from logits with temperature scaling and top-k filtering.

    Args:
        logits: Raw logits of shape (B, V).
        temperature: Scaling temperature > 0.  Higher = more random.
        top_k: If > 0, zero out all but the top-k logits before sampling.

    Returns:
        Tensor of shape (B,) with sampled token ids.
    """
    # Temperature scaling
    scaled = logits / temperature

    # Optional top-k masking
    if top_k > 0:
        k = min(top_k, scaled.size(-1))
        # Values below the k-th largest get set to -inf
        topk_vals, _ = torch.topk(scaled, k, dim=-1)
        threshold = topk_vals[:, -1].unsqueeze(-1)  # (B, 1)
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

    probs = F.softmax(scaled, dim=-1)
    # multinomial returns (B, 1); squeeze to (B,)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# Greedy / temperature autoregressive decoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_decode(model, input_ids: Tensor, max_new_tokens: int) -> Tensor:
    """Autoregressive greedy decoding (argmax at every step).

    Args:
        model: Model whose forward pass returns (_, logits, _).
                logits shape: (B, T, V).
        input_ids: (B, S) prompt token ids.
        max_new_tokens: Number of tokens to generate.

    Returns:
        (B, max_new_tokens) tensor of generated token ids (prompt excluded).
    """
    generated: list[Tensor] = []
    cur_ids = input_ids

    for _ in range(max_new_tokens):
        _, logits, _ = model(cur_ids)
        next_logits = logits[:, -1, :]          # (B, V)
        next_token = next_logits.argmax(dim=-1)  # (B,)
        generated.append(next_token)
        cur_ids = torch.cat([cur_ids, next_token.unsqueeze(-1)], dim=-1)

    return torch.stack(generated, dim=-1)  # (B, max_new_tokens)


@torch.no_grad()
def temperature_decode(
    model,
    input_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tensor:
    """Autoregressive decoding with temperature sampling and optional top-k.

    Args:
        model: Model whose forward pass returns (_, logits, _).
        input_ids: (B, S) prompt token ids.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering; 0 means no filtering.

    Returns:
        (B, max_new_tokens) tensor of generated token ids (prompt excluded).
    """
    generated: list[Tensor] = []
    cur_ids = input_ids

    for _ in range(max_new_tokens):
        _, logits, _ = model(cur_ids)
        next_logits = logits[:, -1, :]  # (B, V)
        next_token = sample_with_temperature(next_logits, temperature, top_k)
        generated.append(next_token)
        cur_ids = torch.cat([cur_ids, next_token.unsqueeze(-1)], dim=-1)

    return torch.stack(generated, dim=-1)  # (B, max_new_tokens)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def majority_vote(sequences: List[Tensor]) -> Tensor:
    """Per-position majority vote across a list of sample tensors.

    Args:
        sequences: List of N tensors, each of shape (B,) or (B, T).

    Returns:
        Tensor of the same shape as each input element, containing the
        most-common token at each position across the N samples.
    """
    if not sequences:
        raise ValueError("sequences must be non-empty")

    # Stack to (N, ...) then find mode along dim=0
    stacked = torch.stack(sequences, dim=0)  # (N, B) or (N, B, T)
    # torch.mode returns (values, indices); mode along dim=0
    voted, _ = torch.mode(stacked, dim=0)
    return voted


def compute_sequence_agreement(sequences: List[Tensor]) -> float:
    """Mean pairwise exact-match agreement across all sample pairs.

    For each pair (i, j) with i < j, computes the fraction of positions
    where sequence i and sequence j are identical, then averages over all
    pairs.

    Args:
        sequences: List of N tensors, each of shape (B,) or (B, T).

    Returns:
        Float in [0, 1].  1.0 means all sequences are identical.
    """
    n = len(sequences)
    if n < 2:
        return 1.0

    total_agreement = 0.0
    n_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            eq = (sequences[i] == sequences[j]).float()
            total_agreement += eq.mean().item()
            n_pairs += 1

    return total_agreement / n_pairs


# ---------------------------------------------------------------------------
# ConsistencyDecoder
# ---------------------------------------------------------------------------


class ConsistencyDecoder:
    """Consistency decoding via majority vote over independently sampled outputs.

    Generates config.n_samples completions with temperature sampling, then
    aggregates per token position via majority vote.

    Args:
        model: Model whose forward returns (_, logits, _) with logits (B, T, V).
        config: ConsistencyConfig controlling sampling and aggregation.
    """

    def __init__(self, model, config: ConsistencyConfig | None = None) -> None:
        self.model = model
        self.config = config or ConsistencyConfig()

    def _generate_samples(self, input_ids: Tensor, max_new_tokens: int) -> List[Tensor]:
        """Generate config.n_samples independent completions.

        Returns list of n_samples tensors, each of shape (B, max_new_tokens).
        """
        samples: List[Tensor] = []
        for _ in range(self.config.n_samples):
            seq = temperature_decode(
                self.model,
                input_ids,
                max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
            )
            samples.append(seq)
        return samples

    def decode(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Generate n_samples completions and apply majority vote per token position.

        Args:
            input_ids: (B, S) prompt ids.
            max_new_tokens: Number of tokens to generate.

        Returns:
            (B, max_new_tokens) majority-voted output.
        """
        samples = self._generate_samples(input_ids, max_new_tokens)
        # Each sample is (B, max_new_tokens); vote per position
        # Transpose to list of (B,) slices? No — vote across samples per step.
        # majority_vote expects list of tensors with same shape; we pass (B, T) tensors.
        return majority_vote(samples)

    def decode_with_score(
        self, input_ids: Tensor, max_new_tokens: int
    ) -> tuple[Tensor, float]:
        """Generate samples, apply majority vote, and compute agreement score.

        Args:
            input_ids: (B, S) prompt ids.
            max_new_tokens: Number of tokens to generate.

        Returns:
            Tuple of (voted_output, agreement_score) where voted_output is
            (B, max_new_tokens) and agreement_score is a float in [0, 1].
        """
        samples = self._generate_samples(input_ids, max_new_tokens)
        voted = majority_vote(samples)
        score = compute_sequence_agreement(samples)
        return voted, float(score)
