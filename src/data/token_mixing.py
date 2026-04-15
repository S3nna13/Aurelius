"""Token-level mixing utilities for multi-domain LLM pre-training."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenMixingConfig:
    """Configuration for token-level domain mixing."""

    domain_weights: Dict[str, float]
    buffer_size: int = 1000
    sequence_length: int = 512
    pack_sequences: bool = True
    eos_token_id: int = 0


# ---------------------------------------------------------------------------
# Weight utilities
# ---------------------------------------------------------------------------

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize a dict of weights so they sum to 1.0.

    Raises:
        ValueError: if any weight is <= 0.
    """
    for key, val in weights.items():
        if val <= 0:
            raise ValueError(
                f"Weight for domain '{key}' must be > 0, got {val}"
            )
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def sample_domain(weights: Dict[str, float], rng: random.Random) -> str:
    """Sample a domain name according to the given weights.

    Args:
        weights: Mapping from domain name to (unnormalized) weight.
        rng:     A ``random.Random`` instance for reproducibility.

    Returns:
        The sampled domain name.
    """
    normalized = normalize_weights(weights)
    domains = list(normalized.keys())
    probs = [normalized[d] for d in domains]

    r = rng.random()
    cumulative = 0.0
    for domain, prob in zip(domains, probs):
        cumulative += prob
        if r < cumulative:
            return domain
    # Fallback (floating-point edge case)
    return domains[-1]


# ---------------------------------------------------------------------------
# Sequence packing
# ---------------------------------------------------------------------------

def pack_sequences(
    sequences: List[List[int]],
    seq_len: int,
    eos_id: int,
) -> List[List[int]]:
    """Concatenate sequences with EOS tokens between them and chunk into fixed windows.

    Sequences are joined as: seq0 + [eos_id] + seq1 + [eos_id] + ...
    The resulting stream is then sliced into non-overlapping chunks of exactly
    ``seq_len`` tokens.  Any partial last chunk is discarded (no padding).

    Args:
        sequences: List of token-id lists.
        seq_len:   Output chunk length.
        eos_id:    Token id used as document separator.

    Returns:
        List of fixed-length chunks.
    """
    if not sequences:
        return []

    # Build the full token stream
    stream: List[int] = []
    for seq in sequences:
        stream.extend(seq)
        stream.append(eos_id)

    # Chunk into seq_len windows; discard the partial tail
    chunks: List[List[int]] = []
    for start in range(0, len(stream) - seq_len + 1, seq_len):
        chunks.append(stream[start : start + seq_len])

    return chunks


# ---------------------------------------------------------------------------
# Attention mask
# ---------------------------------------------------------------------------

def create_attention_mask(token_ids: List[int], eos_id: int) -> List[int]:
    """Return a causal attention mask for the given token sequence.

    Implementation: returns a list of 1s of length ``len(token_ids)``.
    (Simpler causal masking — all positions are attended to.)

    Args:
        token_ids: Sequence of token ids.
        eos_id:    Token id that marks document boundaries (kept for API
                   compatibility; not used in this simplified implementation).

    Returns:
        List of 1s with the same length as ``token_ids``.
    """
    return [1] * len(token_ids)


# ---------------------------------------------------------------------------
# TokenMixer
# ---------------------------------------------------------------------------

class TokenMixer:
    """Sample sequences from multiple domains according to mixing weights."""

    def __init__(
        self,
        domain_sequences: Dict[str, List[List[int]]],
        config: TokenMixingConfig,
    ) -> None:
        self.domain_sequences = domain_sequences
        self.config = config

    def get_batch(
        self,
        batch_size: int,
        rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """Sample ``batch_size`` sequences by domain weights.

        Domains with no sequences are skipped during sampling.

        Returns:
            {
                "input_ids":     List[List[int]],
                "domain_labels": List[str],
                "lengths":       List[int],
            }
        """
        if rng is None:
            rng = random.Random()

        # Filter out empty domains so we can still produce a batch
        active_weights = {
            domain: w
            for domain, w in self.config.domain_weights.items()
            if domain in self.domain_sequences and len(self.domain_sequences[domain]) > 0
        }

        input_ids: List[List[int]] = []
        domain_labels: List[str] = []
        lengths: List[int] = []

        if not active_weights:
            # No data available — return empty batch
            return {"input_ids": input_ids, "domain_labels": domain_labels, "lengths": lengths}

        for _ in range(batch_size):
            domain = sample_domain(active_weights, rng)
            seqs = self.domain_sequences[domain]
            seq = rng.choice(seqs)
            input_ids.append(seq)
            domain_labels.append(domain)
            lengths.append(len(seq))

        return {
            "input_ids": input_ids,
            "domain_labels": domain_labels,
            "lengths": lengths,
        }

    def get_domain_stats(self) -> Dict[str, Dict[str, float]]:
        """Return per-domain statistics.

        Returns:
            {domain: {"n_sequences": float, "mean_length": float, "total_tokens": float}}
        """
        stats: Dict[str, Dict[str, float]] = {}
        for domain, seqs in self.domain_sequences.items():
            n = len(seqs)
            total = sum(len(s) for s in seqs)
            mean = total / n if n > 0 else 0.0
            stats[domain] = {
                "n_sequences": float(n),
                "mean_length": float(mean),
                "total_tokens": float(total),
            }
        return stats

    def compute_mixing_weights_from_tokens(self) -> Dict[str, float]:
        """Compute mixing weights proportional to total tokens per domain.

        Returns:
            Normalized weight dict (sums to 1.0).
        """
        token_counts: Dict[str, float] = {}
        for domain, seqs in self.domain_sequences.items():
            token_counts[domain] = float(sum(len(s) for s in seqs))

        total = sum(token_counts.values())
        if total == 0:
            # Uniform weights as fallback
            n = len(token_counts)
            return {d: 1.0 / n for d in token_counts} if n > 0 else {}

        return {d: cnt / total for d, cnt in token_counts.items()}


# ---------------------------------------------------------------------------
# Standalone token-count weight computation (mirrors TokenMixer method)
# ---------------------------------------------------------------------------

def compute_mixing_weights_from_tokens(
    domain_sequences: Dict[str, List[List[int]]],
) -> Dict[str, float]:
    """Compute mixing weights proportional to total tokens per domain.

    Args:
        domain_sequences: Mapping from domain name to list of token-id sequences.

    Returns:
        Normalized weight dict (sums to 1.0). Falls back to uniform weights
        when total token count is zero.
    """
    token_counts: Dict[str, float] = {
        domain: float(sum(len(s) for s in seqs))
        for domain, seqs in domain_sequences.items()
    }
    total = sum(token_counts.values())
    if total == 0:
        n = len(token_counts)
        return {d: 1.0 / n for d in token_counts} if n > 0 else {}
    return {d: cnt / total for d, cnt in token_counts.items()}


# ---------------------------------------------------------------------------
# Interleaving
# ---------------------------------------------------------------------------

def interleave_sequences(
    sequences_a: List[int],
    sequences_b: List[int],
    ratio: float,
) -> List[int]:
    """Interleave two token sequences.

    For every ``ceil(1 / ratio)`` tokens from *b*, insert 1 token from *a*.
    Remaining tokens from either sequence are appended at the end.

    Args:
        sequences_a: Tokens from stream A (sparse / higher-value stream).
        sequences_b: Tokens from stream B (denser stream).
        ratio:       Fraction of *a* tokens relative to *b* tokens.
                     E.g. ratio=0.5 means for every 2 tokens from b we take 1 from a.

    Returns:
        Interleaved token list.
    """
    step_b = math.ceil(1.0 / ratio) if ratio > 0 else len(sequences_b) + 1

    result: List[int] = []
    idx_a = 0
    idx_b = 0

    while idx_a < len(sequences_a) and idx_b < len(sequences_b):
        # Take `step_b` tokens from b
        take_b = min(step_b, len(sequences_b) - idx_b)
        result.extend(sequences_b[idx_b : idx_b + take_b])
        idx_b += take_b

        # Take 1 token from a
        result.append(sequences_a[idx_a])
        idx_a += 1

    # Append remaining tokens from either stream
    result.extend(sequences_b[idx_b:])
    result.extend(sequences_a[idx_a:])

    return result


# ---------------------------------------------------------------------------
# Distribution utilities
# ---------------------------------------------------------------------------

def compute_domain_distribution(domain_labels: List[str]) -> Dict[str, float]:
    """Compute the empirical frequency of each domain in a list of labels.

    Args:
        domain_labels: List of domain name strings.

    Returns:
        Dict mapping each domain to its fraction (sums to 1.0).
        Returns an empty dict for an empty input.
    """
    if not domain_labels:
        return {}

    counts: Dict[str, int] = {}
    for label in domain_labels:
        counts[label] = counts.get(label, 0) + 1

    total = len(domain_labels)
    return {domain: cnt / total for domain, cnt in counts.items()}
