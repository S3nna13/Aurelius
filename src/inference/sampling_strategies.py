"""Sampling strategies for autoregressive decoding in AureliusTransformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingConfig:
    """Configuration for token sampling."""

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    typical_p: float = 1.0


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by temperature.

    When temperature==0 the distribution collapses to argmax: a one-hot-like
    tensor is returned where the argmax position is 0 and all others are -inf.
    """
    if temperature == 0.0:
        argmax_idx = logits.argmax(dim=-1, keepdim=True)
        result = torch.full_like(logits, float("-inf"))
        result.scatter_(-1, argmax_idx, 0.0)
        return result
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the top-k logits; set the rest to -inf.

    k=0 disables filtering (all logits are kept as-is).
    """
    if k == 0:
        return logits
    vocab_size = logits.size(-1)
    k = min(k, vocab_size)
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    # Minimum value among the top-k
    threshold = top_k_values[..., -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering.

    Tokens outside the smallest set whose cumulative probability exceeds *p*
    are set to -inf.  p=1.0 means no filtering.
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold.
    # We shift by 1 so that the token that pushes the cumsum over p is kept.
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))

    # Scatter back to original ordering
    result = torch.empty_like(logits)
    result.scatter_(-1, sorted_indices, sorted_logits)
    return result


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-p filtering.

    Tokens whose probability is below ``min_p * max_prob`` are set to -inf.
    min_p=0.0 disables filtering.
    """
    if min_p == 0.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * max_prob
    return logits.masked_fill(probs < threshold, float("-inf"))


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits of previously seen tokens.

    Tokens that have already appeared in *input_ids* have their logit divided
    by *penalty* (values >1 discourage repetition).  penalty=1.0 = no effect.
    """
    if penalty == 1.0:
        return logits

    result = logits.clone()
    unique_ids = input_ids.unique()
    for token_id in unique_ids:
        idx = token_id.long()
        if result[idx] < 0:
            result[idx] = result[idx] * penalty
        else:
            result[idx] = result[idx] / penalty
    return result


def apply_typical_p(logits: torch.Tensor, mass: float) -> torch.Tensor:
    """Locally typical sampling.

    Computes the conditional entropy H of the distribution and keeps only
    tokens whose |log p - H| is smallest, summing their probabilities until
    *mass* is covered.  mass=1.0 disables filtering.
    """
    if mass >= 1.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)

    # Conditional entropy: H = -sum(p * log p)
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

    # Deviation from entropy
    deviation = (log_probs - entropy).abs()

    # Sort tokens by deviation (ascending) and accumulate probability mass
    sorted_dev, sorted_indices = torch.sort(deviation, dim=-1)
    sorted_probs = probs.gather(-1, sorted_indices)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens up to the point where cumulative mass >= mass
    # Shift by 1 so the token that pushes over the threshold is also kept
    to_remove = cumulative - sorted_probs >= mass
    sorted_logits = logits.gather(-1, sorted_indices)
    sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))

    # Scatter back to original ordering
    result = torch.empty_like(logits)
    result.scatter_(-1, sorted_indices, sorted_logits)
    return result


# ---------------------------------------------------------------------------
# Unified sampling entry point
# ---------------------------------------------------------------------------

def sample_token(
    logits: torch.Tensor,
    config: SamplingConfig,
    input_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply all configured filters then draw one token via multinomial sampling.

    Filter order: temperature → top_k → top_p → min_p → repetition_penalty
    (typical_p is applied after min_p when configured).

    Returns a (1,) int64 tensor containing the sampled token id.
    """
    # Work on a flat 1-D logit vector
    logits = logits.view(-1).float()

    logits = apply_temperature(logits, config.temperature)
    logits = apply_top_k(logits, config.top_k)
    logits = apply_top_p(logits, config.top_p)
    logits = apply_min_p(logits, config.min_p)

    if input_ids is not None and config.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, input_ids.view(-1), config.repetition_penalty)

    if config.typical_p < 1.0:
        logits = apply_typical_p(logits, config.typical_p)

    # Safety: if all logits are -inf (shouldn't happen in practice), fall back
    # to uniform.
    if torch.all(logits == float("-inf")):
        logits = torch.zeros_like(logits)

    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    return token


# ---------------------------------------------------------------------------
# Decoder classes
# ---------------------------------------------------------------------------

class SamplingDecoder:
    """Autoregressive decoder that uses sample_token at each step."""

    def __init__(
        self,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        config: SamplingConfig,
    ) -> None:
        """
        Args:
            model_fn: A callable that takes an (1, T) int64 token tensor and
                      returns (1, T, vocab_size) or (T, vocab_size) logits.
            config:   Sampling configuration.
        """
        self.model_fn = model_fn
        self.config = config

    def decode(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Autoregressive decode starting from *prompt_ids*.

        Args:
            prompt_ids:     1-D or (1, T) int64 prompt tensor.
            max_new_tokens: Number of new tokens to generate.

        Returns:
            1-D int64 tensor containing the full sequence
            (prompt + generated tokens).
        """
        ids = prompt_ids.view(-1).long()

        for _ in range(max_new_tokens):
            logits = self.model_fn(ids.unsqueeze(0))
            # Accept (1, T, V) or (T, V) or (V,)
            logits = logits.view(-1, logits.size(-1))[-1]  # last token logits → (V,)
            next_token = sample_token(logits, self.config, input_ids=ids)
            ids = torch.cat([ids, next_token])

        return ids

    def decode_batch(
        self,
        prompt_ids: torch.Tensor,
        n_sequences: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Generate *n_sequences* independent continuations of *prompt_ids*.

        Args:
            prompt_ids:     1-D or (1, T) int64 prompt tensor (shared prompt).
            n_sequences:    Number of independent sequences to generate.
            max_new_tokens: Max new tokens per sequence.

        Returns:
            (n_sequences, T) int64 tensor where T = prompt_len + max_new_tokens.
            Shorter sequences (if any) are padded with 0.
        """
        sequences = []
        for _ in range(n_sequences):
            seq = self.decode(prompt_ids, max_new_tokens)
            sequences.append(seq)

        # Pad to the same length
        max_len = max(s.size(0) for s in sequences)
        padded = torch.zeros(n_sequences, max_len, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, : seq.size(0)] = seq

        return padded
