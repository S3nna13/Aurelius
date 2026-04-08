"""Prefix language modeling objective.

Bidirectional attention over a prefix region, causal attention over the suffix.
This allows the model to fully attend within the prefix (like BERT) while
maintaining autoregressive generation in the suffix (like GPT).
"""

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class PrefixLMConfig:
    """Configuration for prefix-LM training."""

    min_prefix_fraction: float = 0.1  # minimum prefix as fraction of sequence
    max_prefix_fraction: float = 0.9  # maximum prefix as fraction of sequence
    mask_prefix_labels: bool = True  # if True, set prefix labels to -100 (don't compute loss on prefix)


def make_prefix_lm_mask(seq_len: int, prefix_len: int) -> torch.Tensor:
    """Build the attention mask for prefix-LM.

    The mask is additive (added to attention logits before softmax):
    - 0 where attention is allowed
    - -inf where attention is blocked

    Rules:
    - Prefix tokens attend to ALL prefix tokens (bidirectional within prefix)
    - Suffix tokens attend to all prefix tokens + earlier suffix tokens (causal in suffix)
    - No token attends to tokens after it in the suffix

    Args:
        seq_len: Total sequence length.
        prefix_len: Number of prefix tokens.

    Returns:
        (seq_len, seq_len) float tensor with 0.0 and -inf values.
    """
    # Start with all blocked
    mask = torch.full((seq_len, seq_len), float("-inf"))

    # Prefix attends to all prefix (bidirectional)
    mask[:prefix_len, :prefix_len] = 0.0

    # Suffix: causal (attends to all prefix + prior suffix tokens)
    for i in range(prefix_len, seq_len):
        mask[i, : i + 1] = 0.0

    return mask


def prepare_prefix_lm_batch(
    input_ids: torch.Tensor,
    cfg: PrefixLMConfig | None = None,
    prefix_len: int | None = None,
) -> dict[str, torch.Tensor]:
    """Prepare a batch for prefix-LM training.

    Args:
        input_ids: (B, S) token IDs.
        cfg: PrefixLMConfig (used for random prefix length if prefix_len not given).
        prefix_len: Fixed prefix length. If None, sample randomly per cfg.

    Returns:
        Dict with:
            "input_ids": (B, S) -- unchanged
            "labels": (B, S) -- input_ids shifted by 1 with -100 on prefix positions
            "attention_mask": (S, S) -- prefix-LM attention mask
    """
    B, S = input_ids.shape
    cfg = cfg or PrefixLMConfig()

    if prefix_len is None:
        min_p = max(1, int(S * cfg.min_prefix_fraction))
        max_p = min(S - 1, int(S * cfg.max_prefix_fraction))
        prefix_len = torch.randint(min_p, max_p + 1, ()).item()

    # Labels: shift right (next-token prediction)
    labels = torch.cat([input_ids[:, 1:], torch.full((B, 1), -100, dtype=input_ids.dtype)], dim=1)

    # Mask prefix positions in labels
    if cfg.mask_prefix_labels:
        labels[:, :prefix_len] = -100

    attn_mask = make_prefix_lm_mask(S, prefix_len)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}
