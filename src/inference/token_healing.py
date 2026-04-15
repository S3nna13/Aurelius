"""Token healing: fix tokenization boundary artifacts by resampling at the rollback position.

Token healing backs up by one (or more) tokens at the boundary of a prompt prefix, then
re-generates those tokens — fixing artifacts that arise when a prompt is split mid-token.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Callable, List, Tuple


@dataclass
class TokenHealingConfig:
    """Configuration for token healing."""
    n_rollback_tokens: int = 1
    top_k_candidates: int = 10
    temperature: float = 1.0
    max_new_tokens: int = 50


def get_token_prefix_logit_bias(vocab_size: int, allowed_token_ids: List[int]) -> torch.Tensor:
    """Return a float bias tensor of shape (vocab_size,).

    Allowed token positions get 0.0; all others get -1e9 (effectively blocked).

    Args:
        vocab_size: Total vocabulary size V.
        allowed_token_ids: List of token IDs that are permitted.

    Returns:
        Float tensor of shape (vocab_size,).
    """
    bias = torch.full((vocab_size,), -1e9, dtype=torch.float)
    if allowed_token_ids:
        allowed = torch.tensor(allowed_token_ids, dtype=torch.long)
        bias[allowed] = 0.0
    return bias


def apply_logit_bias(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Add bias to the last dimension of logits.

    Args:
        logits: Tensor of any shape (..., V).
        bias: Tensor of shape (V,) — broadcast-compatible with logits last dim.

    Returns:
        Tensor of the same shape as logits with bias added.
    """
    return logits + bias


def greedy_extend(model_fn: Callable[[torch.Tensor], torch.Tensor],
                  token_ids: torch.Tensor,
                  n_steps: int) -> torch.Tensor:
    """Greedily decode n_steps additional tokens.

    Args:
        model_fn: Callable that takes (B, T) ids and returns (B, T, V) logits.
        token_ids: Starting token ids of shape (B, T).
        n_steps: Number of additional tokens to generate.

    Returns:
        Token ids tensor of shape (B, T + n_steps).
    """
    current = token_ids
    for _ in range(n_steps):
        logits = model_fn(current)          # (B, T_cur, V)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
        current = torch.cat([current, next_token], dim=1)
    return current


class TokenHealer:
    """Performs token healing on a token sequence.

    Token healing backs up n_rollback_tokens tokens, then re-generates them
    using the model — fixing tokenization boundary artifacts.
    """

    def __init__(self,
                 model_fn: Callable[[torch.Tensor], torch.Tensor],
                 config: TokenHealingConfig | None = None) -> None:
        """Initialize the healer.

        Args:
            model_fn: Callable that takes (B, T) int64 ids and returns (B, T, V) logits.
            config: TokenHealingConfig; defaults to TokenHealingConfig() if None.
        """
        self.model_fn = model_fn
        self.config = config if config is not None else TokenHealingConfig()

    def rollback_tokens(self,
                        token_ids: torch.Tensor,
                        n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split token_ids into a prefix and the removed suffix.

        Args:
            token_ids: (B, T) int64 tensor.
            n: Number of tokens to remove from the end.

        Returns:
            (prefix, removed) where prefix is token_ids[:, :-n] and
            removed is token_ids[:, -n:].

        Raises:
            ValueError: If n >= T (not enough tokens to roll back).
        """
        T = token_ids.shape[1]
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if n >= T:
            raise ValueError(f"Cannot roll back {n} tokens from sequence of length {T}")
        prefix = token_ids[:, :-n]
        removed = token_ids[:, -n:]
        return prefix, removed

    def get_continuation_candidates(self,
                                    prefix_ids: torch.Tensor,
                                    removed_ids: torch.Tensor,
                                    top_k: int) -> torch.Tensor:
        """Run the model on prefix_ids and return top_k candidate next tokens.

        Args:
            prefix_ids: (B, T_prefix) int64 tensor.
            removed_ids: (B, n_removed) int64 tensor (not used in forward pass,
                         retained for API symmetry / future constrained use).
            top_k: How many candidate token IDs to return.

        Returns:
            (B, top_k) int64 tensor of top-k token IDs at the rollback position.
        """
        with torch.no_grad():
            logits = self.model_fn(prefix_ids)  # (B, T_prefix, V)
        position_logits = logits[:, -1, :]       # (B, V)
        # Clamp top_k to vocab size
        vocab_size = position_logits.shape[-1]
        k = min(top_k, vocab_size)
        _, top_ids = position_logits.topk(k, dim=-1)  # (B, k)
        return top_ids

    def heal(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Full token healing: roll back 1 token, pick argmax replacement.

        The returned sequence has the same length as the input (prefix + 1 healed token).

        Args:
            token_ids: (B, T) int64 tensor, T >= 2.

        Returns:
            (B, T) int64 tensor with the last token replaced by the healed token.
        """
        n = self.config.n_rollback_tokens
        prefix, removed = self.rollback_tokens(token_ids, n)

        # Run model on prefix to get next-position logits
        with torch.no_grad():
            logits = self.model_fn(prefix)  # (B, T-n, V)
        position_logits = logits[:, -1, :]  # (B, V)

        # Argmax — no temperature for heal()
        healed_token = position_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

        # Re-attach: prefix + healed token + remaining removed tokens (if n > 1)
        if n == 1:
            healed_seq = torch.cat([prefix, healed_token], dim=1)
        else:
            # Keep first healed token, then re-attach remaining n-1 removed tokens
            healed_seq = torch.cat([prefix, healed_token, removed[:, 1:]], dim=1)

        return healed_seq

    def heal_and_continue(self,
                          token_ids: torch.Tensor,
                          n_new: int) -> torch.Tensor:
        """Heal the token sequence then greedily decode n_new additional tokens.

        Args:
            token_ids: (B, T) int64 tensor, T >= 2.
            n_new: Number of new tokens to decode after healing.

        Returns:
            (B, T + n_new) int64 tensor.
        """
        healed = self.heal(token_ids)          # (B, T)
        extended = greedy_extend(self.model_fn, healed, n_new)  # (B, T + n_new)
        return extended
