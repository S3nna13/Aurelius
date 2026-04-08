"""Attention-based prompt compression to reduce KV cache size.

Long prompts are expensive: they inflate KV cache memory linearly with
sequence length.  This module provides two complementary strategies:

1. **AttentionBasedCompressor** — token-level pruning using attention scores
   captured via forward hooks.  Tokens that receive little attention from other
   tokens are discarded; the most *attended-to* tokens are kept.

2. **SelectiveContextCompressor** — chunk-level pruning.  Tokens are grouped
   into fixed-size chunks; each chunk is scored by its average attention mass,
   and low-scoring chunks are removed wholesale.  This preserves local
   coherence better than individual token removal.

References:
    - Selective Context (Li et al., 2023): context compression via self-information.
    - LLMLingua (Jiang et al., 2023): coarse-to-fine prompt compression.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_attention_weights(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Run a forward pass and collect attention weight tensors via hooks.

    Looks for any ``nn.MultiheadAttention`` sub-module whose ``forward``
    returns ``(output, attn_weights)``.  For modules that do not return
    weights natively (e.g. custom attention classes), falls back to
    capturing the *output* activations and treating their mean as a proxy.

    Returns:
        List of attention weight tensors, each shape
        ``(batch, n_heads, seq_len, seq_len)`` or ``(batch, seq_len, seq_len)``.
        May be empty if the model exposes no suitable hooks.
    """
    captured: list[torch.Tensor] = []

    def _hook(module: nn.Module, inp, out):  # noqa: ANN001
        # nn.MultiheadAttention with need_weights=True returns (output, weights)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            weights = out[1]
            if weights is not None and isinstance(weights, torch.Tensor):
                captured.append(weights.detach())

    handles = []
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in handles:
            h.remove()

    return captured


# ---------------------------------------------------------------------------
# AttentionBasedCompressor
# ---------------------------------------------------------------------------

class AttentionBasedCompressor:
    """Compress a prompt by keeping only the top-k% most attended-to tokens.

    Uses the model's own attention patterns to score token importance.
    Tokens that receive high attention from other tokens are most important.

    Args:
        model: Language model (``nn.Module``).
        keep_ratio: Fraction of tokens to keep (default 0.5).
        strategy: One of ``'attention_sum'`` | ``'last_token'`` | ``'uniform'``.

            * ``'attention_sum'``: sum of attention weights *received* by each
              token — how much other tokens collectively look at it.
            * ``'last_token'``: attention weights emitted *from* the last token
              to every earlier token — what the model focuses on for the next
              prediction step.
            * ``'uniform'``: baseline; every token gets score ``1/seq_len``
              (no real compression quality signal).
    """

    def __init__(
        self,
        model: nn.Module,
        keep_ratio: float = 0.5,
        strategy: str = "attention_sum",
    ) -> None:
        if strategy not in {"attention_sum", "last_token", "uniform"}:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose 'attention_sum', 'last_token', or 'uniform'."
            )
        self.model = model
        self.keep_ratio = keep_ratio
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Score each token by importance.

        Args:
            input_ids: ``(1, seq_len)`` integer token IDs.

        Returns:
            ``(seq_len,)`` float32 importance scores (all >= 0).

        Implementation notes:
            Registers forward hooks on any ``nn.MultiheadAttention`` layers
            to capture ``(batch, n_heads, seq_len, seq_len)`` weight tensors.
            If the model has no such layers the scores fall back to a uniform
            distribution so the compressor degrades gracefully.
        """
        seq_len = input_ids.shape[1]

        if self.strategy == "uniform":
            return torch.full((seq_len,), 1.0 / seq_len)

        attn_weights = _collect_attention_weights(self.model, input_ids)

        if not attn_weights:
            # Fallback: run model, use output logit variance as proxy importance
            with torch.no_grad():
                out = self.model(input_ids)
            # out may be a tensor or a tuple/object
            if isinstance(out, torch.Tensor):
                logits = out  # (1, seq_len, vocab) or (1, vocab)
            elif isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                return torch.ones(seq_len) / seq_len

            if logits.dim() == 3:
                # (1, seq_len, vocab) — variance over vocab per position
                scores = logits[0].float().var(dim=-1)  # (seq_len,)
            else:
                scores = torch.ones(seq_len)
            return scores.clamp(min=0.0)

        # Stack: each element is (batch, heads, seq, seq) or (batch, seq, seq)
        # Normalise to (seq_len, seq_len) — average over batch & heads
        pooled_layers: list[torch.Tensor] = []
        for w in attn_weights:
            w = w.float()
            # Drop batch dim
            if w.dim() == 4:
                w = w[0]  # (heads, seq_len, seq_len)
                w = w.mean(dim=0)  # (seq_len, seq_len)
            elif w.dim() == 3:
                w = w[0]  # (seq_len, seq_len)
            # w is now (seq_len, seq_len): w[i, j] = attention from token i to token j
            pooled_layers.append(w)

        avg_attn = torch.stack(pooled_layers).mean(dim=0)  # (seq_len, seq_len)

        if self.strategy == "attention_sum":
            # Column sum: how much total attention each token *receives*
            scores = avg_attn.sum(dim=0)  # (seq_len,)
        else:  # last_token
            # Row of the last token: where does the last position attend?
            scores = avg_attn[-1]  # (seq_len,)

        return scores.clamp(min=0.0)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(
        self,
        input_ids: torch.Tensor,
        n_keep: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the most important tokens.

        Args:
            input_ids: ``(1, seq_len)`` token IDs.
            n_keep: Exact number of tokens to keep.  If ``None``, derived
                from ``keep_ratio * seq_len`` (minimum 1).

        Returns:
            ``(compressed_ids, kept_indices)`` where

            * ``compressed_ids``: ``(1, n_keep)`` token IDs of kept tokens,
              in their *original* left-to-right order.
            * ``kept_indices``: ``(n_keep,)`` original positions (sorted).
        """
        seq_len = input_ids.shape[1]

        if n_keep is None:
            n_keep = max(1, int(self.keep_ratio * seq_len))
        n_keep = min(n_keep, seq_len)

        scores = self.score_tokens(input_ids)  # (seq_len,)

        # Select top-n_keep tokens by score
        _, top_indices = torch.topk(scores, k=n_keep, sorted=True)
        kept_indices, _ = torch.sort(top_indices)  # restore positional order

        compressed_ids = input_ids[:, kept_indices]  # (1, n_keep)
        return compressed_ids, kept_indices

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def compression_ratio(self, original_len: int, compressed_len: int) -> float:
        """Return ``compressed_len / original_len``."""
        return compressed_len / original_len


# ---------------------------------------------------------------------------
# SelectiveContextCompressor
# ---------------------------------------------------------------------------

class SelectiveContextCompressor:
    """Chunk-level prompt compression: keep sentence-length blocks.

    Groups tokens into fixed-size chunks, scores each chunk by the average
    attention mass it receives, and drops the lowest-scoring chunks.

    Args:
        model: Language model (``nn.Module``).
        chunk_size: Tokens per chunk (default 32).
        keep_ratio: Fraction of chunks to retain (default 0.5).
    """

    def __init__(
        self,
        model: nn.Module,
        chunk_size: int = 32,
        keep_ratio: float = 0.5,
    ) -> None:
        self.model = model
        self.chunk_size = chunk_size
        self.keep_ratio = keep_ratio
        self._token_compressor = AttentionBasedCompressor(
            model, keep_ratio=1.0, strategy="attention_sum"
        )

    def compress(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Remove low-scoring chunks and return remaining token IDs.

        Args:
            input_ids: ``(1, seq_len)`` token IDs.

        Returns:
            ``(1, compressed_len)`` token IDs with low-scoring chunks removed.
            Guaranteed to be shorter than or equal to ``input_ids`` in length.
        """
        seq_len = input_ids.shape[1]

        # Score individual tokens
        token_scores = self._token_compressor.score_tokens(input_ids)  # (seq_len,)

        # Partition into chunks and score each chunk
        n_chunks = max(1, (seq_len + self.chunk_size - 1) // self.chunk_size)
        chunk_scores: list[float] = []
        chunk_ranges: list[tuple[int, int]] = []

        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            chunk_score = token_scores[start:end].mean().item()
            chunk_scores.append(chunk_score)
            chunk_ranges.append((start, end))

        # Select top-k chunks
        n_keep_chunks = max(1, int(self.keep_ratio * n_chunks))
        scores_t = torch.tensor(chunk_scores)
        _, top_chunk_indices = torch.topk(scores_t, k=n_keep_chunks, sorted=True)
        kept_chunks, _ = torch.sort(top_chunk_indices)

        # Gather token indices from kept chunks (preserve order)
        kept_token_indices: list[int] = []
        for ci in kept_chunks.tolist():
            start, end = chunk_ranges[ci]
            kept_token_indices.extend(range(start, end))

        if not kept_token_indices:
            # Safety: return at least one token
            kept_token_indices = [0]

        idx_t = torch.tensor(kept_token_indices, dtype=torch.long)
        return input_ids[:, idx_t]  # (1, compressed_len)
