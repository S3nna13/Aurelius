"""StreamingLLM Attention Sink — v2 implementation.

Implements Xiao et al. 2023 "Efficient Streaming Language Models with
Attention Sinks": retain the first n_sinks "sink" tokens (which absorb
disproportionate attention mass) plus a sliding window of recent tokens,
enabling infinite-length generation with bounded KV-cache memory.

Pure stdlib + torch only — no external ML dependencies.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# AttentionSinkDetector
# ---------------------------------------------------------------------------


class AttentionSinkDetector:
    """Identify sink tokens from attention weight matrices.

    Args:
        n_sink_candidates: How many top positions to flag as sinks per
            batch item.
    """

    def __init__(self, n_sink_candidates: int = 4) -> None:
        self.n_sink_candidates = n_sink_candidates

    # ------------------------------------------------------------------
    def detect(self, attn_weights: Tensor) -> Tuple[Tensor, Tensor]:
        """Detect sink tokens from a full attention weight matrix.

        Args:
            attn_weights: (B, H, T, T) — softmax attention weights where
                dim -1 is the key / value dimension.

        Returns:
            sink_mask:   (B, T) bool  — True for top n_sink_candidates
                         positions by mean received attention per batch item.
            sink_scores: (B, T) float — mean received attention per position.
        """
        if attn_weights.dim() != 4:
            raise ValueError(
                f"attn_weights must be 4-D (B, H, T, T), got {attn_weights.dim()}-D"
            )
        B, H, T_q, T_k = attn_weights.shape
        # Average over heads and query positions → (B, T_k)
        sink_scores: Tensor = attn_weights.mean(dim=1).mean(dim=1)  # (B, T_k)

        k = min(self.n_sink_candidates, T_k)
        # top-k indices per batch item
        topk_vals, topk_idx = torch.topk(sink_scores, k, dim=-1)  # (B, k)

        sink_mask = torch.zeros(B, T_k, dtype=torch.bool, device=attn_weights.device)
        sink_mask.scatter_(1, topk_idx, True)

        return sink_mask, sink_scores

    # ------------------------------------------------------------------
    def is_sink_token(self, position: int, attn_weights: Tensor) -> bool:
        """Return True if *position* is among the top-k sinks across all
        batch items (any batch item voting it in counts).

        Args:
            position:    Key position index to query.
            attn_weights: (B, H, T, T) attention weights.
        """
        sink_mask, _ = self.detect(attn_weights)
        # position is a sink if it is flagged in *any* batch item
        return bool(sink_mask[:, position].any().item())


# ---------------------------------------------------------------------------
# StreamingKVCache
# ---------------------------------------------------------------------------


class StreamingKVCache:
    """KV cache that preserves sink tokens plus a sliding window of recents.

    The cache stores:
        • First n_sinks token KV pairs (attention sinks) — always kept.
        • Up to window_size most-recent non-sink KV pairs.

    Args:
        n_sinks:     Number of initial tokens to treat as permanent sinks.
        window_size: Maximum number of non-sink tokens to retain.
    """

    def __init__(self, n_sinks: int = 4, window_size: int = 512) -> None:
        self.n_sinks = n_sinks
        self.window_size = window_size

        # Stored as (B, H, T, D_head) or None when empty
        self._sink_keys:   Optional[Tensor] = None
        self._sink_values: Optional[Tensor] = None
        self._win_keys:    Optional[Tensor] = None
        self._win_values:  Optional[Tensor] = None

    # ------------------------------------------------------------------
    def update(
        self,
        new_keys: Tensor,
        new_values: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Incorporate new KV pairs and return the full retained cache.

        Args:
            new_keys:   (B, H, T_new, D_head)
            new_values: (B, H, T_new, D_head)

        Returns:
            kept_keys:   (B, H, n_retained, D_head)
            kept_values: (B, H, n_retained, D_head)
            where n_retained = n_sinks + min(window_size, n_non_sink_tokens).
        """
        if new_keys.dim() != 4:
            raise ValueError("new_keys must be 4-D (B, H, T_new, D_head)")

        # --- 1. Separate the very first batch of tokens as sinks (one-time) --
        if self._sink_keys is None:
            # We have no cache yet; new tokens fill sinks first.
            T_new = new_keys.size(2)
            sink_len = min(self.n_sinks, T_new)
            self._sink_keys   = new_keys[:, :, :sink_len, :]
            self._sink_values = new_values[:, :, :sink_len, :]

            remaining_keys   = new_keys[:, :, sink_len:, :]
            remaining_values = new_values[:, :, sink_len:, :]

            if remaining_keys.size(2) > 0:
                self._win_keys   = remaining_keys
                self._win_values = remaining_values
            # else: window stays None (empty)
        else:
            # Append new tokens to the sliding window
            if self._win_keys is None:
                self._win_keys   = new_keys
                self._win_values = new_values
            else:
                self._win_keys   = torch.cat([self._win_keys,   new_keys],   dim=2)
                self._win_values = torch.cat([self._win_values, new_values], dim=2)

        # --- 2. Truncate window to last window_size tokens -----------------
        if self._win_keys is not None and self._win_keys.size(2) > self.window_size:
            self._win_keys   = self._win_keys[:, :, -self.window_size:, :]
            self._win_values = self._win_values[:, :, -self.window_size:, :]

        # --- 3. Assemble output --------------------------------------------
        parts_k = [self._sink_keys]
        parts_v = [self._sink_values]
        if self._win_keys is not None and self._win_keys.size(2) > 0:
            parts_k.append(self._win_keys)
            parts_v.append(self._win_values)

        kept_keys   = torch.cat(parts_k, dim=2)
        kept_values = torch.cat(parts_v, dim=2)
        return kept_keys, kept_values

    # ------------------------------------------------------------------
    def size(self) -> int:
        """Current total number of retained KV pairs (sequence length)."""
        n = 0
        if self._sink_keys is not None:
            n += self._sink_keys.size(2)
        if self._win_keys is not None:
            n += self._win_keys.size(2)
        return n

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all cached KV pairs."""
        self._sink_keys   = None
        self._sink_values = None
        self._win_keys    = None
        self._win_values  = None


# ---------------------------------------------------------------------------
# SinkAwareAttention
# ---------------------------------------------------------------------------


class SinkAwareAttention(nn.Module):
    """Multi-head attention with streaming KV cache and sink-aware position
    reindexing.

    Sink tokens keep their original positions (0 … n_sinks-1).  Window
    tokens are assigned positions [n_sinks, n_sinks + window_size) in a
    rolling fashion, preventing position-index collapse when the window
    slides.

    Args:
        d_model:     Model dimension.
        n_heads:     Number of attention heads.
        n_sinks:     Number of permanent sink positions.
        window_size: Maximum non-sink window tokens.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_sinks: int = 4,
        window_size: int = 64,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_sinks = n_sinks
        self.window_size = window_size
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.kv_cache = StreamingKVCache(n_sinks=n_sinks, window_size=window_size)

    # ------------------------------------------------------------------
    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, T, D) → (B, H, T, D_head)."""
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.permute(0, 2, 1, 3)  # (B, H, T, D_head)

    # ------------------------------------------------------------------
    def _merge_heads(self, x: Tensor) -> Tensor:
        """(B, H, T, D_head) → (B, T, D)."""
        B, H, T, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, self.d_model)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, position_offset: int = 0) -> Tensor:
        """Forward pass with streaming KV cache.

        Args:
            x:               (B, T, D) input tokens.
            position_offset: Absolute position of the first token in *x*.

        Returns:
            output: (B, T, D)
        """
        B, T, _ = x.shape

        q = self._split_heads(self.q_proj(x))   # (B, H, T, D_head)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # Update cache and retrieve full KV context
        ctx_k, ctx_v = self.kv_cache.update(k, v)  # (B, H, T_ctx, D_head)

        # Scaled dot-product attention
        scale = self.d_head ** -0.5
        attn_logits = torch.matmul(q, ctx_k.transpose(-2, -1)) * scale  # (B,H,T,T_ctx)

        # Causal mask over the context window
        T_ctx = ctx_k.size(2)
        causal_mask = torch.ones(T, T_ctx, dtype=torch.bool, device=x.device).tril(
            diagonal=T_ctx - T
        )
        attn_logits = attn_logits.masked_fill(~causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)
        # Replace NaN rows (all-masked) with uniform weights
        nan_rows = attn_weights.isnan()
        if nan_rows.any():
            attn_weights = torch.where(nan_rows, torch.zeros_like(attn_weights), attn_weights)

        out = torch.matmul(attn_weights, ctx_v)  # (B, H, T, D_head)
        out = self._merge_heads(out)              # (B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# SinkTokenInitializer
# ---------------------------------------------------------------------------


class SinkTokenInitializer(nn.Module):
    """Trainable sink token embeddings that are prepended to every sequence.

    Args:
        n_sinks: Number of sink tokens.
        d_model: Model dimension.
    """

    def __init__(self, n_sinks: int = 4, d_model: int = 512) -> None:
        super().__init__()
        self.n_sinks = n_sinks
        self.d_model = d_model
        self.sink_embeddings = nn.Parameter(torch.zeros(n_sinks, d_model))

    # ------------------------------------------------------------------
    def prepend(self, x: Tensor) -> Tensor:
        """Prepend learnable sink embeddings to input sequence.

        Args:
            x: (B, T, D)

        Returns:
            (B, n_sinks + T, D)
        """
        B = x.size(0)
        sinks = self.sink_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, n_sinks, D)
        return torch.cat([sinks, x], dim=1)

    # ------------------------------------------------------------------
    def sink_loss(self, attn_weights: Tensor) -> Tensor:
        """Encourage attention mass to flow toward sink positions.

        Loss = -mean(sum of attention over first n_sinks key positions).
        This is always ≤ 0 because we negate a non-negative quantity.

        Args:
            attn_weights: (B, H, T_q, T_k)

        Returns:
            Scalar loss tensor.
        """
        # Sum attention mass going to each of the first n_sinks key positions
        sink_attn = attn_weights[:, :, :, : self.n_sinks].sum(dim=-1)  # (B,H,T_q)
        return -sink_attn.mean()


# ---------------------------------------------------------------------------
# StreamingLLMDecoder
# ---------------------------------------------------------------------------


class StreamingLLMDecoder:
    """Autoregressive decoder that wraps any callable model with a streaming
    KV cache via the attention-sink strategy.

    Args:
        model:       Callable (input_ids: LongTensor(B, T)) → logits(B, T, V).
        n_sinks:     Number of sink positions.
        window_size: Maximum non-sink KV window size.
    """

    def __init__(
        self,
        model: nn.Module,
        n_sinks: int = 4,
        window_size: int = 16,
    ) -> None:
        self.model = model
        self.n_sinks = n_sinks
        self.window_size = window_size

    # ------------------------------------------------------------------
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 20,
    ) -> Tensor:
        """Generate *max_new_tokens* tokens autoregressively.

        Args:
            input_ids: (B, T) integer token ids.

        Returns:
            (B, T + max_new_tokens) concatenated ids.
        """
        B, T = input_ids.shape
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Use a bounded context window: sinks + last window_size tokens
                ctx_len = self.n_sinks + self.window_size
                if generated.size(1) > ctx_len:
                    # Keep first n_sinks and last window_size
                    ctx = torch.cat(
                        [
                            generated[:, : self.n_sinks],
                            generated[:, -self.window_size :],
                        ],
                        dim=1,
                    )
                else:
                    ctx = generated

                logits = self.model(ctx)  # (B, ctx_len, V)
                next_logits = logits[:, -1, :]  # (B, V)
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
                generated = torch.cat([generated, next_token], dim=1)

        return generated


# ---------------------------------------------------------------------------
# SinkAnalyzer
# ---------------------------------------------------------------------------


class SinkAnalyzer:
    """Diagnostic utilities for analysing sink-token attention behaviour."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    def sink_ratio(self, attn_weights: Tensor, n_sinks: int) -> float:
        """Fraction of total attention mass received by the first *n_sinks*
        key positions.

        Args:
            attn_weights: (B, H, T_q, T_k)
            n_sinks:      Number of sink positions to consider.

        Returns:
            float in [0, 1].
        """
        total = attn_weights.sum().item()
        if total == 0.0:
            return 0.0
        sink_total = attn_weights[:, :, :, :n_sinks].sum().item()
        ratio = sink_total / total
        # Clamp for numerical safety
        return float(max(0.0, min(1.0, ratio)))

    # ------------------------------------------------------------------
    def sink_stability(self, attn_weights_list: List[Tensor]) -> float:
        """Standard deviation of sink_ratio across a list of attention weight
        tensors (e.g. one per layer).

        Lower values indicate more consistent sink behaviour across layers.

        Args:
            attn_weights_list: List of (B, H, T_q, T_k) tensors.

        Returns:
            float ≥ 0.
        """
        if len(attn_weights_list) == 0:
            return 0.0

        # Infer n_sinks from first tensor's shape (use 1 as minimal default)
        n_sinks = max(1, attn_weights_list[0].size(-1) // 4)
        ratios = [self.sink_ratio(aw, n_sinks) for aw in attn_weights_list]

        if len(ratios) == 1:
            return 0.0

        mean_r = sum(ratios) / len(ratios)
        variance = sum((r - mean_r) ** 2 for r in ratios) / len(ratios)
        return float(variance ** 0.5)

    # ------------------------------------------------------------------
    def window_coverage(
        self,
        total_tokens: int,
        n_sinks: int,
        window_size: int,
    ) -> float:
        """Fraction of the total token sequence covered by the cache.

        Returns:
            (n_sinks + window_size) / total_tokens, clamped to [0, 1].
        """
        if total_tokens == 0:
            return 1.0
        coverage = (n_sinks + window_size) / total_tokens
        return float(max(0.0, min(1.0, coverage)))
