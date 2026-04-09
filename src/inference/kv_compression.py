"""KV cache compression via token importance scoring and dropping.

Reduces KV cache size while preserving generation quality using multiple
strategies: heavy-hitter oracle, streaming (sink + window), ScissorHands EMA,
and random eviction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class KVCompressionConfig:
    strategy: str = "heavy_hitter"  # "heavy_hitter" | "streaming" | "scissorhands" | "random"
    keep_ratio: float = 0.5         # fraction of tokens to keep
    local_window: int = 8           # always keep last local_window tokens
    n_init_tokens: int = 4          # always keep first n_init_tokens (attention sink)
    update_interval: int = 16       # how often to re-evaluate importance


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_token_importance_from_attn(
    attention_weights: Tensor,  # (B, H, T, T) or (H, T, T)
    method: str = "sum",        # "sum" | "max"
) -> Tensor:
    """Compute per-token importance from attention weights.

    "sum": sum over all heads and query positions -> importance per key position.
    "max": max over all heads and query positions.
    Returns (T,) importance scores (or (B, T) if batched).
    """
    batched = attention_weights.dim() == 4

    if batched:
        # (B, H, T_q, T_k) -> importance over T_k
        if method == "sum":
            # sum over query dim first, then heads
            scores = attention_weights.sum(dim=2)  # (B, H, T_k)
            scores = scores.sum(dim=1)             # (B, T_k)
        elif method == "max":
            scores = attention_weights.amax(dim=2)  # (B, H, T_k)
            scores = scores.amax(dim=1)             # (B, T_k)
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'sum' or 'max'.")
    else:
        # (H, T_q, T_k)
        if method == "sum":
            scores = attention_weights.sum(dim=1)  # (H, T_k)
            scores = scores.sum(dim=0)             # (T_k,)
        elif method == "max":
            scores = attention_weights.amax(dim=1)  # (H, T_k)
            scores = scores.amax(dim=0)             # (T_k,)
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'sum' or 'max'.")

    return scores


def select_tokens_to_keep(
    importance: Tensor,   # (T,) or (B, T)
    keep_ratio: float,
    local_window: int,
    n_init_tokens: int,
) -> Tensor:
    """Select token indices to keep.

    Always keep: [0:n_init_tokens] + [-local_window:] + top-k by importance from middle.
    where k = int(T * keep_ratio) - n_init_tokens - local_window
    Returns sorted indices tensor.
    """
    batched = importance.dim() == 2
    if batched:
        # Process first batch element (common pattern for single-batch inference)
        importance = importance[0]

    T = importance.shape[0]

    # Mandatory sets
    init_indices = set(range(min(n_init_tokens, T)))
    local_start = max(0, T - local_window)
    local_indices = set(range(local_start, T))
    mandatory = init_indices | local_indices

    # Middle tokens available for scoring
    middle_indices = [i for i in range(T) if i not in mandatory]

    # Number of additional tokens to keep from the middle
    k = int(T * keep_ratio) - len(mandatory)

    if k <= 0 or len(middle_indices) == 0:
        # Keep everything mandatory (or all if T is small)
        if len(mandatory) >= T:
            kept = sorted(range(T))
        else:
            kept = sorted(mandatory)
        return torch.tensor(kept, dtype=torch.long)

    # Safety: if not enough middle tokens, just keep them all
    k = min(k, len(middle_indices))

    middle_tensor = torch.tensor(middle_indices, dtype=torch.long)
    middle_importance = importance[middle_tensor]
    top_k_local = torch.topk(middle_importance, k).indices
    top_k_global = middle_tensor[top_k_local]

    kept = sorted(mandatory | set(top_k_global.tolist()))
    return torch.tensor(kept, dtype=torch.long)


def compress_kv_cache(
    kv_cache: list[tuple[Tensor, Tensor]],  # list of (K, V) pairs, each (B, H, T, D)
    keep_indices: Tensor,                    # (T_keep,) or (B, T_keep)
) -> list[tuple[Tensor, Tensor]]:
    """Select kept positions from KV cache. Returns compressed KV cache."""
    compressed = []
    for k, v in kv_cache:
        if keep_indices.dim() == 1:
            # Broadcast across batch
            k_new = k[:, :, keep_indices, :]
            v_new = v[:, :, keep_indices, :]
        else:
            # Per-batch indices: (B, T_keep) — index each batch element
            B = k.shape[0]
            k_new = torch.stack(
                [k[b, :, keep_indices[b], :] for b in range(B)], dim=0
            )
            v_new = torch.stack(
                [v[b, :, keep_indices[b], :] for b in range(B)], dim=0
            )
        compressed.append((k_new, v_new))
    return compressed


def streaming_kv_drop(
    kv_cache: list[tuple[Tensor, Tensor]],  # (B, H, T, D)
    n_sink: int,
    window_size: int,
) -> list[tuple[Tensor, Tensor]]:
    """StreamingLLM-style: keep first n_sink tokens (attention sinks) + last window_size tokens."""
    result = []
    for k, v in kv_cache:
        T = k.shape[2]
        sink_end = min(n_sink, T)
        window_start = max(sink_end, T - window_size)

        k_sink = k[:, :, :sink_end, :]
        v_sink = v[:, :, :sink_end, :]
        k_window = k[:, :, window_start:, :]
        v_window = v[:, :, window_start:, :]

        k_new = torch.cat([k_sink, k_window], dim=2)
        v_new = torch.cat([v_sink, v_window], dim=2)
        result.append((k_new, v_new))
    return result


def scissorhands_score(
    attention_weights: Tensor,   # (H, T, T) — attention at current step
    history_weights: Tensor,     # (H, T) — accumulated importance
    decay: float = 0.9,
) -> Tensor:
    """ScissorHands: EMA update of token importance.

    new_history = decay * history + (1-decay) * current_attn.sum(dim=-2)
    Returns updated (H, T) importance.
    """
    # Sum over query dimension to get per-key importance
    current_importance = attention_weights.sum(dim=-2)  # (H, T)
    return decay * history_weights + (1.0 - decay) * current_importance


def estimate_compression_quality(
    original_logits: Tensor,    # (B, T, V)
    compressed_logits: Tensor,  # (B, T, V)
) -> dict[str, float]:
    """Estimate quality of compression by comparing logit distributions.

    Returns {"kl_divergence", "top1_agreement", "perplexity_ratio"}
    - kl_divergence: mean KL(original || compressed)
    - top1_agreement: fraction where argmax matches
    - perplexity_ratio: exp(mean_ce_compressed) / exp(mean_ce_original)
    """
    # Compute softmax probabilities
    orig_probs = F.softmax(original_logits, dim=-1)      # (B, T, V)
    comp_probs = F.softmax(compressed_logits, dim=-1)    # (B, T, V)

    # KL divergence: KL(orig || comp) = sum(orig * log(orig / comp))
    # Use log_softmax for numerical stability
    orig_log = F.log_softmax(original_logits, dim=-1)
    comp_log = F.log_softmax(compressed_logits, dim=-1)

    # kl_div expects input=log_probs, target=probs
    kl = F.kl_div(comp_log, orig_probs, reduction="batchmean").item()

    # Top-1 agreement
    orig_top1 = original_logits.argmax(dim=-1)      # (B, T)
    comp_top1 = compressed_logits.argmax(dim=-1)    # (B, T)
    top1_agreement = (orig_top1 == comp_top1).float().mean().item()

    # Perplexity ratio: use cross-entropy against greedy labels
    B, T, V = original_logits.shape
    orig_flat = original_logits.view(B * T, V)
    comp_flat = compressed_logits.view(B * T, V)
    labels = orig_top1.view(B * T)

    ce_orig = F.cross_entropy(orig_flat, labels).item()
    ce_comp = F.cross_entropy(comp_flat, labels).item()
    perplexity_ratio = (ce_comp - ce_orig)  # log-ratio, exp gives ratio
    import math
    perplexity_ratio = math.exp(perplexity_ratio)

    return {
        "kl_divergence": kl,
        "top1_agreement": top1_agreement,
        "perplexity_ratio": perplexity_ratio,
    }


# ---------------------------------------------------------------------------
# KVCacheCompressor class
# ---------------------------------------------------------------------------


class KVCacheCompressor:
    """Manage KV cache compression across generation steps."""

    def __init__(self, cfg: KVCompressionConfig) -> None:
        self.cfg = cfg
        self._history_weights: Tensor | None = None

    def compress(
        self,
        kv_cache: list[tuple[Tensor, Tensor]],
        attention_weights: Tensor | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Apply configured compression strategy."""
        strategy = self.cfg.strategy

        if strategy == "streaming":
            return streaming_kv_drop(
                kv_cache,
                n_sink=self.cfg.n_init_tokens,
                window_size=self.cfg.local_window,
            )

        if strategy == "random":
            # Random eviction respecting mandatory tokens
            if not kv_cache:
                return kv_cache
            T = kv_cache[0][0].shape[2]
            importance = torch.rand(T)
            keep_indices = select_tokens_to_keep(
                importance,
                keep_ratio=self.cfg.keep_ratio,
                local_window=self.cfg.local_window,
                n_init_tokens=self.cfg.n_init_tokens,
            )
            return compress_kv_cache(kv_cache, keep_indices)

        if strategy == "scissorhands":
            if attention_weights is None or self._history_weights is None:
                # Fall back to streaming if no history yet
                return streaming_kv_drop(
                    kv_cache,
                    n_sink=self.cfg.n_init_tokens,
                    window_size=self.cfg.local_window,
                )
            # Use EMA importance
            attn = attention_weights
            if attn.dim() == 4:
                attn = attn[0]  # use first batch element
            H, T_q, T_k = attn.shape
            if self._history_weights.shape[-1] != T_k:
                # Re-initialize if sizes changed
                self._history_weights = torch.zeros(H, T_k)
            importance = self._history_weights.sum(dim=0)  # (T_k,)
            keep_indices = select_tokens_to_keep(
                importance,
                keep_ratio=self.cfg.keep_ratio,
                local_window=self.cfg.local_window,
                n_init_tokens=self.cfg.n_init_tokens,
            )
            return compress_kv_cache(kv_cache, keep_indices)

        # Default: heavy_hitter
        if attention_weights is None:
            # No attention weights available; fall back to streaming
            return streaming_kv_drop(
                kv_cache,
                n_sink=self.cfg.n_init_tokens,
                window_size=self.cfg.local_window,
            )
        importance = compute_token_importance_from_attn(attention_weights, method="sum")
        if importance.dim() == 2:
            importance = importance[0]
        keep_indices = select_tokens_to_keep(
            importance,
            keep_ratio=self.cfg.keep_ratio,
            local_window=self.cfg.local_window,
            n_init_tokens=self.cfg.n_init_tokens,
        )
        return compress_kv_cache(kv_cache, keep_indices)

    def get_compression_stats(
        self,
        original_len: int,
        compressed_len: int,
    ) -> dict[str, float]:
        """Return {"compression_ratio", "kept_tokens", "dropped_tokens"}"""
        return {
            "compression_ratio": compressed_len / max(original_len, 1),
            "kept_tokens": float(compressed_len),
            "dropped_tokens": float(original_len - compressed_len),
        }

    def update_importance(
        self,
        attention_weights: Tensor,
        step: int,
    ) -> None:
        """Update accumulated importance scores (for scissorhands strategy)."""
        attn = attention_weights
        if attn.dim() == 4:
            attn = attn[0]  # (H, T, T)
        H, T_q, T_k = attn.shape

        if self._history_weights is None or self._history_weights.shape[-1] != T_k:
            self._history_weights = torch.zeros(H, T_k)

        self._history_weights = scissorhands_score(
            attn, self._history_weights, decay=0.9
        )
