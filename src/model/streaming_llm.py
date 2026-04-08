"""StreamingLLM: Attention Sink KV Cache for Infinite-Length Generation.

Implements the StreamingLLM approach (Xiao et al. 2023): keep a small fixed
"sink" window (first K tokens) plus a sliding recency window. The key insight
is that LLMs naturally route attention to initial tokens (sinks), so preserving
those plus recent context enables streaming indefinitely with fixed memory.

Reference: https://arxiv.org/abs/2309.17453
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class StreamingConfig:
    """Configuration for StreamingLLM attention sink cache.

    Attributes:
        sink_size: Number of initial "sink" tokens to always keep.
        window_size: Number of recent tokens to keep in the sliding window.
        max_cache_size: Total cache capacity = sink_size + window_size.
    """

    sink_size: int = 4
    window_size: int = 512
    max_cache_size: int = 516  # sink_size + window_size

    def __post_init__(self) -> None:
        # Allow explicit override of max_cache_size; otherwise keep in sync.
        if self.max_cache_size == 516 and (self.sink_size + self.window_size) != 516:
            # User set sink/window differently — auto-sync max_cache_size.
            object.__setattr__(self, "max_cache_size", self.sink_size + self.window_size)


class KVCache:
    """Finite KV cache with attention sink eviction.

    Structure: [sink_tokens | recent_tokens]
    - sink_tokens: first `sink_size` tokens, never evicted.
    - recent_tokens: sliding window of last `window_size` tokens.

    When cache is full and a new token arrives:
    - Keep sink_tokens unchanged.
    - Evict oldest recent_token, add new token at the end.

    Args:
        config: StreamingConfig controlling cache sizes.
        n_layers: Number of transformer layers.
        n_heads: Number of KV heads.
        head_dim: Per-head dimension.
    """

    def __init__(
        self,
        config: StreamingConfig,
        n_layers: int,
        n_heads: int,
        head_dim: int,
    ) -> None:
        self.config = config
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim

        # Per-layer storage: (1, n_heads, T, head_dim) or None if empty.
        self._keys: list[torch.Tensor | None] = [None] * n_layers
        self._values: list[torch.Tensor | None] = [None] * n_layers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new_k / new_v to the cache for layer_idx and return the full cache.

        Args:
            layer_idx: Which layer to update.
            new_k: (1, n_heads, 1, head_dim) — single new token key.
            new_v: (1, n_heads, 1, head_dim) — single new token value.

        Returns:
            (cached_k, cached_v) with shape (1, n_heads, T', head_dim) where
            T' <= max_cache_size.
        """
        sink = self.config.sink_size
        max_size = self.config.max_cache_size

        existing_k = self._keys[layer_idx]
        existing_v = self._values[layer_idx]

        if existing_k is None:
            # First token for this layer — just store.
            self._keys[layer_idx] = new_k
            self._values[layer_idx] = new_v
        else:
            current_len = existing_k.shape[2]

            if current_len < max_size:
                # Still below capacity — simple append.
                self._keys[layer_idx] = torch.cat([existing_k, new_k], dim=2)
                self._values[layer_idx] = torch.cat([existing_v, new_v], dim=2)
            else:
                # At capacity: keep sink tokens, drop oldest recent token,
                # append new token.
                sink_k = existing_k[:, :, :sink, :]
                sink_v = existing_v[:, :, :sink, :]
                # Drop the token at index `sink` (oldest recent).
                recent_k = existing_k[:, :, sink + 1:, :]
                recent_v = existing_v[:, :, sink + 1:, :]
                self._keys[layer_idx] = torch.cat([sink_k, recent_k, new_k], dim=2)
                self._values[layer_idx] = torch.cat([sink_v, recent_v, new_v], dim=2)

        return self._keys[layer_idx], self._values[layer_idx]

    def get(
        self, layer_idx: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return (keys, values) for layer_idx, or (None, None) if empty."""
        return self._keys[layer_idx], self._values[layer_idx]

    def clear(self, layer_idx: int | None = None) -> None:
        """Clear cache for a specific layer, or all layers if layer_idx is None."""
        if layer_idx is None:
            self._keys = [None] * self.n_layers
            self._values = [None] * self.n_layers
        else:
            self._keys[layer_idx] = None
            self._values[layer_idx] = None

    @property
    def cache_size(self) -> int:
        """Current number of cached tokens (from layer 0), or 0 if empty."""
        k = self._keys[0]
        if k is None:
            return 0
        return k.shape[2]


# ---------------------------------------------------------------------------
# SinkAttentionWrapper
# ---------------------------------------------------------------------------


class SinkAttentionWrapper(nn.Module):
    """Wraps an attention module to support StreamingLLM-style KV cache.

    At decode time: uses KVCache to maintain sink+recent tokens.
    At prefill (cache=None): delegates directly to the wrapped attention module.

    Args:
        attention: The attention module to wrap (e.g. GroupedQueryAttention).
        config: StreamingConfig controlling cache behaviour.
    """

    def __init__(self, attention: nn.Module, config: StreamingConfig) -> None:
        super().__init__()
        self.attention = attention
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        layer_idx: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass, optionally using streaming KV cache.

        Args:
            x: (batch, seq_len, d_model).
            cache: If provided, use StreamingLLM decode mode.
                   If None, run normal prefill via self.attention.
            layer_idx: Which cache layer to read/write.
            **kwargs: Passed through to self.attention when cache is None.

        Returns:
            (batch, seq_len, d_model) output tensor.
        """
        if cache is None:
            # Prefill mode — just call the underlying attention normally.
            out = self.attention(x, **kwargs)
            # Some attention modules return (output, kv_cache); strip cache.
            if isinstance(out, tuple):
                return out[0]
            return out

        # Decode mode -------------------------------------------------------
        # x is a single new token: (1, 1, d_model)
        B, S, d = x.shape
        assert S == 1, "Streaming decode mode expects single-token input (S=1)"

        # We need to run attention over the full cached context.
        # Retrieve existing cache for this layer.
        cached_k, cached_v = cache.get(layer_idx)

        # Build full context by updating the cache with the current token.
        # The attention module handles its own Q/K/V projections, so we pass
        # the full key/value cache as past_kv if supported.
        #
        # Strategy: run attention on x with past_kv from cache, then store
        # the returned new kv back into our KVCache with sink eviction.
        #
        # GroupedQueryAttention stores KV as (B, T, n_kv_heads, head_dim),
        # so we convert our (1, n_heads, T, head_dim) format accordingly.

        # Determine freqs_cis position offset from cache.
        past_len = cached_k.shape[2] if cached_k is not None else 0

        # Build past_kv in the format GroupedQueryAttention expects:
        # (B, past_len, n_kv_heads, head_dim) or None.
        if cached_k is not None:
            # Our cache is (1, n_kv_heads, T, head_dim); transpose to (1, T, n_kv_heads, head_dim)
            past_k = cached_k.transpose(1, 2)  # (1, T, n_kv_heads, head_dim)
            past_v = cached_v.transpose(1, 2)
            past_kv_for_attn = (past_k, past_v)
        else:
            past_kv_for_attn = None

        # Get freqs_cis from kwargs or the attention module's parent model.
        # We rely on the caller passing freqs_cis if needed.
        freqs_cis = kwargs.pop("freqs_cis", None)
        mask = kwargs.pop("mask", None)

        if freqs_cis is not None:
            # Slice to the position of the current token.
            token_freqs = freqs_cis[past_len: past_len + S]
            out, new_kv = self.attention(x, token_freqs, mask, past_kv_for_attn)
        else:
            out, new_kv = self.attention(x, **kwargs)

        # new_kv is (batch, T_total, n_kv_heads, head_dim); extract just the
        # new single token's k/v and update our sink cache.
        new_k_full, new_v_full = new_kv
        # The full returned kv already includes past; we want only the last 1 token.
        new_k_single = new_k_full[:, -1:, :, :]  # (1, 1, n_kv_heads, head_dim)
        new_v_single = new_v_full[:, -1:, :, :]

        # Convert to (1, n_kv_heads, 1, head_dim) for our cache format.
        new_k_single = new_k_single.transpose(1, 2)
        new_v_single = new_v_single.transpose(1, 2)

        cache.update(layer_idx, new_k_single, new_v_single)

        return out


# ---------------------------------------------------------------------------
# wrap_model_with_sink_cache
# ---------------------------------------------------------------------------


def wrap_model_with_sink_cache(
    model: nn.Module,
    streaming_cfg: StreamingConfig | None = None,
) -> None:
    """Replace each attention layer in model with SinkAttentionWrapper (in-place).

    Wraps model.layers[i].attn for each i.

    Args:
        model: AureliusTransformer instance.
        streaming_cfg: StreamingConfig to use; defaults to StreamingConfig().
    """
    if streaming_cfg is None:
        streaming_cfg = StreamingConfig()

    for layer in model.layers:
        if hasattr(layer, "attn"):
            layer.attn = SinkAttentionWrapper(layer.attn, streaming_cfg)


# ---------------------------------------------------------------------------
# StreamingGenerator
# ---------------------------------------------------------------------------


class StreamingGenerator:
    """Token-by-token generator using StreamingLLM attention sink cache.

    Maintains a KVCache across steps, enabling theoretically infinite generation
    with fixed memory footprint.

    Args:
        model: AureliusTransformer.
        config: StreamingConfig controlling cache sizes.
        tokenizer_encode: Callable that encodes text to token ids.
        eos_token_id: Stop generation when this token is produced (default 2).
    """

    def __init__(
        self,
        model: nn.Module,
        config: StreamingConfig | None = None,
        tokenizer_encode=None,
        eos_token_id: int = 2,
    ) -> None:
        self.model = model
        self.config = config or StreamingConfig()
        self.tokenizer_encode = tokenizer_encode
        self.eos_token_id = eos_token_id

        # Infer model dimensions from the model config.
        model_cfg = getattr(model, "config", None)
        n_layers = model_cfg.n_layers if model_cfg else len(list(model.layers))
        n_kv_heads = model_cfg.n_kv_heads if model_cfg else 1
        head_dim = model_cfg.head_dim if model_cfg else 64

        self._cache = KVCache(
            config=self.config,
            n_layers=n_layers,
            n_heads=n_kv_heads,
            head_dim=head_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all KV caches."""
        self._cache.clear()

    @torch.no_grad()
    def generate_token(self, input_ids: torch.Tensor) -> int:
        """Run one decode step and return the next token id.

        If the internal cache is empty (first call), this runs as a prefill over
        the full input and populates the cache from the resulting KV pairs.
        If the cache is already populated, only the last token is processed.

        Args:
            input_ids: (1, T) token ids.

        Returns:
            Next token id as a Python int.
        """
        device = input_ids.device
        model_cfg = getattr(self.model, "config", None)

        if self._cache.cache_size == 0:
            # Prefill: run full forward pass to populate KV cache.
            _, logits, present_kvs = self.model(input_ids)

            # Populate our streaming cache with the prefill KV pairs.
            # present_kvs is list of (B, T, n_kv_heads, head_dim).
            for layer_idx, (pk, pv) in enumerate(present_kvs):
                T = pk.shape[1]
                sink = self.config.sink_size
                max_size = self.config.max_cache_size

                if T <= max_size:
                    # Fits: store as-is (transposed to our cache format).
                    self._cache._keys[layer_idx] = pk.transpose(1, 2)   # (1, n_kv_heads, T, head_dim)
                    self._cache._values[layer_idx] = pv.transpose(1, 2)
                else:
                    # Truncate: keep sink + last window_size tokens.
                    sink_k = pk[:, :sink, :, :]
                    sink_v = pv[:, :sink, :, :]
                    recent_k = pk[:, T - self.config.window_size:, :, :]
                    recent_v = pv[:, T - self.config.window_size:, :, :]
                    k_stored = torch.cat([sink_k, recent_k], dim=1)
                    v_stored = torch.cat([sink_v, recent_v], dim=1)
                    self._cache._keys[layer_idx] = k_stored.transpose(1, 2)
                    self._cache._values[layer_idx] = v_stored.transpose(1, 2)

            # Logits from prefill.
            next_token_id = int(logits[0, -1, :].argmax().item())
            return next_token_id
        else:
            # Decode: process only the last token with the streaming cache.
            last_token = input_ids[:, -1:]  # (1, 1)
            _, logits, present_kvs = self.model(last_token)
            # Update streaming cache for each layer using the single new token's kv.
            for layer_idx, (pk, pv) in enumerate(present_kvs):
                # pk: (1, 1, n_kv_heads, head_dim)
                new_k = pk.transpose(1, 2)  # (1, n_kv_heads, 1, head_dim)
                new_v = pv.transpose(1, 2)
                self._cache.update(layer_idx, new_k, new_v)

            next_token_id = int(logits[0, -1, :].argmax().item())
            return next_token_id

    @torch.no_grad()
    def stream(
        self, prompt_ids: list[int], max_tokens: int = 100
    ) -> list[int]:
        """Generate max_tokens tokens autoregressively.

        Args:
            prompt_ids: List of prompt token ids.
            max_tokens: Number of tokens to generate.

        Returns:
            List of generated token ids (length == max_tokens unless EOS).
        """
        self.reset()

        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        generated: list[int] = []
        for _ in range(max_tokens):
            next_id = self.generate_token(input_ids)
            generated.append(next_id)
            if next_id == self.eos_token_id:
                break
            # Append new token for next step.
            new_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, new_token], dim=1)

        return generated
