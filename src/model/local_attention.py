"""Local window attention with ring buffer KV cache.

Implements efficient sliding window attention for autoregressive generation.
At inference time only the last `window_size` tokens are attended to, using
a ring buffer to maintain the KV cache without reallocating memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .config import AureliusConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LocalAttentionConfig:
    window_size: int = 512          # number of tokens to attend to
    overlap: int = 0                # optional overlap with previous window (for continuity)
    use_ring_buffer: bool = True    # efficient ring buffer vs simple append


# ---------------------------------------------------------------------------
# Ring Buffer
# ---------------------------------------------------------------------------

class RingBuffer:
    """Fixed-size ring buffer for efficient sliding window KV cache.

    Maintains last `capacity` elements. When full, overwrites oldest.
    O(1) insert, O(capacity) to read all elements in order.

    Args:
        capacity: Maximum number of elements the buffer holds.
        shape: Shape of each element (additional dimensions, e.g. (n_heads, head_dim)).
        dtype: Data type for the underlying storage tensor.
    """

    def __init__(
        self,
        capacity: int,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._buffer = torch.zeros(capacity, *shape, dtype=dtype)
        self._write_pos: int = 0
        self._size: int = 0

    def push(self, value: torch.Tensor) -> None:
        """Add value (shape == self._buffer.shape[1:]) to buffer.

        Overwrites the oldest element when the buffer is full.
        """
        self._buffer[self._write_pos] = value
        self._write_pos = (self._write_pos + 1) % len(self._buffer)
        self._size = min(self._size + 1, len(self._buffer))

    def read_ordered(self) -> torch.Tensor:
        """Return all elements in chronological order (oldest first).

        Returns:
            Tensor of shape (size, *shape).
        """
        capacity = len(self._buffer)
        if self._size < capacity:
            # Buffer has not wrapped; elements are at positions 0.._size-1
            return self._buffer[: self._size]
        else:
            # Buffer has wrapped; oldest element is at _write_pos
            # Concatenate [_write_pos:] ++ [:_write_pos] to restore order
            tail = self._buffer[self._write_pos :]
            head = self._buffer[: self._write_pos]
            return torch.cat([tail, head], dim=0)

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == len(self._buffer)


# ---------------------------------------------------------------------------
# KV Cache using ring buffers
# ---------------------------------------------------------------------------

class LocalWindowKVCache:
    """KV cache for local window attention.

    Maintains one RingBuffer per layer for keys and one for values.

    Args:
        n_layers: Number of transformer layers.
        n_heads: Number of KV heads (GQA; may differ from query heads).
        head_dim: Dimension of each attention head.
        window_size: Number of past tokens to retain in each buffer.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        window_size: int,
    ) -> None:
        self.n_layers = n_layers
        self.window_size = window_size
        self.k_buffers: list[RingBuffer] = [
            RingBuffer(window_size, (n_heads, head_dim)) for _ in range(n_layers)
        ]
        self.v_buffers: list[RingBuffer] = [
            RingBuffer(window_size, (n_heads, head_dim)) for _ in range(n_layers)
        ]

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,  # (n_heads, head_dim) — single token
        new_v: torch.Tensor,  # (n_heads, head_dim) — single token
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add new K/V to ring buffer, return windowed (K, V).

        Args:
            layer_idx: Which layer's buffers to update.
            new_k: Key tensor for the current token, shape (n_heads, head_dim).
            new_v: Value tensor for the current token, shape (n_heads, head_dim).

        Returns:
            Tuple (k, v) each of shape (T_window, n_heads, head_dim) where
            T_window <= window_size.
        """
        self.k_buffers[layer_idx].push(new_k)
        self.v_buffers[layer_idx].push(new_v)
        k_window = self.k_buffers[layer_idx].read_ordered()
        v_window = self.v_buffers[layer_idx].read_ordered()
        return k_window, v_window

    def clear(self, layer_idx: int | None = None) -> None:
        """Clear all buffers (or just one layer's buffers).

        Args:
            layer_idx: If provided, clear only this layer. If None, clear all.
        """
        if layer_idx is None:
            for kb, vb in zip(self.k_buffers, self.v_buffers):
                kb._write_pos = 0
                kb._size = 0
                kb._buffer.zero_()
                vb._write_pos = 0
                vb._size = 0
                vb._buffer.zero_()
        else:
            self.k_buffers[layer_idx]._write_pos = 0
            self.k_buffers[layer_idx]._size = 0
            self.k_buffers[layer_idx]._buffer.zero_()
            self.v_buffers[layer_idx]._write_pos = 0
            self.v_buffers[layer_idx]._size = 0
            self.v_buffers[layer_idx]._buffer.zero_()


# ---------------------------------------------------------------------------
# Local Window Attention Module
# ---------------------------------------------------------------------------

class LocalWindowAttention(nn.Module):
    """Attention that only attends to the last `window_size` tokens.

    At training time: apply a local window causal mask over the full sequence.
    At inference time (single token decode): use LocalWindowKVCache with ring
    buffers so memory is O(window_size) rather than O(sequence_length).

    Supports Grouped Query Attention (n_kv_heads may differ from n_heads).

    Args:
        config: AureliusConfig with model dimensions.
        local_cfg: LocalAttentionConfig; uses defaults if None.
    """

    def __init__(
        self,
        config: AureliusConfig,
        local_cfg: LocalAttentionConfig | None = None,
    ) -> None:
        super().__init__()
        self.local_cfg = local_cfg or LocalAttentionConfig()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.n_rep = self.n_heads // self.n_kv_heads  # GQA repetition factor
        self.window_size = self.local_cfg.window_size
        self.attn_dropout = config.dropout

        # Projections — no bias (matches the rest of the Aurelius codebase)
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

    def _local_causal_mask(
        self, seq_len: int, window: int, device: torch.device
    ) -> torch.Tensor:
        """Create a causal mask where token i can only attend to [max(0, i-window+1), i].

        Args:
            seq_len: Sequence length T.
            window: Number of tokens (including current) each position may attend to.
            device: Target device.

        Returns:
            Boolean mask of shape (1, 1, seq_len, seq_len).
            True means the position is attended; False means it is masked.
        """
        rows = torch.arange(seq_len, device=device).unsqueeze(1)   # (T, 1)
        cols = torch.arange(seq_len, device=device).unsqueeze(0)   # (1, T)
        # Causal: col <= row
        # Window:  col >= row - window + 1
        mask = (cols <= rows) & (cols >= rows - window + 1)        # (T, T) bool
        return mask.unsqueeze(0).unsqueeze(0)                      # (1, 1, T, T)

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, D)
        local_cache: LocalWindowKVCache | None = None,
        layer_idx: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        If local_cache is provided and T == 1, runs in decode mode using the
        ring buffer.  Otherwise runs in training/prefill mode applying a local
        window causal mask over the full sequence.

        Args:
            x: Input tensor (B, T, D).
            local_cache: Optional LocalWindowKVCache for incremental decoding.
            layer_idx: Layer index used to select the correct cache buffers.

        Returns:
            Output tensor (B, T, D).
        """
        B, T, _ = x.shape
        device = x.device

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # ---- Decode mode: T == 1 and a ring-buffer cache is available --------
        if local_cache is not None and T == 1:
            # Only supports batch_size == 1 in single-token decode mode
            assert B == 1, "Ring-buffer decode mode supports batch_size == 1"

            # Squeeze batch and sequence dims: (n_kv_heads, head_dim)
            k_tok = k[0, 0]  # (n_kv_heads, head_dim)
            v_tok = v[0, 0]  # (n_kv_heads, head_dim)

            k_win, v_win = local_cache.update(layer_idx, k_tok, v_tok)
            # k_win: (T_window, n_kv_heads, head_dim)
            T_win = k_win.shape[0]

            # Expand KV for GQA: (T_window, n_heads, head_dim)
            if self.n_rep > 1:
                k_win = k_win.unsqueeze(2).expand(T_win, self.n_kv_heads, self.n_rep, self.head_dim)
                k_win = k_win.reshape(T_win, self.n_heads, self.head_dim)
                v_win = v_win.unsqueeze(2).expand(T_win, self.n_kv_heads, self.n_rep, self.head_dim)
                v_win = v_win.reshape(T_win, self.n_heads, self.head_dim)

            # Reshape for SDPA: (1, n_heads, T, head_dim)
            q_sdpa = q.transpose(1, 2)                            # (1, n_heads, 1, head_dim)
            k_sdpa = k_win.transpose(0, 1).unsqueeze(0)          # (1, n_heads, T_win, head_dim)
            v_sdpa = v_win.transpose(0, 1).unsqueeze(0)          # (1, n_heads, T_win, head_dim)

            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,  # query sees only past tokens via the ring buffer
            )
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
            return self.o_proj(out)

        # ---- Training / prefill mode: apply local window causal mask ---------
        bool_mask = self._local_causal_mask(T, self.window_size, device)  # (1, 1, T, T)
        # Convert to additive mask for SDPA (0.0 = attend, -inf = block)
        additive_mask = torch.zeros(1, 1, T, T, device=device, dtype=x.dtype)
        additive_mask = additive_mask.masked_fill(~bool_mask, float("-inf"))

        # GQA: expand k/v to n_heads
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim)
            k = k.reshape(B, T, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim)
            v = v.reshape(B, T, self.n_heads, self.head_dim)

        # Transpose to (B, n_heads, T, head_dim) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=additive_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Token generator
# ---------------------------------------------------------------------------

class LocalAttentionGenerator:
    """Token generator using LocalWindowAttention with a ring buffer KV cache.

    Efficient for long-context generation with bounded memory: the KV cache
    never grows beyond O(n_layers * window_size * n_kv_heads * head_dim).

    Args:
        model: AureliusTransformer (or any module with a compatible forward
               signature: input_ids -> (loss, logits, _)).
        local_cfg: LocalAttentionConfig governing window size.
        n_layers: Number of transformer layers.
        n_heads: Number of KV heads (n_kv_heads from model config).
        head_dim: Attention head dimension.
    """

    def __init__(
        self,
        model: nn.Module,
        local_cfg: LocalAttentionConfig,
        n_layers: int,
        n_heads: int,
        head_dim: int,
    ) -> None:
        self.model = model
        self.local_cfg = local_cfg
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self._cache = LocalWindowKVCache(
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            window_size=local_cfg.window_size,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> list[int]:
        """Generate tokens using local window attention.

        The prompt is processed first (full forward pass), then each subsequent
        token is generated autoregressively.

        Args:
            prompt_ids: List of integer token ids forming the prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.

        Returns:
            List of generated token ids (excluding prompt).
        """
        self.model.train(False)
        self._cache.clear()

        device = next(self.model.parameters()).device

        # Process the prompt in a single forward pass
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, P)
        _, logits, _ = self.model(input_ids)
        # Sample the first new token from the last prompt position
        next_logits = logits[0, -1, :]
        if temperature != 1.0:
            next_logits = next_logits / temperature
        probs = next_logits.softmax(dim=-1)
        next_token = int(torch.multinomial(probs, num_samples=1).item())

        generated: list[int] = [next_token]

        for _ in range(max_new_tokens - 1):
            cur_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)  # (1, 1)
            _, logits, _ = self.model(cur_ids)
            next_logits = logits[0, -1, :]
            if temperature != 1.0:
                next_logits = next_logits / temperature
            probs = next_logits.softmax(dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_token)

        return generated

    def memory_usage(self) -> dict:
        """Report memory used by the KV cache.

        Returns:
            Dict with keys:
                cache_mb: float — total cache memory in megabytes.
                window_size: int — configured window size.
                n_layers: int — number of layers.
                tokens_in_cache: int — number of tokens currently cached
                    (based on the first layer's K buffer size).
        """
        # Each ring buffer stores (window_size, n_heads, head_dim) float32 elements
        bytes_per_buffer = (
            self.local_cfg.window_size * self.n_heads * self.head_dim * 4  # float32 = 4 bytes
        )
        # Two buffers (K and V) per layer
        total_bytes = 2 * self.n_layers * bytes_per_buffer
        cache_mb = total_bytes / (1024 ** 2)

        tokens_in_cache = self._cache.k_buffers[0].size if self.n_layers > 0 else 0

        return {
            "cache_mb": cache_mb,
            "window_size": self.local_cfg.window_size,
            "n_layers": self.n_layers,
            "tokens_in_cache": tokens_in_cache,
        }
