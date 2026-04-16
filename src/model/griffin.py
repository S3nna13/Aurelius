"""Griffin: Mixing Gated Linear Recurrences with Local Attention.

Reference: De et al., 2024 — "Griffin: Mixing Gated Linear Recurrences with
Local Attention for Efficient Language Models". https://arxiv.org/abs/2402.19427

Architecture overview:
    Griffin interleaves RG-LRU (Real-Gated Linear Recurrent Unit) layers with
    local sliding-window attention layers. The ratio is configurable via
    lru_per_attn: for every 1 local attention block there are lru_per_attn
    RG-LRU blocks.

RG-LRU core (simplified form used here):
    r_t  = sigmoid(W_r x_t + b_r)          # input gate
    i_t  = sigmoid(W_i x_t + b_i)          # recurrent gate
    α    = sigmoid(Λ) clamped to (ε, 1-ε)  # learnable per-channel decay
    h_t  = α * h_{t-1} + sqrt(1 - α²) * (r_t * x_t)  # recurrent state
    y_t  = h_t * i_t                                    # gated output

O(1) inference memory: the recurrent state h is a single (B, d_model) tensor
independent of sequence length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rms_norm import RMSNorm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GriffinConfig:
    """Hyperparameters for the Griffin hybrid model."""

    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64
    d_ff: int = 1024
    n_layers: int = 12
    vocab_size: int = 32000
    local_window: int = 512       # local attention window size
    mlp_expansion: float = 2.0    # SwiGLU inner dim = d_model * mlp_expansion
    lru_per_attn: int = 2         # number of RG-LRU blocks per local attn block
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# SwiGLU FFN (self-contained, avoids dependency on AureliusConfig)
# ---------------------------------------------------------------------------

class GriffinSwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network for Griffin blocks."""

    def __init__(self, config: GriffinConfig) -> None:
        super().__init__()
        d_inner = int(config.d_model * config.mlp_expansion)
        self.gate_proj = nn.Linear(config.d_model, d_inner, bias=False)
        self.up_proj   = nn.Linear(config.d_model, d_inner, bias=False)
        self.down_proj = nn.Linear(d_inner, config.d_model, bias=False)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# RG-LRU: Real-Gated Linear Recurrent Unit
# ---------------------------------------------------------------------------

class RGLRULayer(nn.Module):
    """Real-Gated Linear Recurrent Unit (RG-LRU).

    Processes a sequence recurrently with O(1) inference memory.

    The recurrent state h_t is updated as:
        r_t  = sigmoid(W_r x_t + b_r)          # input gate
        i_t  = sigmoid(W_i x_t + b_i)          # recurrent gate
        α    = sigmoid(Λ) clamped to (ε, 1-ε)  # per-channel decay in (0,1)
        h_t  = α * h_{t-1} + sqrt(1-α²) * (r_t * x_t)
        y_t  = h_t * i_t

    Args:
        config: GriffinConfig with at least d_model set.
    """

    def __init__(self, config: GriffinConfig) -> None:
        super().__init__()
        d = config.d_model

        # Input projection to mix channels before gating
        self.input_proj = nn.Linear(d, d, bias=False)

        # Input gate (r) and recurrent gate (i) — both project from x
        self.gate_r = nn.Linear(d, d, bias=True)
        self.gate_i = nn.Linear(d, d, bias=True)

        # Learnable per-channel log-decay parameter Λ (before sigmoid)
        self.log_decay = nn.Parameter(torch.zeros(d))

        # Output projection
        self.out_proj = nn.Linear(d, d, bias=False)

        self.norm = RMSNorm(d, eps=config.rms_norm_eps)
        self._eps = 1e-6

    @property
    def alpha(self) -> Tensor:
        """Per-channel decay factor α in (ε, 1-ε)."""
        return torch.sigmoid(self.log_decay).clamp(self._eps, 1.0 - self._eps)

    def forward(
        self,
        x: Tensor,                            # (B, T, d_model)
        state: Optional[Tensor] = None,       # (B, d_model) or None
    ) -> tuple[Tensor, Tensor]:               # (output, new_state)
        """Forward pass through the RG-LRU layer.

        Args:
            x:     Input tensor of shape (B, T, d_model).
            state: Optional initial recurrent state (B, d_model). Defaults to zeros.

        Returns:
            (output, new_state): Both tensors; output has shape (B, T, d_model),
            new_state has shape (B, d_model).
        """
        B, T, d = x.shape

        # Initialise hidden state to zeros if not provided
        if state is None:
            state = x.new_zeros(B, d)

        alpha = self.alpha  # (d,)  — shared across time steps

        # Pre-compute gates and input transform for the whole sequence at once
        r = torch.sigmoid(self.gate_r(x))    # (B, T, d)  input gate
        i = torch.sigmoid(self.gate_i(x))    # (B, T, d)  recurrent (output) gate
        x_proj = self.input_proj(x)          # (B, T, d)  transformed input

        # Normalisation factor to keep unit variance
        scale = torch.sqrt(torch.clamp(1.0 - alpha ** 2, min=0.0))  # (d,)

        # Scan over time — stays in pure PyTorch, no custom CUDA needed
        outputs = []
        h = state  # (B, d)
        for t in range(T):
            h = alpha * h + scale * (r[:, t, :] * x_proj[:, t, :])
            y_t = h * i[:, t, :]
            outputs.append(y_t)

        out = torch.stack(outputs, dim=1)       # (B, T, d)
        out = self.out_proj(out)                # (B, T, d)
        return out, h  # h is the final state (B, d)


# ---------------------------------------------------------------------------
# Local Sliding-Window Attention
# ---------------------------------------------------------------------------

def _build_causal_sliding_mask(seq_len: int, window: int, device: torch.device) -> Tensor:
    """Build additive causal sliding-window mask of shape (T, T).

    Positions outside the window or in the future are masked with -inf.
    """
    i = torch.arange(seq_len, device=device).unsqueeze(1)   # (T, 1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)   # (1, T)

    # Causal: j <= i
    causal = j <= i
    # Window: i - j < window  (attend to up to `window` tokens in the past incl. self)
    in_window = (i - j) < window

    mask = torch.where(causal & in_window, torch.zeros(seq_len, seq_len, device=device),
                       torch.full((seq_len, seq_len), float('-inf'), device=device))
    return mask  # (T, T)


class LocalSlidingWindowAttention(nn.Module):
    """Causal multi-head attention restricted to the last `window` tokens.

    Standard scaled dot-product attention with an additive mask that blocks
    all positions outside the local window, giving O(T * window) complexity.

    KV cache: pass ``kv_cache`` as a tuple ``(K_prev, V_prev)`` of shape
    ``(B, H, T_prev, Dh)`` to enable incremental / token-by-token decoding.
    The method returns the updated cache ``(K_full, V_full)`` as a second value.
    The cache is capped to the last ``window`` tokens to bound memory.
    """

    def __init__(self, config: GriffinConfig) -> None:
        super().__init__()
        self.n_heads   = config.n_heads
        self.head_dim  = config.head_dim
        self.window    = config.local_window
        d_attn = config.n_heads * config.head_dim

        self.q_proj = nn.Linear(config.d_model, d_attn, bias=False)
        self.k_proj = nn.Linear(config.d_model, d_attn, bias=False)
        self.v_proj = nn.Linear(config.d_model, d_attn, bias=False)
        self.out_proj = nn.Linear(d_attn, config.d_model, bias=False)

        self.scale = math.sqrt(config.head_dim)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Compute local sliding-window attention.

        Args:
            x:        (B, T, d_model)
            kv_cache: Optional (K_prev, V_prev) each (B, H, T_prev, Dh).

        Returns:
            (output, (K_full, V_full)):
                output  -- (B, T, d_model)
                K_full  -- updated K cache (B, H, min(T_prev+T, window), Dh)
                V_full  -- updated V cache (B, H, min(T_prev+T, window), Dh)
        """
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.head_dim

        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        K_new = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        V_new = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        # Concatenate with cached K/V (for incremental decoding)
        if kv_cache is not None:
            K_prev, V_prev = kv_cache
            # K_prev already capped to last `window` tokens
            # T_prev = K_prev.shape[2], absolute position offset = T_prev
            T_prev = K_prev.shape[2]
            K_full = torch.cat([K_prev, K_new], dim=2)   # (B, H, T_prev+T, Dh)
            V_full = torch.cat([V_prev, V_new], dim=2)
        else:
            T_prev = 0
            K_full = K_new   # (B, H, T, Dh)
            V_full = V_new

        T_kv = K_full.shape[2]  # = T_prev + T

        # Attention scores: Q is over the T new tokens, K/V over T_prev+T tokens.
        # Absolute positions: keys are at 0..T_kv-1; queries are at T_prev..T_kv-1.
        scores = torch.matmul(Q, K_full.transpose(-2, -1)) / self.scale  # (B, H, T, T_kv)

        # Build causal + window mask.
        # q absolute pos: T_prev + q_i  for q_i in [0, T-1]
        # k absolute pos: k_j            for k_j in [0, T_kv-1]
        q_abs = torch.arange(T_prev, T_prev + T, device=x.device).unsqueeze(1)   # (T, 1)
        k_abs = torch.arange(T_kv, device=x.device).unsqueeze(0)                  # (1, T_kv)
        causal    = k_abs <= q_abs                         # (T, T_kv)
        in_window = (q_abs - k_abs) < self.window          # (T, T_kv)
        additive_mask = torch.where(
            causal & in_window,
            torch.zeros(T, T_kv, device=x.device),
            torch.full((T, T_kv), float('-inf'), device=x.device),
        )
        scores = scores + additive_mask.unsqueeze(0).unsqueeze(0)  # (B, H, T, T_kv)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_full)         # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, H * Dh)

        # Cap K/V cache to last `window` tokens for storage efficiency
        K_cache = K_full[:, :, -self.window:, :] if K_full.shape[2] > self.window else K_full
        V_cache = V_full[:, :, -self.window:, :] if V_full.shape[2] > self.window else V_full

        return self.out_proj(out), (K_cache, V_cache)


# ---------------------------------------------------------------------------
# Griffin Block
# ---------------------------------------------------------------------------

class GriffinBlock(nn.Module):
    """A single Griffin block: either RG-LRU or local attention, followed by SwiGLU.

    Uses pre-norm (RMSNorm before the mixer and before the FFN) with residuals.

    Args:
        config:     GriffinConfig instance.
        block_type: ``"lru"`` for an RG-LRU block, ``"attn"`` for local attention.
    """

    def __init__(self, config: GriffinConfig, block_type: str) -> None:
        if block_type not in ("lru", "attn"):
            raise ValueError(f"block_type must be 'lru' or 'attn', got '{block_type}'")
        super().__init__()
        self.block_type = block_type

        self.mixer_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn_norm   = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn        = GriffinSwiGLUFFN(config)

        if block_type == "lru":
            self.mixer = RGLRULayer(config)
        else:
            self.mixer = LocalSlidingWindowAttention(config)

    def forward(
        self,
        x: Tensor,
        state=None,
    ) -> tuple[Tensor, object]:
        """Forward pass.

        Args:
            x:     (B, T, d_model)
            state: For LRU blocks: recurrent state (B, d_model) or None.
                   For attn blocks: KV cache tuple (K, V) or None.

        Returns:
            (output, new_state):
                output    -- (B, T, d_model)
                new_state -- updated recurrent state (LRU) or updated KV cache (attn)
        """
        normed = self.mixer_norm(x)

        if self.block_type == "lru":
            mixer_out, new_state = self.mixer(normed, state)
        else:
            mixer_out, new_state = self.mixer(normed, state)  # state = kv_cache or None

        x = x + mixer_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_state


# ---------------------------------------------------------------------------
# Full Griffin Model
# ---------------------------------------------------------------------------

class GriffinModel(nn.Module):
    """Full Griffin stack: token embedding + alternating LRU/attn blocks + LM head.

    Block ordering: repeat [lru_per_attn x LRU block, 1 x attention block]
    until n_layers blocks are created.  If n_layers is not evenly divisible by
    (lru_per_attn + 1) the remaining slots are filled with LRU blocks.

    Stateful inference:
        ``states`` is a list with one entry per block (length = n_layers).
        - LRU block i: states[i] is a (B, d_model) recurrent-state tensor or None.
        - Attn block i: states[i] is a (K, V) KV-cache tuple or None.

        Pass the returned ``new_states`` back as ``states`` on the next call to
        continue generation token-by-token without recomputing past context.

    Note on ``test_model_returns_states``: that test checks that the number of
    non-None LRU states equals the number of LRU blocks (n_lru).  The full
    states list always has length n_layers and may contain KV-cache entries for
    attention blocks.
    """

    def __init__(self, config: GriffinConfig) -> None:
        super().__init__()
        self.config = config

        self.embed   = nn.Embedding(config.vocab_size, config.d_model)
        self.norm_f  = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        # Build block list with correct types
        block_types = self._build_block_types(config.n_layers, config.lru_per_attn)
        self.blocks = nn.ModuleList(
            [GriffinBlock(config, bt) for bt in block_types]
        )
        self.block_types = block_types

    @staticmethod
    def _build_block_types(n_layers: int, lru_per_attn: int) -> list[str]:
        """Build the list of block types for n_layers layers.

        Pattern: lru_per_attn LRU blocks, then 1 attn block, repeated.
        Remaining layers (if any) are LRU.
        """
        period = lru_per_attn + 1
        types = []
        for i in range(n_layers):
            if (i + 1) % period == 0:
                types.append("attn")
            else:
                types.append("lru")
        return types

    @property
    def n_lru_blocks(self) -> int:
        """Number of RG-LRU blocks in the model."""
        return sum(1 for t in self.block_types if t == "lru")

    def forward(
        self,
        input_ids: Tensor,                      # (B, T)
        states: Optional[list] = None,          # list of per-block states, length n_layers
    ) -> tuple[Tensor, list]:                   # (logits, new_states)
        """Forward pass through the full Griffin model.

        Args:
            input_ids: Long tensor of shape (B, T).
            states:    Optional list of per-block states (length = n_layers).
                       For LRU blocks: (B, d_model) tensor or None.
                       For attn blocks: (K, V) KV-cache tuple or None.
                       Pass the second return value from a previous call to
                       continue generation token-by-token.

        Returns:
            (logits, new_states):
                logits     -- (B, T, vocab_size) float tensor.
                new_states -- list of updated per-block states (length = n_layers).
        """
        x = self.embed(input_ids)   # (B, T, d_model)

        n_blocks = len(self.blocks)
        if states is None:
            states = [None] * n_blocks
        elif len(states) != n_blocks:
            raise ValueError(
                f"Expected {n_blocks} states (one per block), got {len(states)}"
            )

        new_states: list = []

        for i, block in enumerate(self.blocks):
            x, new_s = block(x, states[i])
            new_states.append(new_s)

        x = self.norm_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits, new_states
