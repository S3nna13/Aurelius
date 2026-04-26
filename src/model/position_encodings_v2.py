"""position_encodings_v2.py — Alternative positional encodings for Aurelius.

Implements ALiBi, sinusoidal, learned, and no-op encodings that generalise
better to sequence lengths unseen at training time than standard absolute
sinusoidal or learned embeddings.

Public API
----------
PosEncConfig           — dataclass holding encoding hyper-parameters
compute_alibi_slopes   — (n_heads,) ALiBi slope tensor
build_alibi_bias       — (n_heads, T, T) causal additive bias for attention logits
build_sinusoidal_encoding — (T, d_model) standard sinusoidal table
LearnedPositionalEncoding — nn.Module: adds learned pos-embeddings to (B,T,d)
ALiBiAttention         — nn.Module: full self-attention with ALiBi bias
get_position_encoding  — dispatcher returning the appropriate tensor
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PosEncConfig:
    """Configuration for positional encoding variants.

    Attributes
    ----------
    n_heads:         Number of attention heads (used by ALiBi).
    max_seq_len:     Maximum sequence length for pre-computed encodings.
    alibi_max_slope: Unused scaling knob kept for API compatibility.
    encoding_type:   One of "alibi", "sinusoidal", "learned", "none".
    """

    n_heads: int = 8
    max_seq_len: int = 2048
    alibi_max_slope: float = 1.0
    encoding_type: str = "alibi"  # "alibi" | "sinusoidal" | "learned" | "none"

    def __post_init__(self) -> None:
        valid = {"alibi", "sinusoidal", "learned", "none"}
        if self.encoding_type not in valid:
            raise ValueError(f"encoding_type must be one of {valid}, got {self.encoding_type!r}")


# ---------------------------------------------------------------------------
# ALiBi helpers
# ---------------------------------------------------------------------------


def compute_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute per-head ALiBi slopes.

    slope_h = 1 / 2^(h * 8 / n_heads)  for h = 1 .. n_heads

    Returns
    -------
    Tensor of shape (n_heads,), all values in (0, 1], strictly decreasing.
    """
    exponents = torch.arange(1, n_heads + 1, dtype=torch.float32)
    slopes = 0.5 ** (exponents * 8.0 / n_heads)
    return slopes


def build_alibi_bias(n_heads: int, seq_len: int) -> torch.Tensor:
    """Build the ALiBi additive bias for causal attention.

    For each head h with slope m_h:
        bias[h, i, j] = m_h * (j - i)   which is <= 0 for j <= i
                       = -inf             for j > i  (causal masking)

    The diagonal (i == j) is 0 (no penalty for self-attention).

    Returns
    -------
    Tensor of shape (n_heads, seq_len, seq_len), dtype float32.
    """
    slopes = compute_alibi_slopes(n_heads)  # (H,)

    # positions: (T,)
    positions = torch.arange(seq_len, dtype=torch.float32)
    # relative position matrix: (T, T);  rel[i, j] = j - i
    rel = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)

    # Causal ALiBi: slope * (j - i), which is <= 0 for j <= i
    # bias[h, i, j] = slopes[h] * rel[i, j]
    bias = slopes.view(n_heads, 1, 1) * rel.unsqueeze(0)  # (H, T, T)

    # Apply causal mask: positions j > i get -inf
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    bias = bias.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

    return bias


# ---------------------------------------------------------------------------
# Sinusoidal encoding
# ---------------------------------------------------------------------------


def build_sinusoidal_encoding(
    seq_len: int,
    d_model: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """Standard fixed sinusoidal position encoding (Vaswani et al. 2017).

    pe[t, 2i]   = sin(t / base^(2i / d_model))
    pe[t, 2i+1] = cos(t / base^(2i / d_model))

    Returns
    -------
    Tensor of shape (seq_len, d_model), values in [-1, 1].
    """
    t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    i = torch.arange(0, d_model, 2, dtype=torch.float32).unsqueeze(0)  # (1, d/2)
    div_term = base ** (i / d_model)  # (1, d/2)

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(t / div_term)
    pe[:, 1::2] = torch.cos(t / div_term)
    return pe


# ---------------------------------------------------------------------------
# Learned positional encoding
# ---------------------------------------------------------------------------


class LearnedPositionalEncoding(nn.Module):
    """Trainable absolute positional embeddings.

    Adds a learned embedding for each position 0 .. T-1 to the input.

    Parameters
    ----------
    max_seq_len: Maximum number of positions.
    d_model:     Embedding / hidden dimension.
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings.

        Parameters
        ----------
        x: (B, T, d_model)

        Returns
        -------
        (B, T, d_model) — input plus position embeddings.
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device)  # (T,)
        pos_emb = self.embedding(positions)  # (T, d_model)
        return x + pos_emb.unsqueeze(0)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# ALiBiAttention
# ---------------------------------------------------------------------------


class ALiBiAttention(nn.Module):
    """Multi-head self-attention with ALiBi positional bias.

    No positional embeddings are added to the token representations;
    instead, the causal ALiBi bias is added directly to the QK^T scores
    before softmax.

    Parameters
    ----------
    d_model: Hidden dimension (must be divisible by n_heads).
    n_heads: Number of attention heads.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"  # noqa: S101
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Pre-compute slopes and register as buffer (not a parameter)
        slopes = compute_alibi_slopes(n_heads)  # (H,)
        self.register_buffer("slopes", slopes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ALiBi-biased causal self-attention.

        Parameters
        ----------
        x: (B, T, d_model)

        Returns
        -------
        (B, T, d_model)
        """
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        # Project and reshape to (B, H, T, D)
        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, H, D).transpose(1, 2)

        q = _split_heads(self.q_proj(x))
        k = _split_heads(self.k_proj(x))
        v = _split_heads(self.v_proj(x))

        # Scaled dot-product scores: (B, H, T, T)
        scale = math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # ALiBi bias: (H, T, T)  — causal; j > i already set to -inf
        alibi = build_alibi_bias(H, T).to(x.device)  # (H, T, T)
        scores = scores + alibi.unsqueeze(0)  # broadcast over B

        # Softmax — -inf entries become 0 probability
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum and merge heads
        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def get_position_encoding(
    config: PosEncConfig,
    d_model: int,
    seq_len: int,
) -> torch.Tensor:
    """Return the positional encoding tensor for the given config.

    Dispatch table
    --------------
    "alibi"      -> build_alibi_bias(config.n_heads, seq_len)   shape (H, T, T)
    "sinusoidal" -> build_sinusoidal_encoding(seq_len, d_model) shape (T, d_model)
    "learned"    -> zeros(seq_len, d_model)  (embedding handles the actual offset)
    "none"       -> zeros(seq_len, d_model)

    Returns
    -------
    The appropriate Tensor (not an nn.Module).
    """
    enc = config.encoding_type
    if enc == "alibi":
        return build_alibi_bias(config.n_heads, seq_len)
    elif enc == "sinusoidal":
        return build_sinusoidal_encoding(seq_len, d_model)
    elif enc in ("learned", "none"):
        return torch.zeros(seq_len, d_model)
    else:
        raise ValueError(f"Unknown encoding_type: {enc!r}")
