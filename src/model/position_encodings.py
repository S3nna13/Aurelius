"""Additional position encoding variants for Aurelius.

Complements the RoPE/YaRN implementation in attention.py with:
  - ALiBi  (Press et al. 2022) — linear bias on attention scores
  - xPos   (Sun et al. 2022)   — extrapolatable RoPE via position-dependent scaling
  - NoPE   (Kazemnejad 2023)   — no positional encoding (passthrough baseline)
  - T5     (Raffel et al. 2020) — learned relative position biases with log-bucket
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ALiBi
# ---------------------------------------------------------------------------

class ALiBiPositionBias(nn.Module):
    """ALiBi (Attention with Linear Biases, Press et al. 2022).

    Adds a linear bias to attention scores:
        score[i, j] = dot(q_i, k_j) - m_h * |i - j|
    where m_h = 2^(-8 * h / n_heads) for h in 1..n_heads.

    No learned parameters — slopes are registered as a buffer.

    Args:
        n_heads: Number of attention heads.
        max_seq_len: Pre-compute upper bound (not strictly required here).
    """

    def __init__(self, n_heads: int, max_seq_len: int = 8192):
        super().__init__()
        slopes = torch.tensor(
            [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)],
            dtype=torch.float32,
        )
        self.register_buffer("slopes", slopes)

    def get_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute ALiBi bias matrix.

        Returns:
            Tensor of shape (1, n_heads, seq_len, seq_len).
            bias[0, h, i, j] = -slopes[h] * |i - j|
        """
        positions = torch.arange(seq_len, device=device)
        # (seq_len, seq_len) absolute distance matrix
        dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs().float()
        # (1, n_heads, seq_len, seq_len)
        bias = -self.slopes.to(device).view(1, -1, 1, 1) * dist.unsqueeze(0).unsqueeze(0)
        return bias

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """Add ALiBi bias to attention scores.

        Args:
            attn_scores: (B, H, T, T)

        Returns:
            (B, H, T, T) with ALiBi bias added.
        """
        seq_len = attn_scores.size(-1)
        bias = self.get_bias(seq_len, attn_scores.device)
        return attn_scores + bias


# ---------------------------------------------------------------------------
# xPos
# ---------------------------------------------------------------------------

class xPosEmbedding(nn.Module):
    """xPos (Sun et al. 2022): Extrapolatable RoPE via position-dependent scaling.

    Multiplies Q by scale(pos) and K by 1/scale(pos):
        scale(pos) = ((gamma + pos / max_seq_len) / (gamma + 1)) ^ 0.5

    The combined effect on the attention dot-product approximates
    scale(i - j), preserving translation invariance of standard RoPE.

    Args:
        head_dim: Per-head feature dimension.
        max_seq_len: Sequence length upper bound (positions clamped here).
        gamma: Stability constant (default 0.4).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 8192, gamma: float = 0.4):
        super().__init__()
        self.max_seq_len = max_seq_len
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        scales = ((gamma + positions / max_seq_len) / (gamma + 1)) ** 0.5
        self.register_buffer("scales", scales)

    def apply_xpos(
        self,
        q: torch.Tensor,  # (B, H, T, D)
        k: torch.Tensor,  # (B, H, T, D)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale Q by scales[pos] and K by 1/scales[pos].

        Positions are clamped to [0, max_seq_len - 1].

        Returns:
            (scaled_q, scaled_k) with the same shapes as inputs.
        """
        seq_len = q.size(2)
        pos_indices = torch.arange(seq_len, device=q.device).clamp(max=self.max_seq_len - 1)
        scale = self.scales[pos_indices]  # (T,)
        # Broadcast: (1, 1, T, 1)
        scale = scale.view(1, 1, seq_len, 1)
        scaled_q = q * scale
        scaled_k = k / scale
        return scaled_q, scaled_k


# ---------------------------------------------------------------------------
# NoPE
# ---------------------------------------------------------------------------

class NoPE(nn.Module):
    """NoPE (No Positional Encoding, Kazemnejad et al. 2023).

    Passthrough module — returns q and k completely unchanged.
    Useful as a drop-in comparison baseline that forces the model to
    learn position information implicitly via attention patterns.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (q, k) unchanged."""
        return q, k


# ---------------------------------------------------------------------------
# T5 Relative Position Bias
# ---------------------------------------------------------------------------

class T5RelativePositionBias(nn.Module):
    """T5-style learned relative position bias (Raffel et al. 2020).

    Buckets relative position (j - i) into n_buckets buckets using
    log-scale bucketing, then looks up a learned bias per bucket per head.

    Bucket assignment:
      - First half of buckets: exact positions 0 .. n_buckets//2 - 1
      - Second half: log-scale over the range up to max_distance

    Args:
        n_heads: Number of attention heads.
        n_buckets: Total number of relative-position buckets (default 32).
        max_distance: Maximum distance for log-scale bucketing (default 128).
        bidirectional: If True, uses separate buckets for + and - directions.
    """

    def __init__(
        self,
        n_heads: int,
        n_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(n_buckets, n_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Map relative positions to bucket indices.

        Half the buckets cover exact small positions; the other half use
        log-scale for larger distances.
        """
        ret = torch.zeros_like(relative_position)
        n = self.n_buckets

        if self.bidirectional:
            # Use half the buckets per direction
            n = n // 2
            # Positive direction gets an offset of n
            ret = ret + (relative_position > 0).long() * n
            relative_position = relative_position.abs()
        else:
            # Unidirectional — clamp to non-positive (causal)
            relative_position = -torch.clamp(relative_position, max=0)

        # Half exact, half log-scale
        max_exact = n // 2
        is_small = relative_position < max_exact

        # Log-scale bucket index for large distances
        val_if_large = max_exact + (
            torch.log(relative_position.float().clamp(min=1) / max_exact)
            / math.log(self.max_distance / max_exact)
            * (n - max_exact)
        ).long().clamp(max=n - 1)

        ret = ret + torch.where(is_small, relative_position, val_if_large)
        return ret

    def forward(
        self,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute relative position biases.

        Returns:
            Tensor of shape (1, n_heads, seq_len_q, seq_len_k).
        """
        q_pos = torch.arange(seq_len_q, dtype=torch.long, device=device)
        k_pos = torch.arange(seq_len_k, dtype=torch.long, device=device)
        # (seq_len_q, seq_len_k): relative position = k - q
        rel_pos = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)
        buckets = self._relative_position_bucket(rel_pos)  # (Q, K)
        # embedding: (Q, K, n_heads) -> (1, n_heads, Q, K)
        biases = self.embedding(buckets).permute(2, 0, 1).unsqueeze(0)
        return biases


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_position_encoding(
    pe_type: str,
    n_heads: int,
    head_dim: int,
    max_seq_len: int = 8192,
    **kwargs,
) -> nn.Module | None:
    """Factory function for position encodings.

    Args:
        pe_type: One of "alibi", "xpos", "nope", "t5", "rope".
        n_heads: Number of attention heads.
        head_dim: Per-head feature dimension.
        max_seq_len: Maximum sequence length.
        **kwargs: Passed through to the constructor (e.g. gamma for xpos).

    Returns:
        An nn.Module, or None when pe_type is "rope" (handled externally).
    """
    pe_type = pe_type.lower()
    if pe_type == "alibi":
        return ALiBiPositionBias(n_heads=n_heads, max_seq_len=max_seq_len)
    elif pe_type == "xpos":
        gamma = kwargs.get("gamma", 0.4)
        return xPosEmbedding(head_dim=head_dim, max_seq_len=max_seq_len, gamma=gamma)
    elif pe_type == "nope":
        return NoPE()
    elif pe_type == "t5":
        n_buckets = kwargs.get("n_buckets", 32)
        max_distance = kwargs.get("max_distance", 128)
        bidirectional = kwargs.get("bidirectional", True)
        return T5RelativePositionBias(
            n_heads=n_heads,
            n_buckets=n_buckets,
            max_distance=max_distance,
            bidirectional=bidirectional,
        )
    elif pe_type == "rope":
        # RoPE is implemented in attention.py; nothing to return here.
        return None
    else:
        raise ValueError(f"Unknown position encoding type: {pe_type!r}")
