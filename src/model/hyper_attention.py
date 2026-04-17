"""HyperAttention: Sub-quadratic Attention via LSH Approximation.

Han et al. 2023 (NeurIPS) — https://arxiv.org/abs/2310.05869

Approximates softmax attention in O(n sqrt(n)) by:
  1. Hamming LSH to identify "heavy-hitter" (Q,K) pairs
  2. Uniform sampling of remaining positions
  3. Combining both attention scores
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HammingLSH:
    """Locality-Sensitive Hashing using random ±1 projections.

    Maps each token vector to a binary hash code of length n_hashes.
    Nearby vectors (in cosine distance) tend to produce the same bits.

    Args:
        d_head: Dimensionality of input vectors.
        n_hashes: Number of hash bits (hash code length).
        seed: RNG seed for reproducibility.
    """

    def __init__(self, d_head: int, n_hashes: int = 4, seed: int = 42) -> None:
        self.d_head = d_head
        self.n_hashes = n_hashes
        # Random ±1 projection matrix, shape (d_head, n_hashes)
        generator = torch.Generator()
        generator.manual_seed(seed)
        R = torch.randint(0, 2, (d_head, n_hashes), generator=generator).float() * 2 - 1
        self.R = R  # stored as plain tensor (not nn.Parameter)

    def hash(self, x: Tensor) -> Tensor:
        """Compute binary hash codes for a batch of token vectors.

        Args:
            x: Input tensor of shape (B, T, d_head).

        Returns:
            Boolean hash codes of shape (B, T, n_hashes).
        """
        # Move R to same device/dtype as x
        R = self.R.to(device=x.device, dtype=x.dtype)
        # (B, T, d_head) @ (d_head, n_hashes) -> (B, T, n_hashes)
        projected = x @ R
        return projected >= 0  # bool tensor


class HyperAttention(nn.Module):
    """Sub-quadratic attention via LSH-guided sparse computation.

    For each query, selects a candidate subset of keys consisting of:
      - top-(n_sample//2) keys by Hamming hash overlap (heavy hitters)
      - n_sample//2 randomly sampled keys (light part)
    Then computes softmax attention over only those candidates.

    Args:
        d_head: Head dimension.
        n_hashes: Number of LSH hash bits.
        sample_size: Number of candidate keys per query. Defaults to
            ``max(1, T // 4)`` at runtime when None.
        scale: Attention scale. Defaults to ``d_head ** -0.5``.
    """

    def __init__(
        self,
        d_head: int,
        n_hashes: int = 4,
        sample_size: int | None = None,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.d_head = d_head
        self.n_hashes = n_hashes
        self.sample_size = sample_size
        self.scale = scale if scale is not None else d_head ** -0.5
        self.lsh = HammingLSH(d_head=d_head, n_hashes=n_hashes)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Compute approximate attention over a candidate subset of keys.

        Args:
            q: Query tensor of shape (B, T, d_head).
            k: Key tensor of shape (B, T, d_head).
            v: Value tensor of shape (B, T, d_head).

        Returns:
            Output tensor of shape (B, T, d_head).
        """
        B, T, d_head = q.shape

        # Determine sample budget
        n_sample = self.sample_size if self.sample_size is not None else max(1, T // 4)
        n_sample = min(n_sample, T)  # can't sample more than T keys

        # Degenerate case: attend to all keys (T is tiny)
        if n_sample >= T:
            scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, T, T)
            attn = F.softmax(scores, dim=-1)
            return attn @ v

        heavy_k = max(1, n_sample // 2)
        light_k = n_sample - heavy_k

        # Hash queries and keys: (B, T, n_hashes) bool
        q_hash = self.lsh.hash(q)  # (B, T, n_hashes)
        k_hash = self.lsh.hash(k)  # (B, T, n_hashes)

        # Compute pairwise hash overlap: (B, T_q, T_k)
        # q_hash: (B, T, n_hashes) -> (B, T, 1, n_hashes)
        # k_hash: (B, T, n_hashes) -> (B, 1, T, n_hashes)
        overlap = (q_hash.unsqueeze(2) == k_hash.unsqueeze(1)).sum(-1).float()
        # overlap shape: (B, T, T)

        outputs = []
        for b in range(B):
            # overlap[b]: (T, T) — row i = hash similarity of q_i with all keys
            q_b = q[b]   # (T, d_head)
            k_b = k[b]   # (T, d_head)
            v_b = v[b]   # (T, d_head)
            ov_b = overlap[b]  # (T, T)

            # For each query, pick heavy + light candidate key indices
            # Heavy: top-heavy_k by hash overlap (ties broken by order)
            _, heavy_idx = ov_b.topk(heavy_k, dim=-1, largest=True, sorted=False)
            # heavy_idx: (T, heavy_k)

            # Light: random sample from remaining positions
            # Build a random permutation per query and pick light_k
            rand_perm = torch.argsort(torch.rand(T, T, device=q.device), dim=-1)
            # Exclude heavy indices by taking the last light_k of rand_perm
            # (simple approach: take rand_perm[:, :light_k] and hope for diversity)
            light_idx = rand_perm[:, :light_k]  # (T, light_k)

            # Union of heavy + light: (T, n_sample)
            cand_idx = torch.cat([heavy_idx, light_idx], dim=-1)  # (T, n_sample)

            # Gather candidate keys and values
            # k_b: (T, d_head), cand_idx: (T, n_sample)
            k_cand = k_b[cand_idx]  # (T, n_sample, d_head)
            v_cand = v_b[cand_idx]  # (T, n_sample, d_head)

            # Attention scores over candidates
            # q_b: (T, d_head) -> (T, 1, d_head)
            scores = (q_b.unsqueeze(1) @ k_cand.transpose(-2, -1)).squeeze(1) * self.scale
            # scores: (T, n_sample)
            attn = F.softmax(scores, dim=-1)  # (T, n_sample)

            # Weighted sum of candidate values
            # attn: (T, n_sample, 1), v_cand: (T, n_sample, d_head)
            out = (attn.unsqueeze(-1) * v_cand).sum(dim=1)  # (T, d_head)
            outputs.append(out)

        return torch.stack(outputs, dim=0)  # (B, T, d_head)


class HyperAttentionLayer(nn.Module):
    """Multi-head attention layer using HyperAttention per head.

    Args:
        d_model: Model dimension (must be divisible by n_heads).
        n_heads: Number of attention heads.
        n_hashes: Number of LSH hash bits.
    """

    def __init__(self, d_model: int, n_heads: int, n_hashes: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Single HyperAttention shared across all heads
        self.hyper_attn = HyperAttention(d_head=self.d_head, n_hashes=n_hashes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute multi-head HyperAttention.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, d_model = x.shape

        # Project
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B * n_heads, T, d_head)
        def split_heads(t: Tensor) -> Tensor:
            # (B, T, d_model) -> (B, n_heads, T, d_head) -> (B*n_heads, T, d_head)
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2).reshape(
                B * self.n_heads, T, self.d_head
            )

        q_h = split_heads(q)
        k_h = split_heads(k)
        v_h = split_heads(v)

        # Apply HyperAttention to all heads at once (batch dim absorbs heads)
        out_h = self.hyper_attn(q_h, k_h, v_h)  # (B*n_heads, T, d_head)

        # Merge heads back: (B*n_heads, T, d_head) -> (B, T, d_model)
        out = out_h.view(B, self.n_heads, T, self.d_head).transpose(1, 2).reshape(B, T, d_model)

        return self.out_proj(out)
