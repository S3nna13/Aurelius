"""BigBird-style sparse attention: local window + global tokens + random keys.

Combines three attention patterns:
  1. Local: each token attends to a symmetric window of size window_size on each side.
  2. Global: the first n_global_tokens tokens attend to all tokens and vice versa.
  3. Random: each non-global token attends to n_random_keys random additional tokens.
"""

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class BigBirdConfig:
    window_size: int = 3  # local attention window (one-sided)
    n_global_tokens: int = 2  # first n tokens attend/are attended globally
    n_random_keys: int = 1  # random keys per query
    d_model: int = 64
    n_heads: int = 2
    head_dim: int = 32
    causal: bool = False  # BigBird is typically non-causal (encoder)


def create_bigbird_mask(
    seq_len: int,
    window_size: int,
    n_global_tokens: int,
    n_random_keys: int,
    seed: int = 0,
) -> Tensor:
    """Create (T, T) boolean attention mask (True = can attend).

    Rules:
    1. Local: token i attends to tokens in [i-window_size, i+window_size]
    2. Global: tokens 0..n_global_tokens-1 attend to ALL tokens and vice versa
    3. Random: each token (non-global) attends to n_random_keys random tokens

    Returns (T, T) bool mask.
    """
    T = seq_len
    mask = torch.zeros(T, T, dtype=torch.bool)

    # 1. Local window: token i attends to [i - window_size, i + window_size]
    rows = torch.arange(T).unsqueeze(1)  # (T, 1)
    cols = torch.arange(T).unsqueeze(0)  # (1, T)
    local = (cols - rows).abs() <= window_size
    mask |= local

    # 2. Global tokens: rows 0..n_global_tokens-1 attend all; columns 0..n_global_tokens-1 attended by all  # noqa: E501
    g = min(n_global_tokens, T)
    if g > 0:
        mask[:g, :] = True  # global tokens attend everything
        mask[:, :g] = True  # everything attends to global tokens

    # 3. Random keys: each non-global token attends to n_random_keys extra random positions
    if n_random_keys > 0:
        rng = random.Random(seed)
        for i in range(g, T):
            # Candidate positions are everything not already attended by token i
            already_attended = mask[i].nonzero(as_tuple=True)[0].tolist()
            already_set = set(already_attended)
            candidates = [j for j in range(T) if j not in already_set]
            if candidates:
                chosen = rng.sample(candidates, min(n_random_keys, len(candidates)))
                for j in chosen:
                    mask[i, j] = True

    return mask


def sparse_attention_with_mask(
    q: Tensor,  # (B, H, T, D)
    k: Tensor,
    v: Tensor,
    mask: Tensor,  # (T, T) bool: True = attend
    scale: float | None = None,
) -> Tensor:
    """Compute attention using a sparse boolean mask.

    Scores = q @ k.T * scale. Set masked=False positions to -inf. Softmax. Weighted v sum.
    Returns (B, H, T, D).
    """
    D = q.shape[-1]
    if scale is None:
        scale = D**-0.5

    # scores: (B, H, T, T)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Expand mask from (T, T) → (1, 1, T, T) for broadcasting
    expanded = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
    scores = scores.masked_fill(~expanded, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    # Handle all-masked rows (softmax of all -inf → nan), replace with 0
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    return torch.matmul(attn_weights, v)  # (B, H, T, D)


def compute_attention_sparsity(mask: Tensor) -> float:
    """Fraction of positions that are masked (False / total)."""
    total = mask.numel()
    attended = int(mask.sum().item())
    return (total - attended) / total


def count_attended_positions(mask: Tensor) -> Tensor:
    """Return (T,) count of positions each token attends to."""
    return mask.sum(dim=-1)  # sum over key dimension


class BigBirdAttention(nn.Module):
    """Multi-head BigBird sparse attention."""

    def __init__(self, cfg: BigBirdConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: Tensor, seed: int = 0) -> Tensor:
        """x: (B, T, D) → (B, T, D).

        Generate BigBird mask, apply sparse_attention_with_mask, project output.
        """
        B, T, D = x.shape
        cfg = self.cfg

        # Build BigBird mask on same device as input
        mask = create_bigbird_mask(
            seq_len=T,
            window_size=cfg.window_size,
            n_global_tokens=cfg.n_global_tokens,
            n_random_keys=cfg.n_random_keys,
            seed=seed,
        ).to(x.device)

        # Project and reshape to (B, H, T, head_dim)
        def project_and_split(proj: nn.Linear) -> Tensor:
            out = proj(x)  # (B, T, D)
            return out.view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)

        Q = project_and_split(self.q_proj)
        K = project_and_split(self.k_proj)
        V = project_and_split(self.v_proj)

        # Sparse attention: (B, H, T, head_dim)
        attn_out = sparse_attention_with_mask(Q, K, V, mask, scale=cfg.head_dim**-0.5)

        # Merge heads and project: (B, T, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_out)


class BigBirdBlock(nn.Module):
    """Transformer block using BigBird attention."""

    def __init__(self, cfg: BigBirdConfig) -> None:
        super().__init__()
        self.attn = BigBirdAttention(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pre-norm: x = x + attn(norm1(x)), x = x + ffn(norm2(x))"""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
