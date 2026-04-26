"""Dynamic sparse attention: content-based top-k key-value selection via a learned indexer."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DSAConfig:
    d_model: int
    n_heads: int
    head_dim: int
    top_k: int = 64
    indexer_dim: int = 32
    dropout: float = 0.0


class LearnedIndexer(nn.Module):
    """Lightweight low-dimensional projector that scores query-key relevance.

    Args:
        d_model: full model dimension
        indexer_dim: reduced dimension for similarity scoring
        n_heads: number of attention heads
    """

    def __init__(self, d_model: int, indexer_dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.indexer_dim = indexer_dim
        # Project from head_dim space via a single linear over d_model
        self.q_proj = nn.Linear(d_model, n_heads * indexer_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * indexer_dim, bias=False)

    def forward(self, q: Tensor, k: Tensor, top_k: int) -> Tensor:
        """Compute top-k key indices per query position.

        Args:
            q: (B, H, T_q, head_dim) — query tensor
            k: (B, H, T_k, head_dim) — key tensor
            top_k: number of keys to select per query

        Returns:
            LongTensor of shape (B, H, T_q, top_k) with indices in [0, T_k).
        """
        B, H, T_q, head_dim = q.shape
        T_k = k.shape[2]

        # Reconstruct approximate d_model-scale input by transposing and reshaping
        # We project from (B, T, H*head_dim) by reshaping q,k to (B, T, H*head_dim)
        q_flat = q.permute(0, 2, 1, 3).reshape(B, T_q, H * head_dim)
        k_flat = k.permute(0, 2, 1, 3).reshape(B, T_k, H * head_dim)

        # Low-dim projections: (B, T, H*indexer_dim) → (B, H, T, indexer_dim)
        q_idx = self.q_proj(q_flat).view(B, T_q, H, self.indexer_dim).permute(0, 2, 1, 3)
        k_idx = self.k_proj(k_flat).view(B, T_k, H, self.indexer_dim).permute(0, 2, 1, 3)

        # Similarity scores: (B, H, T_q, T_k)
        scores = q_idx @ k_idx.transpose(-1, -2)

        # Top-k indices: (B, H, T_q, top_k)
        actual_k = min(top_k, T_k)
        _, indices = torch.topk(scores, actual_k, dim=-1, sorted=False)
        return indices


class DynamicSparseAttention(nn.Module):
    """Attention that dynamically selects the top-k most relevant key-value pairs
    per query using a learned low-dimensional indexer.

    Args:
        config: DSAConfig with model dimensions and sparsity parameters
    """

    def __init__(self, config: DSAConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        n_heads = config.n_heads
        head_dim = config.head_dim

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.indexer = LearnedIndexer(n_heads * head_dim, config.indexer_dim, n_heads)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(head_dim)

    def _gather_kv(self, k: Tensor, v: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
        """Gather sparse key and value vectors for each query position.

        Args:
            k: (B, H, T_k, head_dim)
            v: (B, H, T_k, head_dim)
            indices: (B, H, T_q, top_k) — indices into the T_k dimension

        Returns:
            k_sparse: (B, H, T_q, top_k, head_dim)
            v_sparse: (B, H, T_q, top_k, head_dim)
        """
        B, H, T_q, top_k = indices.shape
        head_dim = k.shape[-1]

        # Expand indices for gathering: (B, H, T_q, top_k, head_dim)
        idx_exp = indices.unsqueeze(-1).expand(B, H, T_q, top_k, head_dim)

        # k and v need to be expanded along the T_q dimension
        k_exp = k.unsqueeze(2).expand(B, H, T_q, k.shape[2], head_dim)
        v_exp = v.unsqueeze(2).expand(B, H, T_q, v.shape[2], head_dim)

        k_sparse = k_exp.gather(3, idx_exp)
        v_sparse = v_exp.gather(3, idx_exp)
        return k_sparse, v_sparse

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Compute dynamic sparse attention.

        Args:
            x: (B, T, d_model)
            mask: optional attention mask (currently unused in sparse path)

        Returns:
            (B, T, d_model)
        """
        B, T, d_model = x.shape
        H = self.config.n_heads
        head_dim = self.config.head_dim
        top_k = min(self.config.top_k, T)

        # Project and reshape to (B, H, T, head_dim)
        q = self.q_proj(x).view(B, T, H, head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, T, H, head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, H, head_dim).permute(0, 2, 1, 3)

        # Get top-k indices via learned indexer
        indices = self.indexer(q, k, top_k)  # (B, H, T_q, top_k)

        # Gather sparse K, V
        k_sparse, v_sparse = self._gather_kv(k, v, indices)  # (B, H, T_q, top_k, head_dim)

        # Attention scores: (B, H, T_q, 1, head_dim) @ (B, H, T_q, head_dim, top_k)
        # → (B, H, T_q, 1, top_k) → (B, H, T_q, top_k)
        attn_scores = (q.unsqueeze(-2) @ k_sparse.transpose(-1, -2)).squeeze(-2) / self.scale

        # Softmax and dropout over top_k
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: (B, H, T_q, 1, top_k) @ (B, H, T_q, top_k, head_dim)
        # → (B, H, T_q, 1, head_dim) → (B, H, T_q, head_dim)
        out = (attn_weights.unsqueeze(-2) @ v_sparse).squeeze(-2)

        # Reshape to (B, T, d_model) and project
        out = out.permute(0, 2, 1, 3).reshape(B, T, H * head_dim)
        return self.o_proj(out)

    def warmup_mode(self, enabled: bool) -> None:
        """Control parameter freezing for warmup training.

        When enabled, freeze all parameters except the indexer so the indexer
        learns a useful routing signal before the full model trains end-to-end.

        Args:
            enabled: if True, freeze base params and keep indexer trainable;
                     if False, unfreeze all parameters.
        """
        if enabled:
            # Freeze everything first
            for param in self.parameters():
                param.requires_grad_(False)
            # Then unfreeze only the indexer
            for param in self.indexer.parameters():
                param.requires_grad_(True)
        else:
            for param in self.parameters():
                param.requires_grad_(True)
