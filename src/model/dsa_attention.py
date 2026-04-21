"""DeepSeek Sparse Attention (DSA) — GLM-5 §3.1 (arXiv:2602.15763).
Lightning Indexer selects top-k key positions per query (content-aware).
Two-stage training: dense warm-up (train indexer only) → sparse adaptation.
O(L^2) → O(L*k) attention computation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class DSAConfig:
    d_model: int = 512
    n_heads: int = 8
    top_k: int = 64
    freeze_indexer: bool = False   # True during sparse adaptation if base model is frozen

class LightningIndexer(nn.Module):
    """Content-aware sparsity predictor: learns which key positions each query attends to."""
    def __init__(self, head_dim: int):
        super().__init__()
        self.score = nn.Linear(head_dim, 1, bias=False)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        # k: [B, n_heads, T, head_dim] → scores [B, n_heads, T]
        return self.score(k).squeeze(-1)

class DSAAttention(nn.Module):
    def __init__(self, cfg: DSAConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.indexer = LightningIndexer(self.head_dim)
        if cfg.freeze_indexer:
            for p in self.indexer.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        top_k = min(self.cfg.top_k, T)

        # Lightning Indexer: score key positions to find most relevant ones
        idx_scores = self.indexer(k)  # [B, n_heads, T]
        _, top_idx = idx_scores.topk(top_k, dim=-1)  # [B, n_heads, top_k]

        # Gather sparse K, V
        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        k_sparse = k.gather(2, idx_exp)   # [B, n_heads, top_k, head_dim]
        v_sparse = v.gather(2, idx_exp)

        # Scaled dot-product attention over sparse set
        scale = self.head_dim ** -0.5
        attn = F.softmax(torch.matmul(q, k_sparse.transpose(-1, -2)) * scale, dim=-1)
        out = torch.matmul(attn, v_sparse)  # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, self.cfg.d_model)
        return self.o_proj(out)
