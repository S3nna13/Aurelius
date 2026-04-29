"""Compressed Sparse Attention (CSA).

Hybrid attention: token-level KV compression followed by sparse
top-k selection via a multi-head lightning indexer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AureliusConfig


class TokenLevelCompressor(nn.Module):
    """Compresses m consecutive KV entries into one via weighted combination.

    Each compressed entry is derived from 2m KV entries with overlap,
    achieving a compression factor of 1/m.
    """

    def __init__(self, d_model: int, head_dim: int, compression_rate: int) -> None:
        super().__init__()
        self.m = compression_rate
        self.w_kv_a = nn.Linear(d_model, head_dim, bias=False)
        self.w_kv_b = nn.Linear(d_model, head_dim, bias=False)
        self.w_z_a = nn.Linear(d_model, head_dim, bias=False)
        self.w_z_b = nn.Linear(d_model, head_dim, bias=False)

        self.b_a = nn.Parameter(torch.zeros(compression_rate, head_dim))
        self.b_b = nn.Parameter(torch.zeros(compression_rate, head_dim))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = h.shape
        m = self.m

        c_a = self.w_kv_a(h)
        c_b = self.w_kv_b(h)
        z_a = self.w_z_a(h)
        z_b = self.w_z_b(h)

        n_compressed = S // m
        compressed = torch.zeros(B, n_compressed, c_a.shape[-1], device=h.device, dtype=h.dtype)

        for i in range(n_compressed):
            start = i * m
            end = (i + 1) * m
            z_a_block = z_a[:, start:end, :]
            z_b_block = z_b[:, max(0, start - m) : start, :] if i > 0 else None

            if z_b_block is not None:
                z_block = torch.cat([z_b_block, z_a_block], dim=1)
                b_block = torch.cat([self.b_b.expand(B, -1, -1), self.b_a.expand(B, -1, -1)], dim=1)
                c_block = torch.cat(
                    [c_b[:, max(0, start - m) : start, :], c_a[:, start:end, :]], dim=1
                )
            else:
                z_block = z_a_block
                b_block = self.b_a.expand(B, -1, -1)
                c_block = c_a[:, start:end, :]

            weights = F.softmax(z_block + b_block, dim=1)
            compressed[:, i, :] = (weights * c_block).sum(dim=1)

        k_compressed = compressed
        v_compressed = compressed
        return k_compressed, v_compressed


class LightningIndexer(nn.Module):
    """Sparse indexer that selects top-k compressed KV entries for each query.

    Uses multi-head indexer queries with learned head weights.
    """

    def __init__(
        self,
        d_model: int,
        d_c: int,
        c_i: int,
        n_h_i: int,
        compression_rate: int,
        top_k: int = 512,
    ) -> None:
        super().__init__()
        self.d_c = d_c
        self.c_i = c_i
        self.n_h_i = n_h_i
        self.top_k = top_k

        self.w_dq = nn.Linear(d_model, d_c, bias=False)
        self.w_iuq = nn.Linear(d_c, c_i * n_h_i, bias=False)
        self.w_w = nn.Linear(d_model, n_h_i, bias=False)
        self.w_k = nn.Linear(d_model, c_i, bias=False)

    def forward(
        self,
        h_q: torch.Tensor,
        h_kv: torch.Tensor,
        query_positions: torch.Tensor | None = None,
        compressed_rate: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S_q, D = h_q.shape
        k_dim = h_kv.shape[-1]
        S_k = h_kv.shape[1] // compressed_rate

        c_q = self.w_dq(h_q)
        q_i = self.w_iuq(c_q).view(B, S_q, self.n_h_i, self.c_i)
        w = self.w_w(h_q).view(B, S_q, self.n_h_i)

        k_orig = (
            h_kv[:, : S_k * compressed_rate].reshape(B, S_k, compressed_rate, k_dim).mean(dim=2)
        )
        k_proj = self.w_k(k_orig)
        k_i = k_proj.unsqueeze(1).unsqueeze(3)
        dot = (q_i.unsqueeze(2) * k_i).sum(dim=-1)
        scores = (torch.relu(dot) * w.unsqueeze(2)).sum(dim=-1)

        if query_positions is not None:
            causal_mask = torch.arange(S_k, device=scores.device).unsqueeze(0) < (
                query_positions.unsqueeze(-1) // (S_k // S_q + 1)
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        top_k_actual = min(self.top_k, S_k)
        top_scores, top_indices = torch.topk(scores, top_k_actual, dim=-1)

        return top_scores, top_indices, c_q


class CompressedSparseAttention(nn.Module):
    """Compressed Sparse Attention — CSA.

    Combines token-level KV compression with sparse top-k attention.
    Includes sliding window branch for local fine-grained dependencies.
    """

    def __init__(self, config: AureliusConfig) -> None:
        super().__init__()
        self.m = config.attention_compression_rate_csa
        self.top_k = config.attention_top_k
        self.n_h = config.attention_num_query_heads_csa
        self.c = config.head_dim
        self.d_c = config.attention_query_compression_dim
        self.n_groups = config.attention_output_projection_groups
        self.d_g = config.attention_intermediate_output_dim
        self.n_win = config.attention_sliding_window_size
        self.rope_dim = config.attention_partial_rope_dim
        self.d_model = config.d_model

        self.compressor = TokenLevelCompressor(config.d_model, config.head_dim, self.m)

        self.indexer = LightningIndexer(
            d_model=config.d_model,
            d_c=self.d_c,
            c_i=config.attention_indexer_head_dim,
            n_h_i=config.attention_num_indexer_heads,
            compression_rate=self.m,
            top_k=config.attention_top_k,
        )

        self.w_uq = nn.Linear(self.d_c, self.c * self.n_h, bias=False)

        self.o_down = nn.ModuleList(
            [
                nn.Linear(self.c * self.n_h // self.n_groups, self.d_g, bias=False)
                for _ in range(self.n_groups)
            ]
        )
        self.o_up = nn.Linear(self.d_g * self.n_groups, config.d_model, bias=False)

        self.attn_dropout = config.dropout

        self.kv_norm = nn.LayerNorm(config.head_dim, eps=config.rms_norm_eps)
        self.q_norm = nn.LayerNorm(config.head_dim, eps=config.rms_norm_eps)

        self.sink_logits = nn.Parameter(torch.zeros(self.n_h))

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        from .attention import precompute_rope_frequencies

        d = x.shape[-1]
        rope_dim = min(self.rope_dim, d)
        x_part, x_rope = x[..., : d - rope_dim], x[..., d - rope_dim :]

        x_c = x_rope.reshape(*x_rope.shape[:-1], -1, 2)

        freqs_cis = precompute_rope_frequencies(rope_dim, positions.max() + 1, device=x.device)
        freqs = freqs_cis[positions.squeeze()].unsqueeze(0).unsqueeze(2)

        x_rotated = torch.view_as_real(torch.view_as_complex(x_c.float()) * freqs).flatten(-2)
        x_rotated = x_rotated.to(x.dtype)

        return torch.cat([x_part, x_rotated], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, S, D = x.shape

        k_compressed, v_compressed = self.compressor(x)

        query_positions = torch.arange(S, device=x.device)
        _, top_indices, c_q = self.indexer(x, x, query_positions, compressed_rate=self.m)

        q = self.w_uq(c_q).view(B, S, self.n_h, self.c)

        k_compressed = self.kv_norm(k_compressed)
        q = self.q_norm(q)

        q_pos = query_positions[None, :, None, None]
        q = self._apply_rope(q, q_pos)

        k_comp = k_compressed.unsqueeze(2).unsqueeze(1).expand(B, S, -1, -1, -1)
        k_selected = k_comp.gather(
            2, top_indices.unsqueeze(-1).unsqueeze(-1).expand(B, S, -1, 1, self.c)
        ).squeeze(3)
        v_selected = k_selected

        scores = torch.einsum("bshc,bskc->bhsk", q.float(), k_selected.float())
        scores = scores / (self.c**0.5)

        causal_pos = torch.arange(S, device=x.device).view(1, 1, S, 1)
        block_idx = top_indices.unsqueeze(1)
        causal_mask = causal_pos < (block_idx * self.m).to(causal_pos.dtype)
        causal_mask = (
            causal_mask.expand(B, self.n_h, -1, -1) if causal_mask.shape[1] == 1 else causal_mask
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        sink_adj = self.sink_logits.view(1, self.n_h, 1, 1).exp()
        denom = scores.exp().sum(dim=-1, keepdim=True) + sink_adj
        attn_weights = scores.exp() / denom

        out = torch.einsum("bhsk,bskc->bshc", attn_weights.to(x.dtype), v_selected)

        out = out.view(B, S, self.n_h * self.c)
        group_size = self.n_h // self.n_groups
        group_outs = []
        for g in range(self.n_groups):
            g_start = g * group_size
            g_end = (g + 1) * group_size
            g_out = out[:, :, g_start * self.c : g_end * self.c]
            group_outs.append(self.o_down[g](g_out))
        out = self.o_up(torch.cat(group_outs, dim=-1))

        return out, (k_compressed, v_compressed)
