"""Heavily Compressed Attention (HCA).

Extreme KV compression with dense attention on compressed entries.
No sparse selection — every query attends to all compressed KV entries.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AureliusConfig


class HeavyCompressor(nn.Module):
    """Aggressive KV compression: consolidates m' tokens into one entry.

    Unlike CSA, there is no overlap between neighboring compression windows.
    """

    def __init__(self, d_model: int, head_dim: int, compression_rate: int) -> None:
        super().__init__()
        self.m = compression_rate
        self.w_kv = nn.Linear(d_model, head_dim, bias=False)
        self.w_z = nn.Linear(d_model, head_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(compression_rate, head_dim))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = h.shape
        m = self.m

        c = self.w_kv(h)
        z = self.w_z(h)

        n_compressed = S // m
        compressed = torch.zeros(B, n_compressed, c.shape[-1], device=h.device, dtype=h.dtype)

        for i in range(n_compressed):
            start = i * m
            end = (i + 1) * m
            z_block = z[:, start:end, :]
            c_block = c[:, start:end, :]
            weights = F.softmax(z_block + self.b.unsqueeze(0), dim=1)
            compressed[:, i, :] = (weights * c_block).sum(dim=1)

        k_compressed = compressed
        v_compressed = compressed
        return k_compressed, v_compressed


class HeavilyCompressedAttention(nn.Module):
    """Heavily Compressed Attention — HCA.

    Extreme compression (m' = 128) with dense attention over compressed KV.
    Faster than CSA for very long contexts since no sparse indexer overhead.
    """

    def __init__(self, config: AureliusConfig) -> None:
        super().__init__()
        self.m = config.attention_compression_rate_hca
        self.n_h = config.attention_num_query_heads_hca
        self.c = config.head_dim
        self.d_c = config.attention_query_compression_dim
        self.n_groups = config.attention_output_projection_groups
        self.d_g = config.attention_intermediate_output_dim
        self.n_win = config.attention_sliding_window_size
        self.rope_dim = config.attention_partial_rope_dim
        self.d_model = config.d_model

        self.compressor = HeavyCompressor(config.d_model, config.head_dim, self.m)

        self.w_dq = nn.Linear(config.d_model, self.d_c, bias=False)
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

        k_comp, v_comp = self.compressor(x)
        S_comp = k_comp.shape[1]

        c_q = self.w_dq(x)
        q = self.w_uq(c_q).view(B, S, self.n_h, self.c)

        k_comp = self.kv_norm(k_comp)
        q = self.q_norm(q)

        query_positions = torch.arange(S, device=x.device)[None, :, None, None]
        q = self._apply_rope(q, query_positions)

        k_comp = k_comp.unsqueeze(2)
        v_comp = v_comp.unsqueeze(2)

        scores = torch.einsum("bshc,bkhc->bhsk", q.float(), k_comp.float())
        scores = scores / (self.c**0.5)

        causal_mask = torch.triu(torch.ones(S, S_comp, device=x.device) * float("-inf"), diagonal=1)
        block_map = torch.arange(S_comp, device=x.device).unsqueeze(0) * self.m
        causal_mask = torch.where(
            torch.arange(S, device=x.device).unsqueeze(1) < block_map,
            float("-inf"),
            0.0,
        )
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        sink_adj = self.sink_logits.view(1, self.n_h, 1, 1).exp()
        denom = scores.exp().sum(dim=-1, keepdim=True) + sink_adj
        attn_weights = scores.exp() / denom

        out = torch.einsum("bhsk,bkhc->bshc", attn_weights.to(x.dtype), v_comp)

        out = out.reshape(B, S, self.n_h * self.c)
        group_size = self.n_h // self.n_groups
        group_outs = []
        for g in range(self.n_groups):
            g_start = g * group_size
            g_end = (g + 1) * group_size
            g_out = out[:, :, g_start * self.c : g_end * self.c]
            group_outs.append(self.o_down[g](g_out))
        out = self.o_up(torch.cat(group_outs, dim=-1))

        return out, (k_comp, v_comp)
