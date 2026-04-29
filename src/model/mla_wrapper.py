"""MLA-integrated FlashMLAAttention wrapper compatible with AureliusTransformer.

Wraps FlashMLAAttention to match the GroupedQueryAttention forward signature
(freqs_cis, mask, past_kv) so it can be used in _build_attention().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AureliusConfig
from .flash_mla import FlashMLAAttention, FlashMLAConfig


class MLACompatibleAttention(nn.Module):
    """Wraps FlashMLAAttention with KV cache and RoPE support.

    Accepts same forward signature as GroupedQueryAttention for drop-in
    replacement in TransformerBlock.
    """

    def __init__(self, config: AureliusConfig) -> None:
        super().__init__()
        mla_cfg = FlashMLAConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            kv_lrank=config.mla_kv_lrank,
            q_lrank=config.mla_q_lrank,
            rope_dim=config.mla_rope_dim,
            dropout=config.dropout,
        )
        self.mla = FlashMLAAttention(mla_cfg)
        self.absorbed = False

    def absorb(self) -> None:
        self.mla.absorb_projections()
        self.absorbed = True

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, S, D = x.shape

        if past_kv is not None and S > 1:
            raise ValueError("MLA: KV cache only supports single-token decode")

        if past_kv is not None:
            past_c, past_r = past_kv
            c = self.mla._compress_kv(x)
            c_full = torch.cat([past_c, c], dim=1)
            out = self._forward_with_cache(c_full, x, B, S)
            return out, (c_full, past_r)
        else:
            c = self.mla._compress_kv(x)
            if self.absorbed:
                out = self.mla._forward_absorbed(x, c, B, S)
            else:
                out = self.mla._forward_standard(x, c, B, S)
            return out, (c, c.new_zeros(B, S, 1))

    def _forward_with_cache(self, c: torch.Tensor, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        if self.absorbed:
            Q_abs = torch.einsum("btd,hdk->bhtk", x, self.mla.absorbed_qk)
            c_heads = c.unsqueeze(1)
            scores = torch.matmul(Q_abs, c_heads.transpose(-2, -1)) * self.mla.scale
            total_seq = c.shape[1]
            causal = torch.ones(1, 1, T, total_seq, device=x.device, dtype=torch.bool)
            pos = torch.arange(total_seq, device=x.device).view(1, 1, 1, total_seq)
            q_pos = torch.arange(T, device=x.device).view(1, 1, T, 1)
            causal = q_pos >= (pos - (total_seq - T))
            scores = scores.masked_fill(~causal, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            V = (
                self.mla.v_up(c)
                .view(B, total_seq, self.mla.n_heads, self.mla.head_dim)
                .transpose(1, 2)
            )
            V_fresh = V[:, :, -T:, :]
            out = torch.matmul(attn, V_fresh)
            out = out.transpose(1, 2).contiguous().view(B, T, self.mla.n_heads * self.mla.head_dim)
            return self.mla.out_proj(out)

        Q = self.mla.q_proj(x).view(B, T, self.mla.n_heads, self.mla.head_dim).transpose(1, 2)
        K = (
            self.mla.k_up(c)
            .view(B, c.shape[1], self.mla.n_heads, self.mla.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.mla.v_up(c)
            .view(B, c.shape[1], self.mla.n_heads, self.mla.head_dim)
            .transpose(1, 2)
        )
        total = c.shape[1]
        mask = torch.ones(1, 1, T, total, device=x.device, dtype=torch.bool).tril()
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask[:, :, -T:, :])
