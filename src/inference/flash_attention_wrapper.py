from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class FlashAttnConfig:
    causal: bool = True
    softmax_scale: float | None = None
    dropout_p: float = 0.0
    window_size: tuple[int, int] = field(default_factory=lambda: (-1, -1))
    alibi_slopes: torch.Tensor | None = None


class FlashAttentionWrapper:
    """Drop-in FlashAttention-2 wrapper with fallback to standard SDPA."""

    def __init__(self, config: FlashAttnConfig | None = None) -> None:
        self.config = config if config is not None else FlashAttnConfig()

    @property
    def is_flash_available(self) -> bool:
        return torch.cuda.is_available() and torch.backends.cuda.flash_sdp_enabled()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = self.config.softmax_scale or (1.0 / math.sqrt(q.size(-1)))
        use_window = self.config.window_size != (-1, -1)

        if self.is_flash_available and not use_window and self.config.alibi_slopes is None:
            dropout = self.config.dropout_p if torch.is_grad_enabled() else 0.0
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=dropout,
                is_causal=self.config.causal if attention_mask is None else False,
                scale=scale,
            )

        mask = attention_mask
        if use_window or self.config.causal:
            mask = self._build_combined_mask(q, k, attention_mask)
        return self._manual_attention(q, k, v, mask)

    def _build_combined_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q_len = q.size(2)
        k_len = k.size(2)
        device = q.device

        mask = torch.zeros(q_len, k_len, device=device, dtype=torch.bool)

        if self.config.causal:
            causal = self._build_causal_mask(q_len, device)
            if q_len != k_len:
                # decode step: q_len < k_len; keep last q_len rows of causal
                causal_full = self._build_causal_mask(k_len, device)
                causal = causal_full[-q_len:, :]
            mask = mask | ~causal

        left, right = self.config.window_size
        if left != -1 or right != -1:
            for i in range(q_len):
                for j in range(k_len):
                    if left != -1 and (i - j) > left:
                        mask[i, j] = True
                    if right != -1 and (j - i) > right:
                        mask[i, j] = True

        combined = mask.unsqueeze(0).unsqueeze(0).float() * (-1e9)
        if attention_mask is not None:
            combined = combined + attention_mask
        return combined

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = self.config.softmax_scale or (1.0 / math.sqrt(q.size(-1)))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores + mask

        weights = torch.softmax(scores, dim=-1)

        if self.config.dropout_p > 0.0 and torch.is_grad_enabled():
            weights = F.dropout(weights, p=self.config.dropout_p)

        return torch.matmul(weights, v)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
