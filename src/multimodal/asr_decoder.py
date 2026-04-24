from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ASRDecoderConfig:
    vocab_size: int = 51864
    hidden_dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    blank_id: int = 0


@dataclass
class ASRResult:
    token_ids: list[int]
    text: str
    confidence: float
    n_frames: int


class CTCDecoder:
    def __init__(self, blank_id: int = 0):
        self.blank_id = blank_id

    def decode(self, log_probs: Tensor) -> list[int]:
        ids = log_probs.argmax(dim=-1).tolist()
        collapsed: list[int] = []
        prev = None
        for t in ids:
            if t != prev:
                collapsed.append(t)
            prev = t
        return [t for t in collapsed if t != self.blank_id]

    def decode_batch(self, log_probs: Tensor) -> list[list[int]]:
        return [self.decode(log_probs[i]) for i in range(log_probs.shape[0])]


class ASRAttentionBlock(nn.Module):
    def __init__(self, config: ASRDecoderConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = nn.MultiheadAttention(
            config.hidden_dim, config.n_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class ASRDecoder(nn.Module):
    def __init__(self, config: ASRDecoderConfig | None = None, encoder_dim: int = 512):
        super().__init__()
        if config is None:
            config = ASRDecoderConfig()
        self.config = config
        self.input_proj = nn.Linear(encoder_dim, config.hidden_dim)
        self.blocks = nn.ModuleList([ASRAttentionBlock(config) for _ in range(config.n_layers)])
        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)
        self.ctc_decoder = CTCDecoder(blank_id=config.blank_id)

    def forward(
        self,
        encoder_output: Tensor,
        decoder_input_ids: Tensor | None = None,
    ) -> Tensor:
        x = self.input_proj(encoder_output)
        for block in self.blocks:
            x = block(x)
        logits = self.output_proj(x)
        return F.log_softmax(logits, dim=-1)

    def transcribe(self, encoder_output: Tensor) -> list[ASRResult]:
        log_probs = self.forward(encoder_output)
        batch_token_ids = self.ctc_decoder.decode_batch(log_probs)
        results: list[ASRResult] = []
        for i, token_ids in enumerate(batch_token_ids):
            T = log_probs.shape[1]
            confidence = log_probs[i].max(dim=-1).values.mean().exp().item()
            text = " ".join(str(t) for t in token_ids)
            results.append(ASRResult(
                token_ids=token_ids,
                text=text,
                confidence=confidence,
                n_frames=T,
            ))
        return results
