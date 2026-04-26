"""Hierarchical context compression: compress long sequences into summary tokens (AutoCompressor/ICAE-style)."""  # noqa: E501

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class CompressionConfig:
    compression_ratio: int = 4
    n_summary_tokens: int = 8
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_compress_layers: int = 2
    dropout: float = 0.1


class SegmentCompressor(nn.Module):
    """Compresses a segment of tokens into fixed-size summary tokens."""

    def __init__(self, config: CompressionConfig):
        super().__init__()
        self.config = config
        self.summary_tokens = nn.Parameter(torch.randn(1, config.n_summary_tokens, config.d_model))
        self.cross_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            batch_first=True,
            dropout=config.dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, segment: Tensor) -> Tensor:
        """
        Args:
            segment: (B, T_seg, D)
        Returns:
            (B, n_summary_tokens, D)
        """
        B = segment.size(0)
        # Expand summary tokens to batch
        queries = self.summary_tokens.expand(B, -1, -1)  # (B, n_summary, D)

        # Cross-attend: summaries query into segment
        attn_out, _ = self.cross_attn(queries, segment, segment)
        queries = self.norm1(queries + attn_out)

        # FFN on summaries
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)

        return queries


class HierarchicalCompressor(nn.Module):
    """Multi-level compression for long sequences."""

    def __init__(self, config: CompressionConfig):
        super().__init__()
        self.config = config
        self.compressor = SegmentCompressor(config)
        self.segment_size = config.n_summary_tokens * config.compression_ratio

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            hidden_states: (B, T, D)
        Returns:
            (compressed, original_hidden_states)
            compressed: (B, n_seg * n_summary_tokens, D)
        """
        B, T, D = hidden_states.shape
        seg_size = self.segment_size

        # Pad last segment if needed
        n_seg = math.ceil(T / seg_size)
        pad_len = n_seg * seg_size - T
        if pad_len > 0:
            padding = hidden_states.new_zeros(B, pad_len, D)
            padded = torch.cat([hidden_states, padding], dim=1)
        else:
            padded = hidden_states

        # Split into segments and compress each
        segments = padded.view(B, n_seg, seg_size, D)
        summaries_list = []
        for i in range(n_seg):
            seg = segments[:, i, :, :]  # (B, seg_size, D)
            summary = self.compressor(seg)  # (B, n_summary_tokens, D)
            summaries_list.append(summary)

        # Stack: (B, n_seg, n_summary, D)
        summaries = torch.stack(summaries_list, dim=1)

        # Flatten: (B, n_seg * n_summary, D)
        compressed = summaries.view(B, n_seg * self.config.n_summary_tokens, D)

        return compressed, hidden_states

    def compute_compression_rate(self, original_len: int, compressed_len: int) -> float:
        return original_len / compressed_len


class CompressedContextAttention(nn.Module):
    """Attention that attends to both compressed past and current tokens."""

    def __init__(self, config: CompressionConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.d_model, config.n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, current: Tensor, compressed_past: Tensor) -> Tensor:
        """
        Args:
            current: (B, T_cur, D)
            compressed_past: (B, T_comp, D)
        Returns:
            (B, T_cur, D)
        """
        # Self-attend current tokens
        self_out, _ = self.self_attn(current, current, current)
        current = self.norm1(current + self_out)

        # Cross-attend current → compressed_past, gated
        cross_out, _ = self.cross_attn(current, compressed_past, compressed_past)
        gate_val = torch.sigmoid(self.gate)
        current = self.norm2(current + gate_val * cross_out)

        return current


def compress_and_attend(
    hidden: Tensor,
    compressor: HierarchicalCompressor,
    attn: CompressedContextAttention,
) -> tuple[Tensor, float]:
    """Full pipeline: compress hidden, then attend to compressed.

    Returns:
        (output, compression_rate)
    """
    compressed, original = compressor(hidden)
    output = attn(original, compressed)
    rate = compressor.compute_compression_rate(original.size(1), compressed.size(1))
    return output, rate


class AutoCompressorBlock(nn.Module):
    """Full AutoCompressor-style block."""

    def __init__(self, config: CompressionConfig):
        super().__init__()
        self.compressor = HierarchicalCompressor(config)
        self.attn = CompressedContextAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: Tensor) -> tuple[Tensor, float]:
        """
        Args:
            x: (B, T, D)
        Returns:
            (output, compression_rate)
        """
        output, rate = compress_and_attend(x, self.compressor, self.attn)
        ffn_out = self.ffn(output)
        output = self.norm(output + ffn_out)
        return output, rate
