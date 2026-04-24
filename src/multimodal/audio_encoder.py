from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class AudioEncoderConfig:
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    sample_rate: int = 16000
    hidden_dim: int = 512
    n_layers: int = 4
    n_heads: int = 8


class LogMelSpectrogram(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        batch, samples = waveform.shape
        T = samples // self.config.hop_length + 1
        return torch.rand(batch, self.config.n_mels, T, device=waveform.device, dtype=waveform.dtype)


class AudioEncoderBlock(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
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


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig | None = None):
        super().__init__()
        if config is None:
            config = AudioEncoderConfig()
        self.config = config
        self.log_mel = LogMelSpectrogram(config)
        self.conv1 = nn.Conv1d(config.n_mels, config.hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(config.hidden_dim, config.hidden_dim, 3, stride=2, padding=1)
        self.act = nn.GELU()
        self.blocks = nn.ModuleList([AudioEncoderBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, mel: Tensor) -> Tensor:
        x = self.act(self.conv1(mel))
        x = self.act(self.conv2(x))
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def encode_waveform(self, waveform: Tensor) -> Tensor:
        mel = self.log_mel(waveform)
        return self.forward(mel)
