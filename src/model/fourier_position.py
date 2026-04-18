"""Learnable Fourier positional encodings with learnable frequency and phase."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class FourierPEConfig:
    d_model: int
    max_seq_len: int = 4096
    n_freqs: Optional[int] = None
    learnable: bool = True

    def __post_init__(self) -> None:
        if self.n_freqs is None:
            self.n_freqs = self.d_model // 2


def _init_frequencies(n_freqs: int) -> Tensor:
    """Standard sinusoidal frequency initialisation (1/10000^(2i/n_freqs))."""
    i = torch.arange(n_freqs, dtype=torch.float32)
    return 1.0 / (10000.0 ** (2.0 * i / n_freqs))


class FourierPositionEncoding(nn.Module):
    """Positional encoding using a learnable sinusoidal basis.

    For each position t:
        enc[t] = Linear([sin(freq_i*t + phase_i), cos(freq_i*t + phase_i)] for all i)
    """

    def __init__(self, config: FourierPEConfig) -> None:
        super().__init__()
        self.config = config
        n_freqs = config.n_freqs

        init_freqs = _init_frequencies(n_freqs)
        init_phases = torch.zeros(n_freqs)

        if config.learnable:
            self.frequencies = nn.Parameter(init_freqs)
            self.phases = nn.Parameter(init_phases)
        else:
            self.register_buffer("frequencies", init_freqs)
            self.register_buffer("phases", init_phases)

        # Project 2*n_freqs -> d_model
        self.mix = nn.Linear(2 * n_freqs, config.d_model, bias=False)
        nn.init.normal_(self.mix.weight, std=0.02)

    def _compute_encodings(self, T: int, device: torch.device) -> Tensor:
        positions = torch.arange(T, dtype=torch.float32, device=device)  # (T,)
        freqs = self.frequencies.to(device)   # (n_freqs,)
        phases = self.phases.to(device)        # (n_freqs,)

        # (T, n_freqs)
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0) + phases.unsqueeze(0)
        sins = torch.sin(angles)
        coss = torch.cos(angles)

        # (T, 2*n_freqs)
        basis = torch.cat([sins, coss], dim=-1)

        # (T, d_model)
        return self.mix(basis)

    def forward(self, x: Tensor) -> Tensor:
        _, T, _ = x.shape
        enc = self._compute_encodings(T, x.device)
        return x + enc.unsqueeze(0)

    def get_encodings(self, T: int, device: torch.device) -> Tensor:
        return self._compute_encodings(T, device)
