"""A lightweight diffusion-style language-model head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DiffusionLMHead(nn.Module):
    """Predict token logits from hidden states and a diffusion timestep."""

    def __init__(self, d_model: int, vocab_size: int, timestep_dim: int | None = None) -> None:
        super().__init__()
        if d_model <= 0 or vocab_size <= 0:
            raise ValueError("d_model and vocab_size must be positive")
        timestep_dim = timestep_dim or d_model
        self.timestep_proj = nn.Sequential(
            nn.Linear(timestep_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.timestep_dim = timestep_dim

    def forward(self, hidden_states: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Condition hidden states on diffusion timesteps and emit logits."""
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be 3D, got {hidden_states.shape}")
        if timesteps.dim() != 1 or timesteps.size(0) != hidden_states.size(0):
            raise ValueError("timesteps must be 1D with batch-matching length")
        timestep_emb = sinusoidal_timestep_embedding(timesteps, self.timestep_dim)
        conditioning = self.timestep_proj(timestep_emb).unsqueeze(1)
        conditioned = self.norm(hidden_states + conditioning)
        return self.out_proj(conditioned)
