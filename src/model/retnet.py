"""RetNet-style multi-scale retention layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RetNetConfig:
    d_model: int
    n_heads: int
    head_dim: int

    @property
    def inner_dim(self) -> int:
        return self.n_heads * self.head_dim


def _head_decay_mask(length: int, gammas: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build the causal decay mask used by parallel retention."""
    positions = torch.arange(length, device=device)
    diff = positions[:, None] - positions[None, :]
    causal = diff >= 0
    diff = diff.clamp_min(0).to(dtype=dtype)
    gamma = gammas.to(device=device, dtype=dtype)[:, None, None]
    mask = torch.pow(gamma, diff.unsqueeze(0))
    return mask * causal.unsqueeze(0)


class MultiScaleRetention(nn.Module):
    """Standalone RetNet block with parallel and recurrent execution."""

    def __init__(self, config: RetNetConfig, gammas: torch.Tensor | None = None) -> None:
        super().__init__()
        if config.d_model <= 0 or config.n_heads <= 0 or config.head_dim <= 0:
            raise ValueError("RetNetConfig dimensions must be positive")
        if config.inner_dim != config.d_model:
            raise ValueError(
                f"d_model must equal n_heads * head_dim, got {config.d_model} and {config.inner_dim}"
            )
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.inner_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.inner_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.inner_dim, bias=False)
        self.out_proj = nn.Linear(config.inner_dim, config.d_model, bias=False)
        if gammas is None:
            gammas = torch.linspace(0.9, 0.99, config.n_heads)
        if gammas.shape != (config.n_heads,):
            raise ValueError(f"gammas must have shape ({config.n_heads},), got {tuple(gammas.shape)}")
        self.register_buffer("gammas", gammas)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.config.n_heads, self.config.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.config.n_heads, self.config.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.config.n_heads, self.config.head_dim)
        return q, k, v

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Compute retention with a parallel causal mask."""
        q, k, v = self._project(x)
        scale = 1.0 / math.sqrt(self.config.head_dim)
        scores = torch.einsum("bthd,bshd->bhts", q * scale, k)
        decay = _head_decay_mask(
            x.size(1),
            self.gammas,
            device=x.device,
            dtype=x.dtype,
        )
        weights = scores * decay.unsqueeze(0)
        retained = torch.einsum("bhts,bshd->bthd", weights, v)
        return self.out_proj(retained.reshape(x.size(0), x.size(1), -1))

    def initial_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """Return an empty recurrent retention state."""
        return torch.zeros(
            batch_size,
            self.config.n_heads,
            self.config.head_dim,
            self.config.head_dim,
            device=device,
            dtype=dtype or self.gammas.dtype,
        )

    def forward_step(self, x_t: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one token with recurrent retention state."""
        if x_t.dim() != 2:
            raise ValueError(f"x_t must be 2D (batch, d_model), got {x_t.shape}")
        batch = x_t.size(0)
        if state is None:
            state = self.initial_state(batch, device=x_t.device, dtype=x_t.dtype)

        q = self.q_proj(x_t).view(batch, self.config.n_heads, self.config.head_dim)
        k = self.k_proj(x_t).view(batch, self.config.n_heads, self.config.head_dim)
        v = self.v_proj(x_t).view(batch, self.config.n_heads, self.config.head_dim)

        gamma = self.gammas.to(device=x_t.device, dtype=x_t.dtype).view(1, self.config.n_heads, 1, 1)
        outer = torch.einsum("bhd,bhe->bhde", k, v)
        new_state = gamma * state + outer
        output = torch.einsum("bhd,bhde->bhe", q / math.sqrt(self.config.head_dim), new_state)
        output = self.out_proj(output.reshape(batch, -1))
        return output, new_state

    def forward_recurrent(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute retention sequentially."""
        outputs = []
        current_state = state
        for index in range(x.size(1)):
            out_t, current_state = self.forward_step(x[:, index], current_state)
            outputs.append(out_t)
        return torch.stack(outputs, dim=1), current_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default to the parallel implementation."""
        return self.forward_parallel(x)
