"""Diffusion-LM: score-based diffusion for language model generation.

Implements continuous diffusion over token embeddings following the Diffusion-LM
paradigm (Li et al. 2022).  A score network (noise predictor) is trained with
a simple DDPM-style MSE objective; generation uses the DDPM ancestral sampler.

Components:
- DiffusionConfig: noise schedule and architecture hyper-parameters.
- get_noise_schedule: beta / alpha_cumprod schedules (linear or cosine).
- q_sample: forward (noising) process.
- ScoreNetwork: MLP noise predictor with sinusoidal timestep embedding.
- diffusion_loss: MSE between predicted and actual noise.
- ddpm_sample: DDPM reverse-diffusion sampler.
- DiffusionLMTrainer: wraps embedding + score network for training / generation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DiffusionConfig:
    """Configuration for Diffusion-LM."""

    n_timesteps: int = 100
    """Total number of diffusion timesteps T."""

    beta_start: float = 0.0001
    """Beta value at t=0 (linear schedule)."""

    beta_end: float = 0.02
    """Beta value at t=T-1 (linear schedule)."""

    schedule: str = "linear"
    """Noise schedule type: 'linear' or 'cosine'."""

    d_embed: int = 64
    """Dimensionality of the token embedding space."""


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def get_noise_schedule(config: DiffusionConfig) -> tuple[Tensor, Tensor]:
    """Compute betas and alphas_cumprod for T timesteps.

    Args:
        config: DiffusionConfig instance.

    Returns:
        betas:          shape (T,) -- noise variance at each step.
        alphas_cumprod: shape (T,) -- cumulative product of (1 - beta_t).
    """
    T = config.n_timesteps

    if config.schedule == "linear":
        betas = torch.linspace(config.beta_start, config.beta_end, T)
    elif config.schedule == "cosine":
        # Nichol & Dhariwal cosine schedule
        steps = torch.arange(T + 1, dtype=torch.float64)
        s = 0.008
        alphas_cos = torch.cos(
            ((steps / T) + s) / (1.0 + s) * math.pi * 0.5
        ) ** 2
        alphas_cos = alphas_cos / alphas_cos[0]
        betas_64 = 1.0 - (alphas_cos[1:] / alphas_cos[:-1])
        betas = betas_64.clamp(0.0, 0.999).float()
    else:
        raise ValueError(f"Unknown schedule: {config.schedule!r}. Use 'linear' or 'cosine'.")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return betas, alphas_cumprod


# ---------------------------------------------------------------------------
# Forward (noising) process
# ---------------------------------------------------------------------------

def q_sample(
    x0: Tensor,
    t: Tensor,
    alphas_cumprod: Tensor,
    noise: Tensor | None = None,
) -> Tensor:
    """Forward diffusion: add noise to x0 at timestep t.

    x_t = sqrt(alpha_t) * x0 + sqrt(1 - alpha_t) * noise

    Args:
        x0:             Clean embeddings, shape (B, T_seq, D).
        t:              Integer timestep indices, shape (B,).
        alphas_cumprod: Cumulative alphas, shape (T,).
        noise:          Optional pre-sampled noise; if None, sampled from N(0,I).

    Returns:
        x_t: Noised embeddings, same shape as x0.
    """
    if noise is None:
        noise = torch.randn_like(x0)

    alpha_t = alphas_cumprod[t]                         # (B,)
    sqrt_alpha = alpha_t.sqrt().view(-1, 1, 1)          # (B, 1, 1)
    sqrt_one_minus_alpha = (1.0 - alpha_t).sqrt().view(-1, 1, 1)

    return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise


# ---------------------------------------------------------------------------
# Score network (noise predictor)
# ---------------------------------------------------------------------------

def _sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal timestep embedding, shape (B, dim)."""
    assert dim % 2 == 0, "dim must be even for sinusoidal embedding"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class ScoreNetwork(nn.Module):
    """MLP that predicts the noise added at timestep t.

    Architecture:
        1. Project x_t from d_embed to hidden_dim.
        2. Add sinusoidal timestep embedding (also hidden_dim).
        3. Two hidden layers with SiLU activations.
        4. Project back to d_embed.
    """

    def __init__(
        self,
        d_embed: int,
        hidden_dim: int = 128,
        n_timesteps: int = 100,
    ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.hidden_dim = hidden_dim

        # Timestep embedding MLP
        self.t_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection + MLP
        self.input_proj = nn.Linear(d_embed, hidden_dim)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_embed),
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """Predict noise given noised embeddings and timestep.

        Args:
            x_t: Noised embeddings, shape (B, T_seq, D).
            t:   Integer timestep indices, shape (B,).

        Returns:
            Predicted noise, shape (B, T_seq, D).
        """
        B, T_seq, D = x_t.shape

        # Sinusoidal timestep -> learned projection -> (B, hidden_dim)
        t_emb = _sinusoidal_embedding(t, self.hidden_dim)  # (B, hidden_dim)
        t_emb = self.t_embed(t_emb)                        # (B, hidden_dim)

        # Project x_t: (B, T_seq, D) -> (B, T_seq, hidden_dim)
        h = self.input_proj(x_t)

        # Broadcast timestep embedding across sequence
        h = h + t_emb.unsqueeze(1)  # (B, T_seq, hidden_dim)

        # MLP -> noise prediction (B, T_seq, D)
        return self.net(h)


# ---------------------------------------------------------------------------
# Diffusion loss
# ---------------------------------------------------------------------------

def diffusion_loss(
    score_net: ScoreNetwork,
    x0: Tensor,
    t: Tensor,
    alphas_cumprod: Tensor,
) -> Tensor:
    """Simple denoising score-matching (DDPM) MSE loss.

    Args:
        score_net:      The ScoreNetwork (noise predictor).
        x0:             Clean embeddings, shape (B, T_seq, D).
        t:              Integer timestep indices, shape (B,).
        alphas_cumprod: Cumulative alphas, shape (T,).

    Returns:
        Scalar MSE loss between true noise and predicted noise.
    """
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, alphas_cumprod, noise=noise)
    predicted_noise = score_net(x_t, t)
    return F.mse_loss(predicted_noise, noise)


# ---------------------------------------------------------------------------
# DDPM reverse-diffusion sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddpm_sample(
    score_net: ScoreNetwork,
    shape: tuple[int, ...],
    alphas_cumprod: Tensor,
    betas: Tensor,
    n_steps: int = 20,
) -> Tensor:
    """DDPM ancestral sampler (reverse diffusion).

    Starts from x_T ~ N(0, I) and iteratively denoises over n_steps timesteps
    spaced uniformly across [0, T-1].

    Args:
        score_net:      Trained ScoreNetwork.
        shape:          Output shape (B, T_seq, D).
        alphas_cumprod: Cumulative alphas, shape (T,).
        betas:          Beta schedule, shape (T,).
        n_steps:        Number of reverse steps to run.

    Returns:
        x0 estimate, shape equal to `shape`.
    """
    device = next(score_net.parameters()).device
    T = betas.shape[0]

    # Uniformly spaced timesteps from T-1 -> 0
    timesteps = torch.linspace(T - 1, 0, n_steps, dtype=torch.long)

    x = torch.randn(shape, device=device)

    alphas = 1.0 - betas
    alphas_cumprod = alphas_cumprod.to(device)
    betas = betas.to(device)
    alphas = alphas.to(device)

    for i, step in enumerate(timesteps):
        t_int = int(step.item())
        t_batch = torch.full((shape[0],), t_int, dtype=torch.long, device=device)

        # Predict noise
        predicted_noise = score_net(x, t_batch)

        alpha_t = alphas_cumprod[t_int]
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt()

        # Predicted x0
        x0_pred = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t.clamp(min=1e-8)

        if i == n_steps - 1 or t_int == 0:
            x = x0_pred
        else:
            prev_t = int(timesteps[i + 1].item())
            alpha_prev = alphas_cumprod[prev_t]

            beta_t = betas[t_int]
            coef1 = (alpha_prev.sqrt() * beta_t) / (1.0 - alpha_t).clamp(min=1e-8)
            coef2 = (alphas[t_int].sqrt() * (1.0 - alpha_prev)) / (1.0 - alpha_t).clamp(min=1e-8)
            posterior_mean = coef1 * x0_pred + coef2 * x

            posterior_var = beta_t * (1.0 - alpha_prev) / (1.0 - alpha_t).clamp(min=1e-8)
            noise = torch.randn_like(x)
            x = posterior_mean + posterior_var.clamp(min=0.0).sqrt() * noise

    return x


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DiffusionLMTrainer:
    """Trains a ScoreNetwork over embedded token sequences.

    Args:
        score_net:   ScoreNetwork instance.
        embed_layer: nn.Embedding (or any module) mapping token ids -> embeddings.
        config:      DiffusionConfig.
        optimizer:   A torch optimizer covering score_net (and optionally embed_layer).
    """

    def __init__(
        self,
        score_net: ScoreNetwork,
        embed_layer: nn.Module,
        config: DiffusionConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.score_net = score_net
        self.embed_layer = embed_layer
        self.config = config
        self.optimizer = optimizer

        betas, alphas_cumprod = get_noise_schedule(config)
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod

    def _device(self) -> torch.device:
        return next(self.score_net.parameters()).device

    def train_step(self, input_ids: Tensor) -> dict:
        """Single training step.

        Args:
            input_ids: Token ids, shape (B, T_seq).

        Returns:
            dict with keys 'loss' (float) and 't_mean' (float).
        """
        device = self._device()
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)

        B = input_ids.shape[0]

        # Embed tokens -> x0 (B, T_seq, D)
        x0 = self.embed_layer(input_ids.to(device))

        # Sample random timesteps
        t = torch.randint(0, self.config.n_timesteps, (B,), device=device)

        self.optimizer.zero_grad()
        loss = diffusion_loss(self.score_net, x0, t, self.alphas_cumprod)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "t_mean": t.float().mean().item()}

    def generate(self, n_samples: int, seq_len: int) -> Tensor:
        """Generate embeddings via DDPM reverse diffusion.

        Args:
            n_samples: Batch size.
            seq_len:   Sequence length.

        Returns:
            Generated embeddings, shape (n_samples, seq_len, d_embed).
        """
        device = self._device()
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)

        shape = (n_samples, seq_len, self.config.d_embed)
        self.score_net.eval()
        result = ddpm_sample(
            self.score_net,
            shape,
            self.alphas_cumprod,
            self.betas,
        )
        self.score_net.train()
        return result
