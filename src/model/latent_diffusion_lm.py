"""Latent Diffusion for Token Representations.

Implements a denoising diffusion probabilistic model (DDPM) that operates in
the continuous embedding space of a language model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Noise Schedule
# ---------------------------------------------------------------------------


class NoiseSchedule:
    """DDPM noise schedule (linear or cosine)."""

    def __init__(self, n_steps: int = 1000, schedule: str = "cosine") -> None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if schedule not in ("linear", "cosine"):
            raise ValueError(f"schedule must be 'linear' or 'cosine', got {schedule}")

        self.n_steps = n_steps
        self.schedule = schedule

        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, n_steps)
        else:  # cosine
            # Nichol & Dhariwal (2021) cosine schedule
            steps = n_steps + 1
            t = torch.linspace(0, n_steps, steps)
            f_t = torch.cos(((t / n_steps) + 0.008) / 1.008 * math.pi / 2.0) ** 2
            alphas_cumprod_full = f_t / f_t[0]
            betas = 1.0 - (alphas_cumprod_full[1:] / alphas_cumprod_full[:-1])
            betas = torch.clamp(betas, min=1e-5, max=0.9999)

        self.betas: torch.Tensor = betas
        self.alphas: torch.Tensor = 1.0 - betas
        self.alphas_cumprod: torch.Tensor = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: x_t = sqrt_alpha_bar_t * x0 + sqrt_1_minus_alpha_bar_t * eps.

        Args:
            x0: clean embeddings [B, T, d]
            t:  timestep indices  [B]  (0-indexed, in range [0, n_steps-1])

        Returns:
            x_t:  noisy embeddings [B, T, d]
            noise: the sampled epsilon [B, T, d]
        """
        if x0.dim() != 3:
            raise ValueError(f"x0 must be 3D [B,T,d], got shape {x0.shape}")
        if t.dim() != 1 or t.size(0) != x0.size(0):
            raise ValueError("t must be 1D with length == batch size")

        sqrt_alpha = self.sqrt_alphas_cumprod[t].to(x0.device)  # [B]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].to(x0.device)  # [B]

        # Reshape for broadcasting: [B, 1, 1]
        sqrt_alpha = sqrt_alpha[:, None, None]
        sqrt_one_minus = sqrt_one_minus[:, None, None]

        noise = torch.randn_like(x0)
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise

    def get_variance(self, t: int) -> float:
        """Return the posterior variance beta_tilde at step t."""
        if t == 0:
            return float(self.betas[0].item())
        beta_t = self.betas[t].item()
        alpha_bar_t = self.alphas_cumprod[t].item()
        alpha_bar_prev = self.alphas_cumprod[t - 1].item()
        variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        return float(max(variance, 1e-20))


# ---------------------------------------------------------------------------
# Denoising Network
# ---------------------------------------------------------------------------


def _sinusoidal_embedding(t_normalized: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for scalar timestep values.

    Args:
        t_normalized: float tensor [B] in [0, 1]
        dim: embedding dimension
    Returns:
        [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=t_normalized.device)
        / max(half - 1, 1)
    )
    args = t_normalized.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim or dim-1]
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.size(0), 1, device=emb.device)], dim=-1)
    return emb


class _TransformerBlock(nn.Module):
    """Standard pre-norm transformer block (self-attn + FFN)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x


class DenoisingNetwork(nn.Module):
    """Time-conditioned transformer that predicts noise epsilon.

    Args:
        d_model:  model dimension
        n_layers: number of transformer blocks
        n_heads:  number of attention heads
    """

    def __init__(self, d_model: int, n_layers: int = 4, n_heads: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_layers = n_layers

        time_inner = max(d_model // 4, 1)
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, time_inner),
            nn.SiLU(),
            nn.Linear(time_inner, d_model),
        )

        self.transformer_blocks = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise epsilon given noisy embedding x_t and timestep t.

        Args:
            x_t: noisy embeddings [B, T, d_model]
            t:   integer timestep indices [B]  (0-indexed)

        Returns:
            predicted_noise [B, T, d_model]
        """
        if x_t.dim() != 3:
            raise ValueError(f"x_t must be 3D [B,T,d_model], got {x_t.shape}")

        # Build time conditioning: sinusoidal emb → MLP → [B, d_model]
        t_normalized = t.float() / max(self.d_model - 1, 1)  # rough normalisation
        t_sin = _sinusoidal_embedding(t_normalized, self.d_model)  # [B, d_model]
        t_emb = self.time_embed(t_sin)  # [B, d_model]

        # Add time embedding to every token position
        x = x_t + t_emb.unsqueeze(1)  # [B, T, d_model]

        for block in self.transformer_blocks:
            x = block(x)

        return self.out_norm(x)


# ---------------------------------------------------------------------------
# DDPM Trainer
# ---------------------------------------------------------------------------


class DDPMTrainer:
    """Wraps a DenoisingNetwork with DDPM training and sampling utilities."""

    def __init__(
        self,
        model: DenoisingNetwork,
        schedule: NoiseSchedule,
        lr: float = 1e-4,
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def diffusion_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute DDPM MSE loss.

        Samples t ~ U[1, n_steps], adds noise, predicts noise, returns MSE.

        Args:
            x0: clean embeddings [B, T, d_model]

        Returns:
            scalar loss tensor
        """
        B = x0.size(0)
        # Sample random timesteps in [0, n_steps-1]
        t = torch.randint(0, self.schedule.n_steps, (B,), device=x0.device)

        x_t, noise = self.schedule.q_sample(x0, t)
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    def train_step(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Single gradient step.

        Args:
            embeddings: [B, T, d_model]

        Returns:
            scalar loss tensor (detached)
        """
        self.optimizer.zero_grad()
        loss = self.diffusion_loss(embeddings)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """One reverse diffusion step: compute x_{t-1} from x_t.

        Uses the DDPM reverse formula:
            mu = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
            x_{t-1} = mu + sqrt(variance) * z   (z=0 at t=0)

        Args:
            x_t: noisy embeddings [B, T, d_model]
            t:   current integer timestep (scalar, 0-indexed)

        Returns:
            x_{t-1} [B, T, d_model]
        """
        B = x_t.size(0)
        t_tensor = torch.full((B,), t, dtype=torch.long, device=x_t.device)

        eps_theta = self.model(x_t, t_tensor)  # predicted noise

        beta_t = self.schedule.betas[t].to(x_t.device)
        alpha_t = self.schedule.alphas[t].to(x_t.device)
        sqrt_one_minus = self.schedule.sqrt_one_minus_alphas_cumprod[t].to(x_t.device)

        # Predicted mean
        coef = beta_t / sqrt_one_minus
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * eps_theta)

        if t == 0:
            return mean

        variance = self.schedule.get_variance(t)
        noise = torch.randn_like(x_t)
        return mean + math.sqrt(variance) * noise

    @torch.no_grad()
    def generate(self, shape: tuple, n_steps: int) -> torch.Tensor:
        """Generate samples by iteratively denoising from pure Gaussian noise.

        Args:
            shape:   (B, T, d_model)
            n_steps: number of denoising steps (must be <= schedule.n_steps)

        Returns:
            denoised embeddings [B, T, d_model]
        """
        if n_steps > self.schedule.n_steps:
            raise ValueError(
                f"n_steps ({n_steps}) exceeds schedule n_steps ({self.schedule.n_steps})"
            )
        device = next(self.model.parameters()).device
        x = torch.randn(shape, device=device)

        # Iterate from T-1 down to 0
        for step in reversed(range(n_steps)):
            x = self.p_sample(x, step)

        return x


# ---------------------------------------------------------------------------
# Latent Diffusion Language Model
# ---------------------------------------------------------------------------


class _TokenEncoder(nn.Module):
    """Simple transformer encoder: token_ids → dense embeddings."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Learnable positional encoding (max 2048 positions)
        self.pos_embedding = nn.Embedding(2048, d_model)
        self.blocks = nn.ModuleList([_TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token ids to embeddings.

        Args:
            input_ids: [B, T] integer token ids

        Returns:
            embeddings [B, T, d_model]
        """
        T = input_ids.size(1)
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # [1, T]
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class LatentDiffusionLM(nn.Module):
    """Full latent diffusion language model.

    Encodes token sequences to a continuous latent space, applies DDPM
    denoising in that space, then decodes back to token logits.

    Args:
        d_model:      model (latent) dimension
        vocab_size:   vocabulary size
        n_layers:     transformer encoder layers (shared for encoder & denoiser)
        n_diff_steps: number of diffusion timesteps
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_diff_steps: int = 100,
        n_heads: int = 4,
        schedule: str = "cosine",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.encoder = _TokenEncoder(vocab_size, d_model, n_layers, n_heads)
        self.denoiser = DenoisingNetwork(d_model, n_layers, n_heads)
        self.schedule = NoiseSchedule(n_diff_steps, schedule)
        self.decoder_head = nn.Linear(d_model, vocab_size)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode token ids to latent embeddings.

        Args:
            input_ids: [B, T]

        Returns:
            embeddings [B, T, d_model]
        """
        return self.encoder(input_ids)

    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project latent embeddings to token logits.

        Args:
            embeddings: [B, T, d_model]

        Returns:
            logits [B, T, vocab_size]
        """
        return self.decoder_head(embeddings)

    def diffusion_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """DDPM loss computed on the encoded latent embeddings.

        Args:
            input_ids: [B, T]

        Returns:
            scalar loss tensor
        """
        with torch.no_grad():
            x0 = self.encode(input_ids)

        B = x0.size(0)
        t = torch.randint(0, self.schedule.n_steps, (B,), device=x0.device)
        x_t, noise = self.schedule.q_sample(x0.detach(), t)
        predicted_noise = self.denoiser(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, B: int, T: int) -> torch.Tensor:
        """Generate token logits via diffusion sampling.

        Starts from pure Gaussian noise in latent space, denoises, then decodes.

        Args:
            B: batch size
            T: sequence length

        Returns:
            logits [B, T, vocab_size]
        """
        device = next(self.parameters()).device
        shape = (B, T, self.d_model)
        x = torch.randn(shape, device=device)

        trainer = DDPMTrainer(self.denoiser, self.schedule, lr=0.0)
        x = trainer.generate(shape, self.schedule.n_steps)

        return self.decode(x)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DiffusionConfig:
    """Default configuration for the LatentDiffusionLM."""

    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    n_diff_steps: int = 50
    n_heads: int = 4
    schedule: str = "cosine"
    lr: float = 1e-4
