"""Latent diffusion for text: diffuse in continuous embedding space.

Distinct from diffusion_lm.py (which diffuses directly in token-embedding space
with sinusoidal timestep embeddings).  Here we use a dedicated TextEncoder /
TextDecoder pair so that the latent space can differ from the vocabulary
embedding dimension, and the LatentDenoiser uses a simple normalised-timestep
MLP rather than a sinusoidal network.

Components
----------
- LatentDiffusionConfig  : hyper-parameters
- TextEncoder            : token ids -> continuous latents
- TextDecoder            : latents -> token logits
- LatentDenoiser         : predicts noise given (z_t, t)
- ldm_noise_schedule     : returns dict of schedule tensors
- ldm_q_sample           : forward diffusion q(z_t | z_0)
- ldm_loss               : DDPM MSE loss in latent space
- LatentDiffusionTrainer : train_step + DDPM sample -> token ids
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LatentDiffusionConfig:
    """Hyper-parameters for the latent diffusion model."""

    latent_dim: int = 64
    """Dimension of the continuous latent space."""

    n_timesteps: int = 100
    """Total diffusion timesteps T."""

    beta_start: float = 1e-4
    """Beta value at t=0 (linear schedule)."""

    beta_end: float = 0.02
    """Beta value at t=T-1 (linear schedule)."""

    vocab_size: int = 256
    """Vocabulary size (input token range)."""

    seq_len: int = 32
    """Default sequence length."""


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Encode token ids into continuous latent vectors.

    A learned embedding followed by a linear projection into ``latent_dim``.
    The projection allows the latent space to be larger or smaller than the
    native embedding dimension.
    """

    def __init__(self, vocab_size: int, latent_dim: int) -> None:
        super().__init__()
        # Internal embedding dimension chosen as 2 * latent_dim for capacity;
        # a linear projection then maps into the actual latent space.
        embed_dim = max(latent_dim, 32)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, latent_dim)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Map token ids to latent vectors.

        Parameters
        ----------
        input_ids : Tensor, shape (B, T)

        Returns
        -------
        latents : Tensor, shape (B, T, latent_dim)
        """
        return self.proj(self.embedding(input_ids))


class TextDecoder(nn.Module):
    """Decode continuous latents back to per-token logits.

    A single linear layer from ``latent_dim`` to ``vocab_size``.
    """

    def __init__(self, latent_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, vocab_size)

    def forward(self, latents: Tensor) -> Tensor:
        """Map latents to token logits.

        Parameters
        ----------
        latents : Tensor, shape (B, T, latent_dim)

        Returns
        -------
        logits : Tensor, shape (B, T, vocab_size)
        """
        return self.linear(latents)


# ---------------------------------------------------------------------------
# Denoiser
# ---------------------------------------------------------------------------

class LatentDenoiser(nn.Module):
    """Predict the noise added to latents at a given diffusion timestep.

    Architecture (as specified):
        linear(latent_dim + 1 -> 2 * latent_dim) -> relu
        -> linear(2 * latent_dim -> latent_dim)

    The single extra input dimension carries the normalised timestep
    ``t / n_timesteps`` broadcast to every sequence position.
    """

    def __init__(self, latent_dim: int, n_timesteps: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_timesteps = n_timesteps

        hidden_dim = 2 * latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: Tensor, t: Tensor) -> Tensor:
        """Predict noise from noisy latents and timestep.

        Parameters
        ----------
        z_t : Tensor, shape (B, T, latent_dim)
        t   : Tensor, shape (B,) — integer timesteps in [0, n_timesteps)

        Returns
        -------
        noise_pred : Tensor, shape (B, T, latent_dim)
        """
        B, T, _ = z_t.shape
        # Normalise timestep to [0, 1] and broadcast to (B, T, 1)
        t_norm = (t.float() / self.n_timesteps).view(B, 1, 1).expand(B, T, 1)
        # Concatenate along last dim: (B, T, latent_dim + 1)
        inp = torch.cat([z_t, t_norm], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def ldm_noise_schedule(
    n_timesteps: int,
    beta_start: float,
    beta_end: float,
) -> dict[str, Tensor]:
    """Compute a linear noise schedule for LDM.

    Parameters
    ----------
    n_timesteps : int
    beta_start  : float
    beta_end    : float

    Returns
    -------
    dict with keys:
        betas, alphas, alphas_cumprod,
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
    All tensors have shape (n_timesteps,).
    """
    betas = torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }


# ---------------------------------------------------------------------------
# Forward diffusion (q sample)
# ---------------------------------------------------------------------------

def ldm_q_sample(
    z0: Tensor,
    t: Tensor,
    schedule: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Forward diffusion: corrupt z0 to z_t at timestep t.

    z_t = sqrt(alpha_bar_t) * z0 + sqrt(1 - alpha_bar_t) * eps

    Parameters
    ----------
    z0       : Tensor, shape (B, T, latent_dim)
    t        : Tensor, shape (B,) — integer timesteps
    schedule : dict returned by ldm_noise_schedule

    Returns
    -------
    (z_t, noise) : both shape (B, T, latent_dim)
    """
    sqrt_alpha = schedule["sqrt_alphas_cumprod"].to(z0.device)[t]      # (B,)
    sqrt_one_minus = schedule["sqrt_one_minus_alphas_cumprod"].to(z0.device)[t]  # (B,)

    # Broadcast over (T, latent_dim)
    sqrt_alpha = sqrt_alpha.view(-1, 1, 1)
    sqrt_one_minus = sqrt_one_minus.view(-1, 1, 1)

    noise = torch.randn_like(z0)
    z_t = sqrt_alpha * z0 + sqrt_one_minus * noise
    return z_t, noise


# ---------------------------------------------------------------------------
# Diffusion loss
# ---------------------------------------------------------------------------

def ldm_loss(
    denoiser: nn.Module,
    encoder: nn.Module,
    input_ids: Tensor,
    schedule: dict[str, Tensor],
) -> Tensor:
    """DDPM-style MSE loss in latent space.

    Steps:
        1. Encode input_ids to z0 via encoder.
        2. Sample random timesteps t ~ U[0, T).
        3. Forward-diffuse to z_t with true noise eps.
        4. Predict eps_hat = denoiser(z_t, t).
        5. Return MSE(eps_hat, eps).

    Parameters
    ----------
    denoiser  : LatentDenoiser (or any compatible module)
    encoder   : TextEncoder (or any compatible module)
    input_ids : Tensor, shape (B, T)
    schedule  : dict from ldm_noise_schedule

    Returns
    -------
    loss : scalar Tensor
    """
    device = input_ids.device
    B = input_ids.shape[0]
    n_timesteps = schedule["betas"].shape[0]

    # 1. Encode
    z0 = encoder(input_ids)  # (B, T, latent_dim)

    # 2. Sample timesteps
    t = torch.randint(0, n_timesteps, (B,), device=device)

    # 3. Forward diffusion
    z_t, noise = ldm_q_sample(z0, t, schedule)

    # 4. Predict noise
    noise_pred = denoiser(z_t, t)

    # 5. MSE loss
    return F.mse_loss(noise_pred, noise)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LatentDiffusionTrainer:
    """Orchestrates training and sampling for latent text diffusion.

    Parameters
    ----------
    encoder   : TextEncoder
    decoder   : TextDecoder
    denoiser  : LatentDenoiser
    config    : LatentDiffusionConfig
    optimizer : torch.optim.Optimizer covering all trainable parameters
    """

    def __init__(
        self,
        encoder: TextEncoder,
        decoder: TextDecoder,
        denoiser: LatentDenoiser,
        config: LatentDiffusionConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.denoiser = denoiser
        self.config = config
        self.optimizer = optimizer

        self.schedule = ldm_noise_schedule(
            config.n_timesteps, config.beta_start, config.beta_end
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _device(self) -> torch.device:
        return next(self.denoiser.parameters()).device

    def _schedule_to_device(self, device: torch.device) -> dict[str, Tensor]:
        return {k: v.to(device) for k, v in self.schedule.items()}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, input_ids: Tensor) -> dict:
        """One gradient update step.

        Parameters
        ----------
        input_ids : Tensor, shape (B, T)

        Returns
        -------
        dict with keys:
            'loss'          : float
            'timestep_mean' : float — mean sampled timestep
        """
        device = self._device()
        input_ids = input_ids.to(device)
        schedule = self._schedule_to_device(device)

        B = input_ids.shape[0]
        n_timesteps = self.config.n_timesteps

        # Encode
        z0 = self.encoder(input_ids)

        # Sample timesteps (keep for reporting)
        t = torch.randint(0, n_timesteps, (B,), device=device)

        # Forward diffuse
        z_t, noise = ldm_q_sample(z0, t, schedule)

        # Predict noise and compute loss
        self.optimizer.zero_grad()
        noise_pred = self.denoiser(z_t, t)
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "timestep_mean": t.float().mean().item(),
        }

    # ------------------------------------------------------------------
    # Sampling (DDPM reverse process)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int) -> Tensor:
        """Generate token ids via DDPM reverse diffusion in latent space.

        Starts from z_T ~ N(0, I), iteratively denoises to z_0, then
        decodes with TextDecoder and returns the argmax token ids.

        Parameters
        ----------
        batch_size : int
        seq_len    : int

        Returns
        -------
        token_ids : Tensor, shape (batch_size, seq_len), dtype int64
        """
        device = self._device()
        schedule = self._schedule_to_device(device)

        latent_dim = self.config.latent_dim
        n_timesteps = self.config.n_timesteps

        # Start from pure noise
        z = torch.randn(batch_size, seq_len, latent_dim, device=device)

        betas = schedule["betas"]
        alphas = schedule["alphas"]
        alphas_cumprod = schedule["alphas_cumprod"]

        # Reverse pass: T-1, T-2, ..., 0
        for t_int in reversed(range(n_timesteps)):
            t_batch = torch.full((batch_size,), t_int, dtype=torch.long, device=device)

            # Predict noise
            noise_pred = self.denoiser(z, t_batch)

            alpha_bar_t = alphas_cumprod[t_int]
            sqrt_alpha_bar = alpha_bar_t.sqrt()
            sqrt_one_minus = (1.0 - alpha_bar_t).sqrt()

            # Estimate x0 from current z_t
            z0_pred = (z - sqrt_one_minus * noise_pred) / sqrt_alpha_bar.clamp(min=1e-8)

            if t_int == 0:
                z = z0_pred
            else:
                # Posterior mean (DDPM)
                alpha_bar_prev = alphas_cumprod[t_int - 1]
                beta_t = betas[t_int]
                alpha_t = alphas[t_int]

                coef1 = (alpha_bar_prev.sqrt() * beta_t) / (1.0 - alpha_bar_t).clamp(min=1e-8)
                coef2 = (alpha_t.sqrt() * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t).clamp(min=1e-8)
                posterior_mean = coef1 * z0_pred + coef2 * z

                posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp(min=1e-8)
                noise = torch.randn_like(z)
                z = posterior_mean + posterior_var.clamp(min=0.0).sqrt() * noise

        # Decode latents to logits, then argmax for token ids
        logits = self.decoder(z)          # (B, T, vocab_size)
        token_ids = logits.argmax(dim=-1)  # (B, T)
        return token_ids.long()
