"""Mixture Density Networks (MDN) for uncertainty estimation.

Implements Bishop (1994)-style MDNs:
  - MDNHead: predicts mixture coefficients (pi), means (mu), and stds (sigma)
  - mdn_loss: negative log-likelihood under the mixture of Gaussians
  - mdn_sample: ancestral sampling from the predicted mixture
  - mdn_mean: expected value under the mixture
  - mdn_variance: total variance (law of total variance)
  - MDNModel: wraps an encoder backbone with an MDNHead
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MDNConfig:
    """Configuration for a Mixture Density Network head.

    Args:
        n_components: number of mixture components K
        input_dim:    dimensionality of the input hidden state
        output_dim:   dimensionality of the output D
        hidden_dim:   width of the intermediate projection layer
    """

    n_components: int = 5
    input_dim: int = 64
    output_dim: int = 1
    hidden_dim: int = 128


# ---------------------------------------------------------------------------
# MDN Head
# ---------------------------------------------------------------------------


class MDNHead(nn.Module):
    """Mixture Density Network head.

    Maps a hidden state h of shape (B, input_dim) to mixture parameters:
        pi    — mixing coefficients, softmax-normalised, shape (B, K)
        mu    — component means,                          shape (B, K, D)
        sigma — component standard deviations (> 0),     shape (B, K, D)

    Args:
        config: MDNConfig instance
    """

    def __init__(self, config: MDNConfig) -> None:
        super().__init__()
        self.config = config
        K = config.n_components
        D = config.output_dim
        H = config.hidden_dim

        # Shared hidden projection
        self.hidden = nn.Linear(config.input_dim, H)

        # Per-output heads
        self.pi_head = nn.Linear(H, K)  # mixing logits
        self.mu_head = nn.Linear(H, K * D)  # means
        self.sigma_head = nn.Linear(H, K * D)  # log-std logits

    def forward(self, h: torch.Tensor):
        """Forward pass.

        Args:
            h: hidden state of shape (B, input_dim)

        Returns:
            pi:    (B, K)    — softmax-normalised mixing coefficients
            mu:    (B, K, D) — component means
            sigma: (B, K, D) — component std deviations (positive)
        """
        K = self.config.n_components
        D = self.config.output_dim

        z = F.relu(self.hidden(h))  # (B, H)

        pi = F.softmax(self.pi_head(z), dim=-1)  # (B, K)
        mu = self.mu_head(z).view(-1, K, D)  # (B, K, D)
        sigma = torch.exp(self.sigma_head(z)).view(-1, K, D)  # (B, K, D)

        return pi, mu, sigma


# ---------------------------------------------------------------------------
# MDN loss
# ---------------------------------------------------------------------------


def mdn_loss(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood loss for a mixture of Gaussians.

    Loss = -log sum_k { pi_k * N(target; mu_k, sigma_k^2) }

    Args:
        pi:     (B, K)    mixing coefficients (must sum to 1 along dim=-1)
        mu:     (B, K, D) component means
        sigma:  (B, K, D) component std deviations (positive)
        target: (B, D)    ground-truth targets

    Returns:
        Scalar loss (mean over batch).
    """
    # Expand target for broadcasting: (B, 1, D)
    target = target.unsqueeze(1)  # (B, 1, D)

    # Log-probability under each Gaussian component, summed over D
    # log N(x; mu, sigma) = -0.5 * ((x-mu)/sigma)^2 - log(sigma) - 0.5*log(2pi)
    log_norm = -0.5 * math.log(2.0 * math.pi)
    log_prob = log_norm - sigma.log() - 0.5 * ((target - mu) / sigma) ** 2  # (B, K, D)
    log_prob = log_prob.sum(dim=-1)  # (B, K)  — sum over output dims

    # log sum_k { pi_k * p_k } = logsumexp( log(pi_k) + log_prob_k )
    log_pi = torch.log(pi + 1e-8)  # (B, K)
    log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # (B,)

    return -log_mix.mean()


# ---------------------------------------------------------------------------
# MDN sampling
# ---------------------------------------------------------------------------


def mdn_sample(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Sample from the predicted mixture of Gaussians.

    For each batch element:
      1. Draw a component index k ~ Categorical(pi)
      2. Draw a sample from N(mu_k, sigma_k^2)

    Args:
        pi:    (B, K)    mixing coefficients
        mu:    (B, K, D) component means
        sigma: (B, K, D) component standard deviations

    Returns:
        samples: (B, D) tensor
    """
    B, K, D = mu.shape

    # Sample component indices
    k_idx = torch.multinomial(pi, num_samples=1).squeeze(-1)  # (B,)

    # Gather selected mu and sigma: (B, D)
    k_idx_exp = k_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, D)
    mu_k = mu.gather(1, k_idx_exp).squeeze(1)  # (B, D)
    sigma_k = sigma.gather(1, k_idx_exp).squeeze(1)  # (B, D)

    # Sample from the selected Gaussian
    eps = torch.randn_like(mu_k)
    return mu_k + sigma_k * eps  # (B, D)


# ---------------------------------------------------------------------------
# MDN mean
# ---------------------------------------------------------------------------


def mdn_mean(pi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Expected value of the mixture: E[x] = sum_k pi_k * mu_k.

    Args:
        pi: (B, K)    mixing coefficients
        mu: (B, K, D) component means

    Returns:
        mean: (B, D)
    """
    # pi: (B, K) -> (B, K, 1) for broadcasting
    return (pi.unsqueeze(-1) * mu).sum(dim=1)  # (B, D)


# ---------------------------------------------------------------------------
# MDN variance
# ---------------------------------------------------------------------------


def mdn_variance(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Total variance via the law of total variance.

    Var[x] = E[Var[x|k]] + Var[E[x|k]]
           = sum_k pi_k * sigma_k^2  +  sum_k pi_k * (mu_k - E[x])^2

    Args:
        pi:    (B, K)    mixing coefficients
        mu:    (B, K, D) component means
        sigma: (B, K, D) component standard deviations

    Returns:
        variance: (B, D)  — total variance per output dimension
    """
    pi_exp = pi.unsqueeze(-1)  # (B, K, 1)

    # E[Var] = sum_k pi_k * sigma_k^2
    e_var = (pi_exp * sigma**2).sum(dim=1)  # (B, D)

    # Var[E] = sum_k pi_k * (mu_k - mean)^2
    mean = mdn_mean(pi, mu)  # (B, D)
    diff = mu - mean.unsqueeze(1)  # (B, K, D)
    var_e = (pi_exp * diff**2).sum(dim=1)  # (B, D)

    return e_var + var_e


# ---------------------------------------------------------------------------
# MDN Model (encoder + head)
# ---------------------------------------------------------------------------


class MDNModel(nn.Module):
    """Backbone encoder wrapped with an MDNHead.

    Args:
        encoder: any nn.Module whose forward(x) returns a tensor
                 of shape (B, input_dim) (e.g. a mean-pooled transformer)
        config:  MDNConfig
    """

    def __init__(self, encoder: nn.Module, config: MDNConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.mdn_head = MDNHead(config)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: input to the encoder

        Returns:
            (pi, mu, sigma) — see MDNHead.forward
        """
        h = self.encoder(x)
        return self.mdn_head(h)
