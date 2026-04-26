"""Sparse Autoencoder (SAE) — learns sparse dictionary features from activations.

Used to decompose transformer residual stream activations into interpretable
sparse features (Bricken et al. 2023 style).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SAEConfig:
    input_dim: int
    hidden_dim: int
    sparsity_coef: float = 1e-3
    lr: float = 1e-3


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder that learns an overcomplete dictionary of features.

    Architecture:
        encoder: Linear(input_dim, hidden_dim) + ReLU
        decoder: Linear(hidden_dim, input_dim)

    The sparsity penalty (L1 on hidden activations) encourages each input to
    be explained by a small subset of dictionary features.
    """

    def __init__(self, config: SAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim)
        self.decoder = nn.Linear(config.hidden_dim, config.input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode input activations.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            Tuple of (reconstruction, hidden_features):
                reconstruction: Same shape as x — decoded output.
                hidden_features: (..., hidden_dim) — sparse feature activations.
        """
        hidden = F.relu(self.encoder(x))
        recon = self.decoder(hidden)
        return recon, hidden

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction + sparsity loss.

        L = MSE(x, recon) + sparsity_coef * mean(|hidden|)

        Args:
            x: Original activations (..., input_dim).
            recon: Reconstructed activations (..., input_dim).
            hidden: Hidden (feature) activations (..., hidden_dim).

        Returns:
            Scalar loss tensor.
        """
        mse = F.mse_loss(recon, x)
        l1 = self.config.sparsity_coef * hidden.abs().mean()
        return mse + l1

    def get_live_features(
        self,
        activations: torch.Tensor,
        threshold: float = 0.01,
    ) -> torch.Tensor:
        """Return feature indices whose mean activation exceeds threshold.

        A feature is considered "live" if it activates on average across
        the provided sample, suggesting it represents a real pattern.

        Args:
            activations: Input tensor of shape (N, input_dim) — a batch of
                activations to probe.
            threshold: Mean activation threshold; features below this are dead.

        Returns:
            1-D tensor of live feature indices (sorted ascending).
        """
        with torch.no_grad():
            _, hidden = self.forward(activations)
            # hidden: (N, hidden_dim)
            mean_acts = hidden.mean(dim=0)  # (hidden_dim,)
        live = (mean_acts > threshold).nonzero(as_tuple=False).squeeze(1)
        return live


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SAE_REGISTRY: dict = {}
