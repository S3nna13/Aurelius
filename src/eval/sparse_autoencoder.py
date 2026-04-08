"""Sparse Autoencoder (SAE) for mechanistic interpretability.

Implements the Bricken et al. 2023 / Cunningham et al. 2023 approach:
train a bottleneck autoencoder with L1 sparsity on hidden states so that
learned features tend to be monosemantic (one concept per feature).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    """Configuration for a SparseAutoencoder."""

    d_hidden: int = 64          # input dimension (d_model of the LLM)
    n_features: int = 256       # dictionary size (>> d_hidden, typically 4-16x)
    l1_coeff: float = 1e-3      # sparsity penalty weight
    normalize_decoder: bool = True  # keep decoder columns unit-norm


class SparseAutoencoder(nn.Module):
    """Single-layer sparse autoencoder.

    Architecture:
    - Encoder: Linear(d_hidden, n_features) + ReLU → sparse feature activations f
    - Decoder: Linear(n_features, d_hidden, bias=False) → reconstruction x_hat
    - Pre-encoder bias b_pre: subtracted before encoding (centering trick)

    Loss = reconstruction_loss + l1_coeff * L1(f)
    reconstruction_loss = MSE(x_hat, x - b_pre)  [normalized by d_hidden]

    Decoder columns are normalized to unit norm if normalize_decoder=True.
    """

    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Pre-encoder bias (centering trick)
        self.b_pre = nn.Parameter(torch.zeros(cfg.d_hidden))

        # Encoder: Linear(d_hidden, n_features) + ReLU
        self.encoder = nn.Linear(cfg.d_hidden, cfg.n_features, bias=True)

        # Decoder: Linear(n_features, d_hidden, bias=False)
        self.decoder = nn.Linear(cfg.n_features, cfg.d_hidden, bias=False)

        # Initialize weights
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.encoder.bias)

        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)
        # Normalize decoder columns to unit norm at init
        self.normalize_decoder_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_hidden) → f: (..., n_features), sparse activations via ReLU."""
        x_centered = x - self.b_pre
        return F.relu(self.encoder(x_centered))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """f: (..., n_features) → x_hat: (..., d_hidden)."""
        return self.decoder(f)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns (x_hat, f, metrics) where:
        - x_hat: reconstruction of (x - b_pre)
        - f: sparse feature activations
        - metrics: {'reconstruction_loss': float, 'l1_loss': float,
                    'total_loss': float, 'sparsity': float (fraction of active features),
                    'l0': float (mean number of active features per token)}
        """
        x_centered = x - self.b_pre

        # Encode
        f = F.relu(self.encoder(x_centered))

        # Decode
        x_hat = self.decoder(f)

        # Losses
        reconstruction_loss = F.mse_loss(x_hat, x_centered) / self.cfg.d_hidden
        l1_loss = self.cfg.l1_coeff * f.abs().mean()
        total_loss = reconstruction_loss + l1_loss

        # Sparsity metrics
        # sparsity: fraction of (token, feature) pairs that are active
        active = (f > 0).float()
        sparsity = active.mean().item()
        # l0: mean number of active features per token
        l0 = active.sum(dim=-1).mean().item()

        metrics = {
            "reconstruction_loss": reconstruction_loss.item(),
            "l1_loss": l1_loss.item(),
            "total_loss": total_loss.item(),
            "sparsity": sparsity,
            "l0": l0,
        }

        return x_hat, f, metrics

    def normalize_decoder_(self) -> None:
        """In-place normalize decoder weight columns to unit norm.

        decoder.weight has shape (d_hidden, n_features); columns = dim=0.
        """
        self.decoder.weight.data = F.normalize(
            self.decoder.weight.data, dim=0, p=2
        )

    def feature_activation_stats(self, x: torch.Tensor) -> dict:
        """Run encoder on x, return per-feature statistics.

        Returns:
            {'mean_activation': Tensor(n_features),
             'activation_frequency': Tensor(n_features),
             'max_activation': Tensor(n_features)}
        """
        with torch.no_grad():
            f = self.encode(x)
            # Flatten to (N, n_features)
            f_flat = f.reshape(-1, self.cfg.n_features)

            mean_activation = f_flat.mean(dim=0)
            activation_frequency = (f_flat > 0).float().mean(dim=0)
            max_activation = f_flat.max(dim=0).values

        return {
            "mean_activation": mean_activation,
            "activation_frequency": activation_frequency,
            "max_activation": max_activation,
        }


class SAETrainer:
    """Train a SparseAutoencoder on hidden states extracted from a language model.

    Args:
        sae: SparseAutoencoder
        optimizer: torch.optim.Optimizer (Adam with lr=1e-4 is typical)
        normalize_every: int (normalize decoder columns every N steps, default 100)
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        optimizer: torch.optim.Optimizer,
        normalize_every: int = 100,
    ) -> None:
        self.sae = sae
        self.optimizer = optimizer
        self.normalize_every = normalize_every
        self._step = 0

    def train_step(self, hidden_states: torch.Tensor) -> dict:
        """One gradient update.

        hidden_states: (B, T, d_hidden) or (N, d_hidden).
        Flatten to (N, d_hidden), forward, backward, step, normalize if due.
        Returns metrics dict from forward.
        """
        # Flatten to (N, d_hidden)
        if hidden_states.dim() == 3:
            B, T, D = hidden_states.shape
            hidden_states = hidden_states.reshape(B * T, D)

        self.sae.train()
        self.optimizer.zero_grad()

        x_hat, f, metrics = self.sae(hidden_states)

        # Recompute total_loss as a tensor for backward
        x_centered = hidden_states - self.sae.b_pre
        recon_loss = F.mse_loss(x_hat, x_centered) / self.sae.cfg.d_hidden
        l1_loss = self.sae.cfg.l1_coeff * f.abs().mean()
        total_loss = recon_loss + l1_loss

        total_loss.backward()
        self.optimizer.step()

        self._step += 1
        if self.sae.cfg.normalize_decoder and (self._step % self.normalize_every == 0):
            self.sae.normalize_decoder_()

        return metrics

    def train_on_activations(
        self,
        activations: list[torch.Tensor],
        n_steps: int,
    ) -> list[dict]:
        """Cycle through activations list for n_steps total.

        Returns list of per-step metric dicts.
        """
        all_metrics: list[dict] = []
        n = len(activations)
        for i in range(n_steps):
            batch = activations[i % n]
            metrics = self.train_step(batch)
            all_metrics.append(metrics)
        return all_metrics


def extract_sae_features_from_model(
    model: nn.Module,
    sae: SparseAutoencoder,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Hook into model.layers[layer_idx] output to capture hidden states,
    run SAE encoder on them, return feature activations (B, T, n_features).

    Uses a forward hook: register → forward → remove.
    """
    hidden: list[torch.Tensor] = []

    def hook_fn(module: nn.Module, input: tuple, output: tuple) -> None:
        # TransformerBlock returns (x, kv_cache); grab x
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        hidden.append(h.detach())

    hook = model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        hook.remove()

    h = hidden[0]  # (B, T, d_hidden)
    with torch.no_grad():
        features = sae.encode(h)  # (B, T, n_features)

    return features
