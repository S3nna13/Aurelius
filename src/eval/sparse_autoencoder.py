"""Sparse autoencoder (SAE): learn sparse feature directions from model activations for mechanistic interpretability."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SAEConfig:
    """Configuration for a SparseAutoencoder."""

    d_model: int = 512          # input dimension (model hidden dim)
    n_features: int = 4096      # number of sparse features (> d_model)
    l1_coeff: float = 0.001     # sparsity penalty
    learning_rate: float = 1e-3
    n_steps_warmup: int = 1000
    normalize_decoder: bool = True  # normalize decoder columns to unit norm


class SparseAutoencoder(nn.Module):
    """Overcomplete sparse dictionary autoencoder."""

    def __init__(self, config: SAEConfig) -> None:
        super().__init__()
        self.config = config

        self.W_enc = nn.Parameter(torch.empty(config.d_model, config.n_features))
        self.b_enc = nn.Parameter(torch.zeros(config.n_features))
        self.W_dec = nn.Parameter(torch.empty(config.n_features, config.d_model))
        self.b_dec = nn.Parameter(torch.zeros(config.d_model))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

    def encode(self, x: Tensor) -> Tensor:
        """Encode activations to sparse features.

        Args:
            x: (B, d_model) activations

        Returns:
            (B, n_features) sparse features via ReLU
        """
        x_centered = x - self.b_dec
        features = F.relu((x_centered @ self.W_enc) + self.b_enc)
        return features

    def decode(self, features: Tensor) -> Tensor:
        """Decode sparse features back to activation space.

        Args:
            features: (B, n_features)

        Returns:
            (B, d_model) reconstructed activations
        """
        return (features @ self.W_dec) + self.b_dec

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through SAE.

        Args:
            x: (B, d_model) activations

        Returns:
            (reconstructed, features, x_centered)
        """
        x_centered = x - self.b_dec
        features = F.relu((x_centered @ self.W_enc) + self.b_enc)
        reconstructed = (features @ self.W_dec) + self.b_dec
        return reconstructed, features, x_centered

    def normalize_decoder_weights(self) -> None:
        """Normalize W_dec rows to unit norm in-place."""
        with torch.no_grad():
            norms = self.W_dec.data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.W_dec.data = self.W_dec.data / norms


def sae_loss(
    reconstructed: Tensor,
    original: Tensor,
    features: Tensor,
    l1_coeff: float,
) -> tuple[Tensor, dict]:
    """Compute SAE training loss = MSE reconstruction + L1 sparsity.

    Args:
        reconstructed: (B, d_model) reconstructed activations
        original: (B, d_model) original activations
        features: (B, n_features) sparse feature activations
        l1_coeff: sparsity penalty coefficient

    Returns:
        (total_loss, metrics_dict) where metrics contains recon_loss, sparsity_loss, l0_norm
    """
    recon_loss = F.mse_loss(reconstructed, original)
    sparsity_loss = l1_coeff * features.abs().sum(dim=-1).mean()
    total_loss = recon_loss + sparsity_loss

    l0_norm = (features > 0).float().sum(dim=-1).mean().item()

    metrics = {
        "recon_loss": recon_loss.item(),
        "sparsity_loss": sparsity_loss.item(),
        "l0_norm": l0_norm,
    }

    return total_loss, metrics


class SAETrainer:
    """Train SAE on model activations."""

    def __init__(self, sae: SparseAutoencoder, config: SAEConfig) -> None:
        self.sae = sae
        self.config = config
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
        self.step: int = 0

    def train_step(self, activations: Tensor) -> dict[str, float]:
        """One gradient update step.

        Args:
            activations: (B, d_model) or (B, T, d_model) activation batch

        Returns:
            dict with total_loss, recon_loss, sparsity_loss, l0_norm
        """
        # Flatten if 3D
        if activations.dim() == 3:
            B, T, D = activations.shape
            activations = activations.reshape(B * T, D)

        self.sae.train()
        self.optimizer.zero_grad()

        reconstructed, features, x_centered = self.sae(activations)
        total_loss, metrics = sae_loss(reconstructed, activations, features, self.config.l1_coeff)

        total_loss.backward()
        self.optimizer.step()

        self.step += 1

        if self.config.normalize_decoder:
            self.sae.normalize_decoder_weights()

        return {"total_loss": total_loss.item(), **metrics}


def extract_features_from_model(
    model: nn.Module,
    input_ids: Tensor,
    layer_idx: int,
    sae: SparseAutoencoder,
) -> tuple[Tensor, Tensor]:
    """Extract activations from a specific layer and run through SAE.

    Uses a forward hook on model.blocks[layer_idx].

    Args:
        model: transformer model with .blocks attribute
        input_ids: (B, T) token ids
        layer_idx: which layer to hook
        sae: trained SparseAutoencoder

    Returns:
        (activations, features) — both (B, T, D) and (B, T, n_features)
    """
    captured: list[Tensor] = []

    def hook_fn(module: nn.Module, input: tuple, output) -> None:
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        captured.append(h.detach())

    hook = model.blocks[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        hook.remove()

    activations = captured[0]  # (B, T, d_model)

    B, T, D = activations.shape
    flat = activations.reshape(B * T, D)
    with torch.no_grad():
        features_flat = sae.encode(flat)
    features = features_flat.reshape(B, T, -1)

    return activations, features


def find_top_activating_examples(
    features: Tensor,
    feature_idx: int,
    n_top: int = 10,
) -> Tensor:
    """Find positions where a given feature is most active.

    Args:
        features: (B, T, n_features) or (N, n_features) flat
        feature_idx: which feature to analyze
        n_top: how many top indices to return

    Returns:
        (n_top,) indices of highest activating samples/positions
    """
    if features.dim() == 3:
        B, T, n_features = features.shape
        flat = features.reshape(B * T, n_features)
    else:
        flat = features

    activations = flat[:, feature_idx]
    n_top = min(n_top, activations.shape[0])
    top_indices = torch.topk(activations, k=n_top).indices
    return top_indices


def compute_feature_statistics(features: Tensor) -> dict[str, float]:
    """Analyze feature activations.

    Args:
        features: (N, n_features)

    Returns:
        dict with mean_l0, mean_activation, dead_features, max_activation
    """
    l0_per_sample = (features > 0).float().sum(dim=-1)
    mean_l0 = l0_per_sample.mean().item()

    mean_activation = features.mean().item()

    max_per_feature = features.max(dim=0).values
    dead_features = int((max_per_feature == 0).sum().item())

    max_activation = features.max().item()

    return {
        "mean_l0": mean_l0,
        "mean_activation": mean_activation,
        "dead_features": dead_features,
        "max_activation": max_activation,
    }
