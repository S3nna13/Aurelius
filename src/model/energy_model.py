"""Energy-Based Model (EBM) scoring for Aurelius transformer.

Provides an energy function over sequences using the backbone's hidden
representations, with contrastive divergence training and Langevin MCMC
sampling for negative examples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import AureliusConfig
from .transformer import AureliusTransformer


@dataclass
class EBMConfig:
    """Configuration for energy-based model scoring."""

    d_model: int = 64
    energy_hidden: int = 128
    noise_scale: float = 0.1
    n_mcmc_steps: int = 10
    step_size: float = 0.01
    temperature: float = 1.0


class EnergyHead(nn.Module):
    """MLP that maps pooled hidden states to a scalar energy value.

    Architecture: d_model -> energy_hidden (SiLU) -> 1
    Output shape: (B,)
    """

    def __init__(self, config: EBMConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.energy_hidden),
            nn.SiLU(),
            nn.Linear(config.energy_hidden, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute energy from hidden states.

        Args:
            hidden_states: (B, S, d_model) — backbone hidden representations.

        Returns:
            (B,) energy scalar per sequence (mean-pooled then projected).
        """
        # Mean-pool over sequence dimension
        pooled = hidden_states.mean(dim=1)  # (B, d_model)
        energy = self.net(pooled).squeeze(-1)  # (B,)
        return energy


def _get_hidden_states(
    model: AureliusTransformer, input_ids: torch.Tensor
) -> torch.Tensor:
    """Run backbone forward pass and return final hidden states (before lm_head).

    This extracts intermediate representations by running embed -> layers -> norm,
    without the final language model head projection.
    """
    B, S = input_ids.shape
    x = model.embed(input_ids)
    freqs_cis = model.freqs_cis[:S]

    for layer in model.layers:
        x, _ = layer(x, freqs_cis, None, None)

    x = model.norm(x)
    return x


def compute_sequence_energy(
    model: AureliusTransformer,
    energy_head: EnergyHead,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute energy score for a batch of token sequences.

    Args:
        model: Aurelius backbone transformer.
        energy_head: Energy scoring MLP.
        input_ids: (B, S) token ids.

    Returns:
        (B,) energy scalar per sequence.
    """
    hidden = _get_hidden_states(model, input_ids)
    return energy_head(hidden)


def contrastive_divergence_loss(
    pos_energy: torch.Tensor,
    neg_energy: torch.Tensor,
) -> torch.Tensor:
    """Contrastive divergence loss: push positive energy down, negative up.

    L = mean(E_pos) - mean(E_neg)

    Minimizing this makes positive (real) sequences have lower energy
    and negative (fake/noisy) sequences have higher energy.

    Args:
        pos_energy: (B,) energy of positive (real) sequences.
        neg_energy: (B,) energy of negative (noisy/fake) sequences.

    Returns:
        Scalar loss.
    """
    return pos_energy.mean() - neg_energy.mean()


def langevin_step(
    model: AureliusTransformer,
    energy_head: EnergyHead,
    x_ids: torch.Tensor,
    config: EBMConfig,
) -> torch.Tensor:
    """One step of Langevin MCMC in the embedding space.

    Operates in continuous embedding space: converts token ids to embeddings,
    performs a gradient-based step to lower the energy, then finds the nearest
    token ids via cosine similarity.

    Args:
        model: Aurelius backbone transformer.
        energy_head: Energy scoring MLP.
        x_ids: (B, S) current token ids.
        config: EBM configuration with step_size and noise_scale.

    Returns:
        (B, S) updated token ids after one MCMC step.
    """
    # Get embeddings and enable gradients
    embeddings = model.embed(x_ids).detach().requires_grad_(True)

    # Forward through layers + norm + energy head
    B, S = x_ids.shape
    freqs_cis = model.freqs_cis[:S]
    x = embeddings
    for layer in model.layers:
        x, _ = layer(x, freqs_cis, None, None)
    x = model.norm(x)
    energy = energy_head(x).sum()

    # Compute gradient of energy w.r.t. embeddings
    grad = torch.autograd.grad(energy, embeddings, create_graph=False)[0]

    # Langevin step: move in negative gradient direction + noise
    noise = torch.randn_like(embeddings) * config.noise_scale
    new_embeddings = embeddings - config.step_size * grad + noise

    # Project back to nearest tokens via cosine similarity
    embed_weight = model.embed.weight.detach()  # (V, d_model)
    # Normalize for cosine similarity
    new_emb_norm = nn.functional.normalize(new_embeddings, dim=-1)  # (B, S, d)
    weight_norm = nn.functional.normalize(embed_weight, dim=-1)  # (V, d)
    # (B, S, V) similarity scores
    similarity = torch.matmul(new_emb_norm, weight_norm.T)
    new_ids = similarity.argmax(dim=-1)  # (B, S)

    return new_ids


class EBMTrainer:
    """Training wrapper for energy-based model scoring.

    Manages the energy head, optimizer, and contrastive divergence training loop.
    """

    def __init__(
        self,
        model: AureliusTransformer,
        ebm_config: EBMConfig,
        lr: float = 1e-4,
    ) -> None:
        self.model = model
        self.config = ebm_config
        self.energy_head = EnergyHead(ebm_config)
        # Move energy head to same device as model
        device = next(model.parameters()).device
        self.energy_head = self.energy_head.to(device)
        self.optimizer = torch.optim.Adam(self.energy_head.parameters(), lr=lr)

    def train_step(
        self,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> dict[str, float]:
        """One training step of contrastive divergence.

        Args:
            pos_ids: (B, S) positive (real) token sequences.
            neg_ids: (B, S) negative (noisy/fake) token sequences.

        Returns:
            Dict with keys: cd_loss, pos_energy, neg_energy.
        """
        self.energy_head.train()
        self.optimizer.zero_grad()

        pos_energy = compute_sequence_energy(self.model, self.energy_head, pos_ids)
        neg_energy = compute_sequence_energy(self.model, self.energy_head, neg_ids)

        loss = contrastive_divergence_loss(pos_energy, neg_energy)
        loss.backward()
        self.optimizer.step()

        return {
            "cd_loss": loss.item(),
            "pos_energy": pos_energy.mean().item(),
            "neg_energy": neg_energy.mean().item(),
        }
