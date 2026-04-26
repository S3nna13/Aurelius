"""Contrastive representation learning: NT-Xent loss, MoCo momentum encoder, and supervised contrastive."""  # noqa: E501

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive representation learning."""

    temperature: float = 0.07
    """Contrastive temperature scaling (lower = sharper distribution)."""

    momentum: float = 0.999
    """Momentum coefficient for MoCo momentum encoder EMA update."""

    queue_size: int = 256
    """Size of the MoCo negative queue (number of stored negatives)."""

    projection_dim: int = 128
    """Output dimension of the projection head."""

    use_l2_normalize: bool = True
    """Whether to L2-normalize embeddings before computing similarities."""

    n_positives: int = 2
    """Number of positive views per anchor (SimCLR uses 2)."""


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head for contrastive learning.

    Maps pooled sequence representations to a lower-dimensional
    contrastive embedding space.
    """

    def __init__(self, d_model: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project pooled representation.

        Args:
            x: (B, D) pooled sequence representation.

        Returns:
            (B, output_dim) projected embedding.
        """
        return self.layers(x)


def nt_xent_loss(z1: Tensor, z2: Tensor, temperature: float = 0.07) -> Tensor:
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss from SimCLR.

    Positive pairs: (z1[i], z2[i]) — two augmented views of the same sample.
    Negative pairs: all other cross-sample combinations within the batch.

    Args:
        z1: (B, D) L2-normalized embeddings for view 1.
        z2: (B, D) L2-normalized embeddings for view 2.
        temperature: Temperature scaling factor.

    Returns:
        Scalar loss averaged over all anchor-positive pairs.
    """
    B = z1.shape[0]
    device = z1.device

    # Concatenate both views: (2B, D)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=-1)

    # (2B, 2B) cosine similarity matrix, scaled by temperature
    sim = z @ z.T / temperature

    # Mask out self-similarity (diagonal)
    mask_self = torch.eye(2 * B, dtype=torch.bool, device=device)
    sim = sim.masked_fill(mask_self, float("-inf"))

    # Positive labels: for i in [0, B), positive is i+B; for i in [B, 2B), positive is i-B
    labels = torch.cat(
        [
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B, device=device),
        ]
    )

    return F.cross_entropy(sim, labels)


def supervised_contrastive_loss(
    features: Tensor, labels: Tensor, temperature: float = 0.07
) -> Tensor:
    """Supervised Contrastive Loss (Khosla et al. 2020).

    Uses class labels to define positive pairs (same class) and
    negative pairs (different class) instead of data augmentations.

    Args:
        features: (B, D) L2-normalized feature embeddings.
        labels: (B,) long tensor of class labels.
        temperature: Temperature scaling factor.

    Returns:
        Scalar loss.
    """
    B = features.shape[0]
    device = features.device

    features = F.normalize(features, dim=-1)

    # (B, B) similarity matrix
    sim = features @ features.T / temperature

    # Build mask: same class but not self
    labels_col = labels.unsqueeze(0)  # (1, B)
    labels_row = labels.unsqueeze(1)  # (B, 1)
    pos_mask = (labels_row == labels_col).float()  # (B, B)

    # Zero out self-pairs
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = pos_mask.masked_fill(self_mask, 0.0)

    # For numerical stability, subtract max before exp
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim)

    # Exclude self from denominator
    exp_sim_no_self = exp_sim.masked_fill(self_mask, 0.0)
    log_denom = torch.log(exp_sim_no_self.sum(dim=1, keepdim=True) + 1e-8)

    # Log probability of each pair
    log_prob = sim - log_denom

    # For anchors with no positives (all-same-class or singleton), return 0 contribution
    n_positives = pos_mask.sum(dim=1)  # (B,)
    has_positives = n_positives > 0

    if not has_positives.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Mean log-prob over positives per anchor
    per_anchor_loss = -(pos_mask * log_prob).sum(dim=1) / (n_positives + 1e-8)

    # Average over anchors that have at least one positive
    loss = per_anchor_loss[has_positives].mean()
    return loss


class MomentumEncoder(nn.Module):
    """MoCo-style momentum encoder with a FIFO negative queue.

    Maintains a slowly-updated copy of the encoder (momentum encoder)
    to generate consistent negative representations. Negatives are stored
    in a fixed-size queue that is updated each training step.
    """

    def __init__(self, encoder: nn.Module, config: ContrastiveConfig) -> None:
        super().__init__()
        self.config = config

        # Online encoder (receives gradients)
        self.encoder = encoder

        # Momentum encoder: deep copy, frozen
        self.momentum_encoder = copy.deepcopy(encoder)
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False

        # Negative queue: (queue_size, projection_dim), L2-normalized random init
        queue_init = F.normalize(torch.randn(config.queue_size, config.projection_dim), dim=-1)
        self.register_buffer("queue", queue_init)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_momentum_encoder(self) -> None:
        """EMA update of the momentum encoder parameters.

        momentum_param = momentum * momentum_param + (1 - momentum) * online_param
        """
        m = self.config.momentum
        for param, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = m * param_m.data + (1.0 - m) * param.data

    @torch.no_grad()
    def enqueue_and_dequeue(self, keys: Tensor) -> None:
        """Add new key embeddings to the queue (FIFO), update pointer.

        Args:
            keys: (B, projection_dim) L2-normalized key embeddings.
        """
        B = keys.shape[0]
        ptr = int(self.queue_ptr.item())

        # Handle wrap-around if batch doesn't fit exactly
        if ptr + B <= self.config.queue_size:
            self.queue[ptr : ptr + B] = keys
        else:
            # Wrap around
            tail = self.config.queue_size - ptr
            self.queue[ptr:] = keys[:tail]
            self.queue[: B - tail] = keys[tail:]

        ptr = (ptr + B) % self.config.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, query_features: Tensor, key_features: Tensor) -> tuple[Tensor, Tensor]:
        """Compute query and key representations.

        Query goes through online encoder; key goes through momentum encoder.

        Args:
            query_features: (B, D) input features for query branch.
            key_features: (B, D) input features for key branch.

        Returns:
            Tuple of (query_out, key_out), each (B, projection_dim).
        """
        query_out = self.encoder(query_features)
        with torch.no_grad():
            key_out = self.momentum_encoder(key_features)

        if self.config.use_l2_normalize:
            query_out = F.normalize(query_out, dim=-1)
            key_out = F.normalize(key_out, dim=-1)

        return query_out, key_out


def moco_loss(query: Tensor, key: Tensor, queue: Tensor, temperature: float = 0.07) -> Tensor:
    """MoCo contrastive loss.

    Positive: each (query[i], key[i]) pair.
    Negatives: query[i] against all entries in the queue.

    Args:
        query: (B, D) L2-normalized query embeddings.
        key: (B, D) L2-normalized key embeddings.
        queue: (K, D) L2-normalized negative queue embeddings.
        temperature: Temperature scaling factor.

    Returns:
        Scalar loss.
    """
    B = query.shape[0]
    device = query.device

    # Positive logits: (B, 1)
    l_pos = (query * key).sum(dim=-1, keepdim=True) / temperature

    # Negative logits: (B, K)
    l_neg = (query @ queue.T) / temperature

    # Concatenate: (B, 1+K); positive is always at index 0
    logits = torch.cat([l_pos, l_neg], dim=1)

    labels = torch.zeros(B, dtype=torch.long, device=device)

    return F.cross_entropy(logits, labels)


class ContrastiveTrainer:
    """Training loop for contrastive representation learning.

    Supports SimCLR-style (NT-Xent) and Supervised Contrastive training.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ContrastiveConfig,
        optimizer: torch.optim.Optimizer,
        d_model: int,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.d_model = d_model

        self.proj_head = ProjectionHead(d_model, d_model, config.projection_dim)

        # Linear to map logits (vocab_size) back to d_model for representation
        # Will be lazily initialized once we know vocab_size from a forward pass
        self._vocab_proj: nn.Linear | None = None

    def _get_vocab_proj(self, vocab_size: int) -> nn.Linear:
        """Lazily initialize the vocab->d_model projection."""
        if self._vocab_proj is None or self._vocab_proj.in_features != vocab_size:
            self._vocab_proj = nn.Linear(vocab_size, self.d_model)
            # Move to same device as model
            device = next(self.model.parameters()).device
            self._vocab_proj = self._vocab_proj.to(device)
        return self._vocab_proj

    def get_representation(self, input_ids: Tensor) -> Tensor:
        """Extract a pooled representation from the model.

        Runs a forward pass, pools the logits over the sequence dimension,
        then projects to d_model size.

        Args:
            input_ids: (B, T) token IDs.

        Returns:
            (B, d_model) representation tensor.
        """
        loss, logits, pkv = self.model(input_ids)
        # logits: (B, T, V) — pool over T
        pooled = logits.mean(dim=1)  # (B, V)
        vocab_proj = self._get_vocab_proj(pooled.shape[-1])
        return vocab_proj(pooled)  # (B, d_model)

    def train_step_simclr(self, input_ids: Tensor, augmented_ids: Tensor) -> dict[str, float]:
        """One SimCLR training step using NT-Xent loss.

        Args:
            input_ids: (B, T) original token IDs.
            augmented_ids: (B, T) augmented token IDs (second view).

        Returns:
            Dict with keys "loss" and "temperature".
        """
        self.model.train()
        self.proj_head.train()
        self.optimizer.zero_grad()

        z1 = self.get_representation(input_ids)
        z2 = self.get_representation(augmented_ids)

        z1 = self.proj_head(z1)
        z2 = self.proj_head(z2)

        if self.config.use_l2_normalize:
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)

        loss = nt_xent_loss(z1, z2, self.config.temperature)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "temperature": self.config.temperature}

    def train_step_supcon(self, input_ids: Tensor, labels: Tensor) -> dict[str, float]:
        """One Supervised Contrastive training step.

        Args:
            input_ids: (B, T) token IDs.
            labels: (B,) class labels.

        Returns:
            Dict with key "loss".
        """
        self.model.train()
        self.proj_head.train()
        self.optimizer.zero_grad()

        z = self.get_representation(input_ids)
        z = self.proj_head(z)

        if self.config.use_l2_normalize:
            z = F.normalize(z, dim=-1)

        loss = supervised_contrastive_loss(z, labels, self.config.temperature)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
