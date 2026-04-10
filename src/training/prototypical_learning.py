"""Prototype-based few-shot learning and nearest-neighbor classification.

Implements prototypical networks (Snell et al., 2017) over Aurelius model
embeddings. Unlike MAML, there is no gradient-based inner-loop adaptation —
classification is done purely by nearest-prototype lookup in embedding space.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProtoConfig:
    """Configuration for prototypical-network few-shot learning."""

    n_way: int = 5            # Number of classes per episode
    n_support: int = 5        # Support examples per class
    n_query: int = 10         # Query examples per class
    embedding_dim: int = 64   # Dimension of sentence embeddings (= d_model)
    distance: str = "euclidean"  # "euclidean" | "cosine"


# ---------------------------------------------------------------------------
# Core functional utilities
# ---------------------------------------------------------------------------

def extract_embeddings(model: nn.Module, input_ids: Tensor) -> Tensor:
    """Extract L2-normalised sentence embeddings from the last transformer layer.

    Registers a forward hook on ``model.layers[-1]`` to capture its output
    hidden states, then mean-pools over the sequence dimension.

    Args:
        model: AureliusTransformer (or any model with ``.layers`` ModuleList
               whose final layer returns ``(hidden, kv)``).
        input_ids: (B, T) token indices.

    Returns:
        (B, d_model) L2-normalised float tensor.
    """
    captured: list[Tensor] = []

    def _hook(module, inputs, output):
        # TransformerBlock returns (hidden_states, kv_cache_tuple)
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(hidden.detach())

    handle = model.layers[-1].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    # captured[0]: (B, T, d_model)
    hidden_states = captured[0]
    # Mean-pool over the sequence dimension → (B, d_model)
    embeddings = hidden_states.mean(dim=1)
    # L2-normalise
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings


def compute_prototypes(embeddings: Tensor, labels: Tensor, n_classes: int) -> Tensor:
    """Compute per-class prototype as the mean of its support embeddings.

    Args:
        embeddings: (N, D) support embeddings.
        labels: (N,) integer class indices in ``[0, n_classes)``.
        n_classes: Total number of classes.

    Returns:
        (n_classes, D) prototype tensor.
    """
    D = embeddings.size(-1)
    prototypes = torch.zeros(n_classes, D, dtype=embeddings.dtype, device=embeddings.device)
    counts = torch.zeros(n_classes, dtype=torch.long, device=embeddings.device)

    for cls in range(n_classes):
        mask = labels == cls
        if mask.any():
            prototypes[cls] = embeddings[mask].mean(dim=0)
            counts[cls] = mask.sum()

    return prototypes


def prototypical_loss(
    query_embs: Tensor,
    prototypes: Tensor,
    query_labels: Tensor,
    distance: str = "euclidean",
) -> tuple[Tensor, float]:
    """Compute prototypical-network loss and accuracy.

    Scores each query against every prototype via negative distance, then
    takes the cross-entropy of those scores against the true labels.

    Args:
        query_embs: (Q, D) query embeddings.
        prototypes: (C, D) class prototypes.
        query_labels: (Q,) integer class labels.
        distance: ``"euclidean"`` or ``"cosine"``.

    Returns:
        Tuple of (loss_scalar, accuracy_float).
    """
    if distance == "euclidean":
        # Squared Euclidean: ||q - p||^2 = ||q||^2 + ||p||^2 - 2 q·p
        # query_embs: (Q, D), prototypes: (C, D)
        dists = torch.cdist(query_embs, prototypes, p=2)  # (Q, C)
        logits = -dists
    elif distance == "cosine":
        # Cosine similarity in [-1, 1]; we use it directly as a score
        q_norm = F.normalize(query_embs, p=2, dim=-1)   # (Q, D)
        p_norm = F.normalize(prototypes, p=2, dim=-1)   # (C, D)
        logits = q_norm @ p_norm.t()                    # (Q, C)
    else:
        raise ValueError(f"Unsupported distance metric: {distance!r}")

    loss = F.cross_entropy(logits, query_labels)

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        accuracy = (preds == query_labels).float().mean().item()

    return loss, accuracy


# ---------------------------------------------------------------------------
# PrototypicalClassifier
# ---------------------------------------------------------------------------

class PrototypicalClassifier:
    """Few-shot classifier using stored class prototypes.

    Usage::

        clf = PrototypicalClassifier(model, config)
        clf.fit(support_ids_list, support_labels_list)
        preds = clf.predict(query_ids)
    """

    def __init__(self, model: nn.Module, config: ProtoConfig) -> None:
        self.model = model
        self.config = config
        self.prototypes: Optional[Tensor] = None

    def fit(self, support_ids: list[Tensor], support_labels: list[int]) -> None:
        """Extract support embeddings and compute per-class prototypes.

        Args:
            support_ids: List of (1, T) or (T,) tensors, one per support example.
            support_labels: Integer class label for each support example.
        """
        emb_list: list[Tensor] = []
        with torch.no_grad():
            for ids in support_ids:
                # Ensure (1, T) shape
                if ids.ndim == 1:
                    ids = ids.unsqueeze(0)
                emb = extract_embeddings(self.model, ids)  # (1, D)
                emb_list.append(emb)

        embeddings = torch.cat(emb_list, dim=0)  # (N, D)
        labels = torch.tensor(support_labels, dtype=torch.long, device=embeddings.device)
        n_classes = self.config.n_way
        self.prototypes = compute_prototypes(embeddings, labels, n_classes)

    def predict(self, query_ids: Tensor) -> Tensor:
        """Return predicted class indices for each query.

        Args:
            query_ids: (B, T) query token ids.

        Returns:
            (B,) long tensor of predicted class indices.
        """
        proba = self.predict_proba(query_ids)
        return proba.argmax(dim=-1)

    def predict_proba(self, query_ids: Tensor) -> Tensor:
        """Return softmax probability distribution over classes.

        Args:
            query_ids: (B, T) query token ids.

        Returns:
            (B, n_classes) float tensor summing to 1 per row.
        """
        assert self.prototypes is not None, "Call fit() before predict_proba()"
        with torch.no_grad():
            query_embs = extract_embeddings(self.model, query_ids)  # (B, D)

        distance = self.config.distance
        if distance == "euclidean":
            dists = torch.cdist(query_embs, self.prototypes, p=2)  # (B, C)
            logits = -dists
        elif distance == "cosine":
            q_norm = F.normalize(query_embs, p=2, dim=-1)
            p_norm = F.normalize(self.prototypes, p=2, dim=-1)
            logits = q_norm @ p_norm.t()
        else:
            raise ValueError(f"Unsupported distance metric: {distance!r}")

        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# ProtoNetTrainer
# ---------------------------------------------------------------------------

class ProtoNetTrainer:
    """Trains the embedding model with episodic prototypical-network loss.

    Each call to ``train_episode`` constitutes one N-way K-shot episode:
    support set → prototypes → compute loss on query set → backprop.

    Args:
        model: Language model (AureliusTransformer or compatible).
        config: ProtoConfig.
        optimizer: Any PyTorch optimizer bound to ``model.parameters()``.
    """

    def __init__(self, model: nn.Module, config: ProtoConfig, optimizer) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer

    def train_episode(
        self,
        support_ids: list[Tensor],
        support_labels: list[int],
        query_ids: list[Tensor],
        query_labels: list[int],
    ) -> dict:
        """Run one prototypical-network episode and update model weights.

        Args:
            support_ids: List of (1, T) tensors for support set.
            support_labels: Integer labels for support examples.
            query_ids: List of (1, T) tensors for query set.
            query_labels: Integer labels for query examples.

        Returns:
            Dict with keys ``"loss"`` (float) and ``"accuracy"`` (float).
        """
        self.model.train()

        # --- Extract support embeddings (no grad needed for support set) ---
        support_emb_list: list[Tensor] = []
        with torch.no_grad():
            for ids in support_ids:
                if ids.ndim == 1:
                    ids = ids.unsqueeze(0)
                emb = extract_embeddings(self.model, ids)  # (1, D)
                support_emb_list.append(emb)

        support_embs = torch.cat(support_emb_list, dim=0)  # (N_support, D)
        s_labels = torch.tensor(
            support_labels, dtype=torch.long, device=support_embs.device
        )
        prototypes = compute_prototypes(support_embs, s_labels, self.config.n_way)

        # --- Extract query embeddings WITH grad for backprop ---
        # We need gradients through the query embeddings.
        # Re-run the model in train mode capturing the last-layer hidden states.
        query_emb_list: list[Tensor] = []
        for ids in query_ids:
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            emb = self._extract_embeddings_with_grad(ids)  # (1, D)
            query_emb_list.append(emb)

        query_embs = torch.cat(query_emb_list, dim=0)  # (N_query, D)
        q_labels = torch.tensor(
            query_labels, dtype=torch.long, device=query_embs.device
        )

        # --- Prototypical loss ---
        loss, accuracy = prototypical_loss(
            query_embs, prototypes, q_labels, distance=self.config.distance
        )

        # --- Optimizer step ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "accuracy": accuracy}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_embeddings_with_grad(self, input_ids: Tensor) -> Tensor:
        """Like extract_embeddings but keeps the computation graph for backprop."""
        captured: list[Tensor] = []

        def _hook(module, inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured.append(hidden)

        handle = self.model.layers[-1].register_forward_hook(_hook)
        try:
            self.model(input_ids)
        finally:
            handle.remove()

        hidden_states = captured[0]  # (B, T, D)
        embeddings = hidden_states.mean(dim=1)  # (B, D)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
