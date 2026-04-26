"""GNN trainer for knowledge graph reasoning tasks.

Supports node classification, link prediction, and graph classification.
License: MIT.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .gnn_layer import GNNStack


@dataclass
class GNNTrainConfig:
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    task: str = "node_classification"


@dataclass
class GNNTrainResult:
    final_loss: float
    epochs_trained: int
    task: str


class GNNTrainer:
    """Train GNN models for graph-based reasoning tasks."""

    def __init__(self, model: GNNStack, config: GNNTrainConfig | None = None) -> None:
        self.model = model
        self.config = config or GNNTrainConfig()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def node_classification_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]
        return F.cross_entropy(logits, labels)

    def link_prediction_loss(
        self, embeddings: torch.Tensor, pos_edges: torch.Tensor, neg_edges: torch.Tensor
    ) -> torch.Tensor:
        pos_scores = (embeddings[pos_edges[:, 0]] * embeddings[pos_edges[:, 1]]).sum(dim=-1)
        neg_scores = (embeddings[neg_edges[:, 0]] * embeddings[neg_edges[:, 1]]).sum(dim=-1)
        pos_labels = torch.ones(pos_scores.size(0), device=embeddings.device)
        neg_labels = torch.zeros(neg_scores.size(0), device=embeddings.device)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_labels, neg_labels])
        return F.binary_cross_entropy_with_logits(scores, labels)

    def train_step(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        embeddings = self.model(x, adj)

        if self.config.task == "node_classification":
            if labels is None:
                raise ValueError("labels required for node_classification")
            loss = self.node_classification_loss(embeddings, labels, mask)
        elif self.config.task == "link_prediction":
            if labels is None:
                raise ValueError("labels (edge tensor pairs) required for link_prediction")
            n = embeddings.size(0)
            torch.arange(n, device=x.device)
            pos_edges = labels
            neg_src = torch.randint(0, n, (pos_edges.size(0),), device=x.device)
            neg_dst = torch.randint(0, n, (pos_edges.size(0),), device=x.device)
            neg_edges = torch.stack([neg_src, neg_dst], dim=1)
            loss = self.link_prediction_loss(embeddings, pos_edges, neg_edges)
        elif self.config.task == "graph_classification":
            pooled = embeddings.mean(dim=0, keepdim=True)
            if labels is None:
                raise ValueError("labels required for graph_classification")
            loss = F.cross_entropy(pooled, labels[:1])
        else:
            raise ValueError(f"unknown task: {self.config.task!r}")

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        n_epochs: int | None = None,
    ) -> GNNTrainResult:
        epochs = n_epochs if n_epochs is not None else self.config.epochs
        last_loss = 0.0
        for _ in range(epochs):
            last_loss = self.train_step(x, adj, labels, mask)
        return GNNTrainResult(
            final_loss=last_loss,
            epochs_trained=epochs,
            task=self.config.task,
        )
