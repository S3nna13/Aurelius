"""Training loop for GNN-based reasoning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from .gnn_layer import GNNConfig, GNNStack


@dataclass
class GNNTrainConfig:
    task: str = "node_classification"
    epochs: int = 10
    lr: float = 0.001
    weight_decay: float = 5e-4
    early_stopping_patience: int = 5
    val_split: float = 0.1
    batch_size: int = 32
    gnn_config: Any = None


@dataclass
class GNNTrainResult:
    task: str
    final_loss: float
    best_loss: float
    n_epochs_run: int
    converged: bool


class GNNTrainer:
    def __init__(self, config: GNNTrainConfig) -> None:
        self.config = config
        gnn_cfg = config.gnn_config or GNNConfig()
        self.model = GNNStack(gnn_cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def _train_node_classification(self, x, edge_index, labels, val_labels=None):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(x, edge_index)
        loss = self.loss_fn(out, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _step_link_prediction(self, x, edge_index, neg_edge_index):
        self.model.train()
        self.optimizer.zero_grad()
        emb = self.model(x, edge_index)
        if edge_index.dim() == 2 and edge_index.size(0) == 2:
            pos_src, pos_dst = edge_index[0], edge_index[1]
            neg_src, neg_dst = neg_edge_index[0], neg_edge_index[1]
        else:
            pos_src = torch.arange(len(edge_index) // 2, device=x.device)
            pos_dst = torch.arange(len(edge_index) // 2, device=x.device)
            neg_src = torch.arange(len(neg_edge_index) // 2, device=x.device)
            neg_dst = torch.arange(len(neg_edge_index) // 2, device=x.device)
        pos_scores = (emb[pos_src] * emb[pos_dst]).sum(-1)
        neg_scores = (emb[neg_src] * emb[neg_dst]).sum(-1)
        loss = torch.mean(
            -torch.log(torch.sigmoid(pos_scores) + 1e-8)
            - torch.log(1 - torch.sigmoid(neg_scores) + 1e-8)
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _step_graph_classification(self, batch_graphs):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0
        for graph_x, graph_edge_index, graph_label in batch_graphs:
            graph_x = graph_x.to(self.device)
            graph_edge_index = graph_edge_index.to(self.device)
            graph_label = graph_label.to(self.device)
            out = self.model(graph_x, graph_edge_index)
            pred = out.mean(dim=0)
            loss = self.loss_fn(pred.unsqueeze(0), graph_label.unsqueeze(0))
            total_loss += loss.item()
            loss.backward()
        self.optimizer.step()
        return total_loss / max(len(batch_graphs), 1)

    def fit(
        self,
        x: Any,
        edge_index: Any,
        labels: Any = None,
        neg_edge_index: Any = None,
        batch_graphs: Any = None,
        val_x: Any = None,
        val_edge_index: Any = None,
        val_labels: Any = None,
    ) -> GNNTrainResult:
        best_loss = float("inf")
        patience_counter = 0
        n_run = 0

        if labels is None and self.config.task == "node_classification":
            raise ValueError("labels required for node_classification task")

        for epoch in range(self.config.epochs):
            if self.config.task == "node_classification":
                loss = self._train_node_classification(x, edge_index, labels)
            elif self.config.task == "link_prediction":
                loss = self._step_link_prediction(x, edge_index, neg_edge_index)
            elif self.config.task == "graph_classification":
                loss = self._step_graph_classification(batch_graphs)
            else:
                raise ValueError(f"Unknown task: {self.config.task}")

            n_run += 1
            if loss < best_loss:
                best_loss = loss

            if val_x is not None and val_labels is not None:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(val_x, val_edge_index)
                    val_loss = self.loss_fn(val_out, val_labels).item()
                if val_loss > best_loss:
                    patience_counter += 1
                else:
                    patience_counter = 0
                if patience_counter >= self.config.early_stopping_patience:
                    break

        return GNNTrainResult(
            task=self.config.task,
            final_loss=loss,
            best_loss=best_loss,
            n_epochs_run=n_run,
            converged=patience_counter < self.config.early_stopping_patience,
        )

    def step(self, x, edge_index, labels=None, neg_edge_index=None, batch_graphs=None):
        if self.config.task == "node_classification":
            return self._train_node_classification(x, edge_index, labels)
        elif self.config.task == "link_prediction":
            return self._step_link_prediction(x, edge_index, neg_edge_index)
        elif self.config.task == "graph_classification":
            return self._step_graph_classification(batch_graphs)
        raise ValueError(f"Unknown task: {self.config.task}")