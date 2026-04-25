"""Tests for GNN trainer."""
from __future__ import annotations

import torch

from src.reasoning.gnn_layer import GNNConfig
from src.reasoning.gnn_trainer import GNNTrainConfig, GNNTrainer, GNNTrainResult


def test_node_classification_step():
    cfg = GNNTrainConfig(
        task="node_classification",
        epochs=3,
        lr=0.01,
        gnn_config=GNNConfig(in_dim=8, out_dim=8, n_layers=1),
    )
    trainer = GNNTrainer(cfg)
    x = torch.randn(8, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    loss0 = trainer.step(x, edge_index, labels)
    loss1 = trainer.step(x, edge_index, labels)
    if not loss1 < loss0:
        raise ValueError(f"Loss should decrease: {loss0} -> {loss1}")


def test_link_prediction_step():
    cfg = GNNTrainConfig(
        task="link_prediction",
        epochs=2,
        lr=0.01,
        gnn_config=GNNConfig(in_dim=8, out_dim=8, n_layers=1),
    )
    trainer = GNNTrainer(cfg)
    x = torch.randn(8, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    neg_edge_index = torch.tensor([[0, 1, 2, 3], [3, 0, 1, 2]], dtype=torch.long)
    loss = trainer.step(x, edge_index, neg_edge_index=neg_edge_index)
    if not isinstance(loss, float):
        raise ValueError(f"Expected float loss, got {type(loss)}")


def test_graph_classification_step():
    cfg = GNNTrainConfig(
        task="graph_classification",
        epochs=2,
        lr=0.01,
        gnn_config=GNNConfig(in_dim=8, out_dim=8, n_layers=1),
    )
    trainer = GNNTrainer(cfg)
    batch = [
        (torch.randn(4, 8), torch.tensor([[0, 1], [1, 2]], dtype=torch.long), torch.tensor(0)),
        (torch.randn(4, 8), torch.tensor([[0, 1], [1, 2]], dtype=torch.long), torch.tensor(1)),
    ]
    loss = trainer.step(x=None, edge_index=None, batch_graphs=batch)
    if not isinstance(loss, float):
        raise ValueError(f"Expected float loss, got {type(loss)}")


def test_fit_labels_missing():
    cfg = GNNTrainConfig(
        task="node_classification",
        epochs=1,
        gnn_config=GNNConfig(in_dim=8, out_dim=8, n_layers=1),
    )
    trainer = GNNTrainer(cfg)
    x = torch.randn(8, 8)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    caught = False
    try:
        trainer.fit(x, edge_index, labels=None)
    except ValueError as e:
        if "labels required" in str(e):
            caught = True
    if not caught:
        raise ValueError("Expected ValueError for missing labels")


def test_fit_returns_gnntrainresult():
    cfg = GNNTrainConfig(
        task="node_classification",
        epochs=2,
        lr=0.01,
        gnn_config=GNNConfig(in_dim=8, out_dim=8, n_layers=1),
    )
    trainer = GNNTrainer(cfg)
    x = torch.randn(8, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    result = trainer.fit(x, edge_index, labels=labels)
    if not isinstance(result, GNNTrainResult):
        raise ValueError(f"Expected GNNTrainResult, got {type(result)}")
    if not result.n_epochs_run == 2:
        raise ValueError(f"Expected 2 epochs, got {result.n_epochs_run}")