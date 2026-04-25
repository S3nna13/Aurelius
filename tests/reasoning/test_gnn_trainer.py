"""Tests for src/reasoning/gnn_trainer.py"""
from __future__ import annotations

import pytest
import torch

from src.reasoning.gnn_layer import GNNConfig
from src.reasoning.gnn_trainer import (
    GNNTrainConfig,
    GNNTrainer,
    GNNTrainResult,
)


@pytest.fixture
def tiny_config():
    return GNNConfig(in_dim=64, out_dim=64, n_layers=2, n_heads=4, use_residual=True)


class TestGNNTrainerNodeClassification:
    def test_training_step_decreases_loss(self, tiny_config):
        cfg = GNNTrainConfig(task="node_classification", epochs=3, lr=0.01, gnn_config=tiny_config)
        trainer = GNNTrainer(cfg)
        x = torch.randn(16, 64)
        edge = torch.tensor([[i, (i + 1) % 16] for i in range(16)], dtype=torch.long).t()
        labels = torch.randint(0, 4, (16,))
        losses = []
        for _ in range(5):
            loss = trainer.step(x, edge, labels)
            losses.append(loss)
        assert losses[-1] <= losses[0]

    def test_fit_completes_and_returns_result(self, tiny_config):
        cfg = GNNTrainConfig(task="node_classification", epochs=3, lr=0.01, gnn_config=tiny_config)
        trainer = GNNTrainer(cfg)
        x = torch.randn(16, 64)
        edge = torch.tensor([[i, (i + 1) % 16] for i in range(16)], dtype=torch.long).t()
        labels = torch.randint(0, 4, (16,))
        result = trainer.fit(x, edge, labels)
        assert isinstance(result, GNNTrainResult)
        assert result.task == "node_classification"
        assert result.n_epochs_run > 0
        assert result.best_loss >= 0.0


class TestGNNTrainerLinkPrediction:
    def test_link_prediction_step(self, tiny_config):
        cfg = GNNTrainConfig(task="link_prediction", epochs=2, lr=0.01, gnn_config=tiny_config)
        trainer = GNNTrainer(cfg)
        x = torch.randn(8, 64)
        edge = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        neg_edge = torch.tensor([[4, 5, 6, 7], [5, 6, 7, 4]], dtype=torch.long)
        loss = trainer.step(x, edge, neg_edge_index=neg_edge)
        assert isinstance(loss, float)


class TestGNNTrainerGraphClassification:
    def test_graph_classification_step(self, tiny_config):
        cfg = GNNTrainConfig(task="graph_classification", epochs=2, lr=0.01, gnn_config=tiny_config)
        trainer = GNNTrainer(cfg)
        x = torch.randn(4, 64)
        edge = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        label = torch.tensor(1)
        batch = [(x, edge, label)]
        loss = trainer.step(x, edge, batch_graphs=batch)
        assert isinstance(loss, float)


class TestGNNTrainerValueErrors:
    def test_missing_labels_raises(self, tiny_config):
        cfg = GNNTrainConfig(task="node_classification", epochs=1, gnn_config=tiny_config)
        trainer = GNNTrainer(cfg)
        x = torch.randn(8, 64)
        edge = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        with pytest.raises(ValueError):
            trainer.fit(x, edge, labels=None)

    def test_fit_without_labels_raises(self, tiny_config):
        cfg = GNNTrainConfig(task="node_classification", epochs=1, gnn_config=tiny_config)
        trainer = GNNTrainer(cfg)
        x = torch.randn(8, 64)
        edge = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        with pytest.raises(ValueError):
            trainer.fit(x, edge)


class TestGNNTrainConfigCustomization:
    def test_custom_epochs(self):
        cfg = GNNTrainConfig(epochs=20, lr=0.0001, weight_decay=0.0)
        assert cfg.epochs == 20
        assert cfg.lr == 0.0001
        assert cfg.weight_decay == 0.0

    def test_with_gnn_config(self):
        from src.reasoning.gnn_layer import GNNConfig
        gc = GNNConfig(gnn_type="gat", in_dim=32, out_dim=16, n_layers=3)
        cfg = GNNTrainConfig(gnn_config=gc)
        assert cfg.gnn_config is gc
        assert cfg.gnn_config.gnn_type == "gat"