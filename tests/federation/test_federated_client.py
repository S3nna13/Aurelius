"""Tests for federated_client — federated learning client."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.federation.federated_client import ClientConfig, FederatedClient, ModelDelta


class TestModelDelta:
    def test_delta_from_weights(self):
        old = {"weight": torch.tensor([1.0, 2.0, 3.0])}
        new = {"weight": torch.tensor([1.5, 2.5, 3.5])}
        delta = ModelDelta.compute(old, new)
        assert "weight" in delta
        assert torch.allclose(delta["weight"], torch.tensor([0.5, 0.5, 0.5]))

    def test_apply_delta(self):
        params = {"weight": torch.tensor([1.0, 2.0])}
        delta = {"weight": torch.tensor([0.1, 0.2])}
        result = ModelDelta.apply(params, delta)
        assert torch.allclose(result["weight"], torch.tensor([1.1, 2.2]))

    def test_empty_delta(self):
        assert ModelDelta.compute({}, {}) == {}


class TestClientConfig:
    def test_default_values(self):
        cfg = ClientConfig()
        assert cfg.learning_rate == 0.01
        assert cfg.local_epochs == 1
        assert cfg.batch_size == 32
        assert cfg.client_id is None


class TestFederatedClient:
    def test_train_returns_delta(self):
        model = nn.Linear(4, 2)
        data = torch.randn(16, 4)
        client = FederatedClient(client_id="test-1", model=model)
        delta = client.train(data, lr=0.01, epochs=1)
        assert isinstance(delta, dict)
        assert len(delta) > 0

    def test_client_id_persists(self):
        client = FederatedClient(client_id="node-42", model=nn.Linear(2, 2))
        assert client.client_id == "node-42"

    def test_train_stats(self):
        model = nn.Linear(8, 4)
        data = torch.randn(32, 8)
        client = FederatedClient(client_id="stats-test", model=model)
        delta = client.train(data, lr=0.1, epochs=2)
        stats = client.get_stats()
        assert stats["client_id"] == "stats-test"
        assert stats["epochs"] == 2
        assert stats["samples"] == 32
        assert isinstance(delta, dict)
