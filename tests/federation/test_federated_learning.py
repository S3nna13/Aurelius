"""Tests for src/federation/federated_learning.py."""

from __future__ import annotations

import pytest

from src.federation.federated_learning import (
    ClientUpdate,
    FederatedClient,
    FederatedServer,
    FEDERATION_REGISTRY,
)


# ---------------------------------------------------------------------------
# ClientUpdate dataclass
# ---------------------------------------------------------------------------

class TestClientUpdate:
    def test_fields_required(self):
        cu = ClientUpdate(
            client_id="c1", round_id=1, weight_delta=[0.1, -0.2], n_samples=50
        )
        assert cu.client_id == "c1"
        assert cu.round_id == 1
        assert cu.weight_delta == [0.1, -0.2]
        assert cu.n_samples == 50

    def test_loss_default(self):
        cu = ClientUpdate(client_id="c1", round_id=0, weight_delta=[], n_samples=10)
        assert cu.loss == 0.0

    def test_loss_explicit(self):
        cu = ClientUpdate(
            client_id="c2", round_id=3, weight_delta=[1.0], n_samples=20, loss=0.5
        )
        assert cu.loss == 0.5

    def test_client_id_type(self):
        cu = ClientUpdate(client_id="abc", round_id=0, weight_delta=[0.0], n_samples=1)
        assert isinstance(cu.client_id, str)

    def test_round_id_type(self):
        cu = ClientUpdate(client_id="x", round_id=7, weight_delta=[], n_samples=1)
        assert isinstance(cu.round_id, int)

    def test_n_samples_stored(self):
        cu = ClientUpdate(client_id="x", round_id=0, weight_delta=[], n_samples=999)
        assert cu.n_samples == 999

    def test_weight_delta_stored(self):
        delta = [1.0, 2.0, 3.0]
        cu = ClientUpdate(client_id="x", round_id=0, weight_delta=delta, n_samples=1)
        assert cu.weight_delta == delta


# ---------------------------------------------------------------------------
# FederatedClient
# ---------------------------------------------------------------------------

class TestFederatedClient:
    def test_init_default_n_samples(self):
        client = FederatedClient("c1")
        assert client.n_samples == 100

    def test_init_custom_n_samples(self):
        client = FederatedClient("c2", n_samples=500)
        assert client.n_samples == 500

    def test_client_id_stored(self):
        client = FederatedClient("my_client")
        assert client.client_id == "my_client"

    def test_compute_update_delta_correct(self):
        client = FederatedClient("c1", n_samples=10)
        global_w = [1.0, 2.0, 3.0]
        local_w = [1.5, 1.8, 3.3]
        update = client.compute_update(global_w, local_w, round_id=0)
        expected = [0.5, -0.2, 0.3]
        assert len(update.weight_delta) == 3
        for got, exp in zip(update.weight_delta, expected):
            assert abs(got - exp) < 1e-9

    def test_compute_update_round_id_stored(self):
        client = FederatedClient("c1")
        update = client.compute_update([0.0], [1.0], round_id=5)
        assert update.round_id == 5

    def test_compute_update_client_id_stored(self):
        client = FederatedClient("alice")
        update = client.compute_update([0.0], [1.0], round_id=0)
        assert update.client_id == "alice"

    def test_compute_update_n_samples_stored(self):
        client = FederatedClient("c1", n_samples=42)
        update = client.compute_update([0.0, 0.0], [1.0, 2.0], round_id=1)
        assert update.n_samples == 42

    def test_compute_update_returns_client_update(self):
        client = FederatedClient("c1")
        update = client.compute_update([1.0], [2.0], round_id=0)
        assert isinstance(update, ClientUpdate)

    def test_compute_update_zero_delta(self):
        client = FederatedClient("c1")
        w = [1.0, 2.0]
        update = client.compute_update(w, w, round_id=0)
        assert update.weight_delta == [0.0, 0.0]

    def test_apply_update_default_lr(self):
        client = FederatedClient("c1")
        weights = [1.0, 2.0, 3.0]
        delta = [0.1, -0.2, 0.5]
        result = client.apply_update(weights, delta)
        expected = [1.1, 1.8, 3.5]
        for got, exp in zip(result, expected):
            assert abs(got - exp) < 1e-9

    def test_apply_update_custom_lr(self):
        client = FederatedClient("c1")
        weights = [0.0, 0.0]
        delta = [1.0, 2.0]
        result = client.apply_update(weights, delta, lr=0.5)
        assert abs(result[0] - 0.5) < 1e-9
        assert abs(result[1] - 1.0) < 1e-9

    def test_apply_update_zero_lr(self):
        client = FederatedClient("c1")
        weights = [3.0, 4.0]
        delta = [10.0, 10.0]
        result = client.apply_update(weights, delta, lr=0.0)
        assert result == [3.0, 4.0]

    def test_apply_update_returns_list(self):
        client = FederatedClient("c1")
        result = client.apply_update([1.0], [0.5])
        assert isinstance(result, list)

    def test_apply_update_length_preserved(self):
        client = FederatedClient("c1")
        weights = [1.0, 2.0, 3.0, 4.0]
        delta = [0.1] * 4
        result = client.apply_update(weights, delta)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# FederatedServer
# ---------------------------------------------------------------------------

class TestFederatedServer:
    def test_initial_round_is_zero(self):
        server = FederatedServer()
        assert server.global_round == 0

    def test_global_round_property(self):
        server = FederatedServer()
        assert hasattr(server, "global_round")

    def test_advance_round_returns_new_value(self):
        server = FederatedServer()
        result = server.advance_round()
        assert result == 1

    def test_advance_round_increments(self):
        server = FederatedServer()
        server.advance_round()
        server.advance_round()
        assert server.global_round == 2

    def test_advance_round_multiple(self):
        server = FederatedServer()
        for i in range(5):
            r = server.advance_round()
            assert r == i + 1

    def test_aggregate_equal_n_samples_is_mean(self):
        server = FederatedServer()
        u1 = ClientUpdate("c1", 0, [1.0, 2.0], n_samples=10)
        u2 = ClientUpdate("c2", 0, [3.0, 4.0], n_samples=10)
        result = server.aggregate([u1, u2])
        assert abs(result[0] - 2.0) < 1e-9
        assert abs(result[1] - 3.0) < 1e-9

    def test_aggregate_weighted_fedavg(self):
        server = FederatedServer()
        u1 = ClientUpdate("c1", 0, [0.0], n_samples=1)
        u2 = ClientUpdate("c2", 0, [10.0], n_samples=9)
        result = server.aggregate([u1, u2])
        # expected = (0*1 + 10*9) / 10 = 9.0
        assert abs(result[0] - 9.0) < 1e-9

    def test_aggregate_larger_n_samples_more_weight(self):
        server = FederatedServer()
        u1 = ClientUpdate("c1", 0, [0.0], n_samples=1)
        u2 = ClientUpdate("c2", 0, [100.0], n_samples=99)
        result = server.aggregate([u1, u2])
        # Should be much closer to 100 than to 0
        assert result[0] > 90.0

    def test_aggregate_single_update(self):
        server = FederatedServer()
        u = ClientUpdate("c1", 0, [5.0, 7.0], n_samples=50)
        result = server.aggregate([u])
        assert result == [5.0, 7.0]

    def test_aggregate_three_clients(self):
        server = FederatedServer()
        updates = [
            ClientUpdate("c1", 0, [3.0, 6.0], n_samples=10),
            ClientUpdate("c2", 0, [3.0, 6.0], n_samples=10),
            ClientUpdate("c3", 0, [3.0, 6.0], n_samples=10),
        ]
        result = server.aggregate(updates)
        assert abs(result[0] - 3.0) < 1e-9
        assert abs(result[1] - 6.0) < 1e-9

    def test_aggregate_returns_list(self):
        server = FederatedServer()
        u = ClientUpdate("c1", 0, [1.0], n_samples=1)
        result = server.aggregate([u])
        assert isinstance(result, list)

    def test_aggregate_empty_returns_empty(self):
        server = FederatedServer()
        result = server.aggregate([])
        assert result == []

    def test_client_participation_rate_full(self):
        server = FederatedServer()
        updates = [ClientUpdate(f"c{i}", 0, [1.0], n_samples=10) for i in range(5)]
        rate = server.client_participation_rate(5, updates)
        assert abs(rate - 1.0) < 1e-9

    def test_client_participation_rate_partial(self):
        server = FederatedServer()
        updates = [ClientUpdate("c1", 0, [1.0], n_samples=10)]
        rate = server.client_participation_rate(4, updates)
        assert abs(rate - 0.25) < 1e-9

    def test_client_participation_rate_zero(self):
        server = FederatedServer()
        rate = server.client_participation_rate(10, [])
        assert rate == 0.0

    def test_client_participation_rate_returns_float(self):
        server = FederatedServer()
        updates = [ClientUpdate("c1", 0, [1.0], n_samples=1)]
        rate = server.client_participation_rate(2, updates)
        assert isinstance(rate, float)


# ---------------------------------------------------------------------------
# FEDERATION_REGISTRY
# ---------------------------------------------------------------------------

class TestFederationRegistry:
    def test_registry_exists(self):
        assert FEDERATION_REGISTRY is not None

    def test_registry_has_server(self):
        assert "server" in FEDERATION_REGISTRY

    def test_registry_server_is_federated_server(self):
        assert isinstance(FEDERATION_REGISTRY["server"], FederatedServer)

    def test_registry_has_client_factory(self):
        assert "client_factory" in FEDERATION_REGISTRY

    def test_registry_client_factory_is_class(self):
        assert FEDERATION_REGISTRY["client_factory"] is FederatedClient

    def test_registry_client_factory_callable(self):
        factory = FEDERATION_REGISTRY["client_factory"]
        client = factory("test_client")
        assert isinstance(client, FederatedClient)
