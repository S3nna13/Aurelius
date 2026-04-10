"""Tests for src/training/federated_learning.py."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.federated_learning import (
    FedConfig,
    ClientUpdate,
    fedavg_aggregate,
    fedprox_loss,
    simulate_client_update,
    FederatedServer,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


def tiny_model() -> AureliusTransformer:
    return AureliusTransformer(TINY_CFG)


def tiny_config(**kwargs) -> FedConfig:
    defaults = dict(
        n_clients=10,
        fraction_fit=0.3,
        local_epochs=1,
        local_lr=1e-3,
        aggregation="fedavg",
        mu=0.01,
    )
    defaults.update(kwargs)
    return FedConfig(**defaults)


def random_batch(batch: int = 2, seq: int = 8) -> torch.Tensor:
    return torch.randint(0, 256, (batch, seq))


def make_client_datasets(n: int, batches: int = 2) -> list[list[torch.Tensor]]:
    return [[random_batch() for _ in range(batches)] for _ in range(n)]


def make_delta(model: AureliusTransformer, fill: float = 0.1) -> dict[str, torch.Tensor]:
    return {k: torch.full_like(v, fill) for k, v in model.named_parameters()}


# ---------------------------------------------------------------------------
# Test 1: FedConfig defaults
# ---------------------------------------------------------------------------

def test_fedconfig_defaults():
    """FedConfig has the correct default values."""
    cfg = FedConfig()
    assert cfg.n_clients == 10
    assert cfg.fraction_fit == 0.3
    assert cfg.local_epochs == 2
    assert cfg.local_lr == 1e-3
    assert cfg.aggregation == "fedavg"
    assert cfg.mu == 0.01


# ---------------------------------------------------------------------------
# Test 2: ClientUpdate fields
# ---------------------------------------------------------------------------

def test_client_update_fields():
    """ClientUpdate holds client_id, delta_weights, n_samples, and loss."""
    delta = {"w": torch.zeros(3)}
    cu = ClientUpdate(client_id=5, delta_weights=delta, n_samples=100, loss=0.42)
    assert cu.client_id == 5
    assert cu.delta_weights is delta
    assert cu.n_samples == 100
    assert cu.loss == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Test 3: fedavg_aggregate equal samples → equal weights
# ---------------------------------------------------------------------------

def test_fedavg_aggregate_equal_samples_equal_weights():
    """With equal n_samples, fedavg_aggregate produces the arithmetic mean of deltas."""
    delta_a = {"w": torch.tensor([0.0, 0.0])}
    delta_b = {"w": torch.tensor([2.0, 2.0])}
    updates = [
        ClientUpdate(client_id=0, delta_weights=delta_a, n_samples=10, loss=0.0),
        ClientUpdate(client_id=1, delta_weights=delta_b, n_samples=10, loss=0.0),
    ]
    result = fedavg_aggregate(updates)
    expected = torch.tensor([1.0, 1.0])
    assert torch.allclose(result["w"].float(), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 4: fedavg_aggregate more samples → higher weight
# ---------------------------------------------------------------------------

def test_fedavg_aggregate_more_samples_higher_weight():
    """Client with more samples should contribute more to the aggregated delta."""
    delta_a = {"w": torch.tensor([0.0])}
    delta_b = {"w": torch.tensor([10.0])}
    updates = [
        ClientUpdate(client_id=0, delta_weights=delta_a, n_samples=1, loss=0.0),
        ClientUpdate(client_id=1, delta_weights=delta_b, n_samples=9, loss=0.0),
    ]
    result = fedavg_aggregate(updates)
    # Weighted average: 1/10 * 0 + 9/10 * 10 = 9.0
    assert torch.allclose(result["w"].float(), torch.tensor([9.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 5: fedavg_aggregate returns dict with same keys as input
# ---------------------------------------------------------------------------

def test_fedavg_aggregate_same_keys():
    """fedavg_aggregate output has the same keys as the input deltas."""
    keys = ["layer.weight", "layer.bias"]
    updates = [
        ClientUpdate(
            client_id=i,
            delta_weights={k: torch.zeros(3) for k in keys},
            n_samples=5,
            loss=0.0,
        )
        for i in range(3)
    ]
    result = fedavg_aggregate(updates)
    assert set(result.keys()) == set(keys)


# ---------------------------------------------------------------------------
# Test 6: fedprox_loss returns scalar
# ---------------------------------------------------------------------------

def test_fedprox_loss_returns_scalar():
    """fedprox_loss returns a scalar tensor."""
    params = {"w": torch.tensor([1.0, 2.0])}
    global_p = {"w": torch.tensor([0.0, 0.0])}
    loss = fedprox_loss(params, global_p, mu=0.01)
    assert loss.ndim == 0, "fedprox_loss should return a scalar (0-dim) tensor"


# ---------------------------------------------------------------------------
# Test 7: fedprox_loss zero when params equal
# ---------------------------------------------------------------------------

def test_fedprox_loss_zero_when_equal():
    """fedprox_loss is zero when local and global params are identical."""
    params = {"w": torch.tensor([1.5, -2.3, 0.7])}
    loss = fedprox_loss(params, params, mu=0.1)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Test 8: fedprox_loss positive when params differ
# ---------------------------------------------------------------------------

def test_fedprox_loss_positive_when_differ():
    """fedprox_loss is strictly positive when local and global params differ."""
    local_p = {"w": torch.tensor([1.0, 2.0])}
    global_p = {"w": torch.tensor([0.0, 0.0])}
    loss = fedprox_loss(local_p, global_p, mu=0.01)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# Test 9: simulate_client_update returns ClientUpdate
# ---------------------------------------------------------------------------

def test_simulate_client_update_returns_client_update():
    """simulate_client_update returns a ClientUpdate instance."""
    model = tiny_model()
    cfg = tiny_config(local_epochs=1)
    data = [random_batch()]
    result = simulate_client_update(model, data, cfg, client_id=3)
    assert isinstance(result, ClientUpdate)
    assert result.client_id == 3


# ---------------------------------------------------------------------------
# Test 10: simulate_client_update delta has same keys as model params
# ---------------------------------------------------------------------------

def test_simulate_client_update_delta_keys_match_model():
    """delta_weights in ClientUpdate has the same keys as the model's named parameters."""
    model = tiny_model()
    cfg = tiny_config(local_epochs=1)
    data = [random_batch()]
    result = simulate_client_update(model, data, cfg)
    model_keys = set(k for k, _ in model.named_parameters())
    assert set(result.delta_weights.keys()) == model_keys


# ---------------------------------------------------------------------------
# Test 11: FederatedServer.select_clients returns list of ints
# ---------------------------------------------------------------------------

def test_federated_server_select_clients_returns_list_of_ints():
    """select_clients returns a list of integer client IDs."""
    model = tiny_model()
    cfg = tiny_config(fraction_fit=0.5)
    server = FederatedServer(model, cfg)
    selected = server.select_clients(10)
    assert isinstance(selected, list)
    assert all(isinstance(i, int) for i in selected)


# ---------------------------------------------------------------------------
# Test 12: FederatedServer.select_clients length = fraction * n_clients
# ---------------------------------------------------------------------------

def test_federated_server_select_clients_correct_length():
    """select_clients returns max(1, int(fraction_fit * n_clients)) client IDs."""
    model = tiny_model()
    cfg = tiny_config(fraction_fit=0.3)
    server = FederatedServer(model, cfg)
    selected = server.select_clients(10)
    expected = max(1, int(0.3 * 10))
    assert len(selected) == expected


# ---------------------------------------------------------------------------
# Test 13: FederatedServer.aggregate_and_update returns required keys
# ---------------------------------------------------------------------------

def test_federated_server_aggregate_and_update_returns_required_keys():
    """aggregate_and_update returns dict with 'round', 'n_clients', 'mean_loss', 'weight_norm'."""
    model = tiny_model()
    cfg = tiny_config()
    server = FederatedServer(model, cfg)

    delta = make_delta(model, fill=0.0)
    updates = [
        ClientUpdate(client_id=0, delta_weights=delta, n_samples=10, loss=0.5),
    ]
    result = server.aggregate_and_update(updates)
    for key in ("round", "n_clients", "mean_loss", "weight_norm"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Test 14: FederatedServer.federated_round returns dict with "mean_loss"
# ---------------------------------------------------------------------------

def test_federated_round_returns_mean_loss():
    """federated_round returns a dict containing 'mean_loss'."""
    model = tiny_model()
    cfg = tiny_config(n_clients=4, fraction_fit=0.5, local_epochs=1)
    server = FederatedServer(model, cfg)
    client_datasets = make_client_datasets(4, batches=1)
    result = server.federated_round(client_datasets)
    assert "mean_loss" in result


# ---------------------------------------------------------------------------
# Test 15: fedavg_aggregate with single update returns same delta
# ---------------------------------------------------------------------------

def test_fedavg_aggregate_single_update_returns_same_delta():
    """With one update, fedavg_aggregate returns the same delta (weight = 1.0)."""
    delta = {"w": torch.tensor([3.0, -1.5, 0.0])}
    updates = [ClientUpdate(client_id=0, delta_weights=delta, n_samples=50, loss=0.1)]
    result = fedavg_aggregate(updates)
    assert torch.allclose(result["w"].float(), delta["w"].float(), atol=1e-6)
