"""Tests for src/training/federated.py — 11 tests."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.federated import (
    FedConfig,
    FederatedClient,
    FederatedServer,
    fedavg_aggregate,
    fedprox_loss,
    simulate_data_heterogeneity,
)


# ---------------------------------------------------------------------------
# Fixtures
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

N_CLIENTS = 2
SEQ_LEN = 8
N_BATCHES = 2
LOCAL_EPOCHS = 1


def tiny_model() -> AureliusTransformer:
    return AureliusTransformer(TINY_CFG)


def tiny_fed_config(**kwargs) -> FedConfig:
    defaults = dict(
        n_clients=N_CLIENTS,
        local_epochs=LOCAL_EPOCHS,
        local_lr=1e-3,
    )
    defaults.update(kwargs)
    return FedConfig(**defaults)


def make_batch(vocab_size: int = 256, seq_len: int = SEQ_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    labels = torch.randint(0, vocab_size, (1, seq_len))
    return input_ids, labels


def make_data_loader(n_batches: int = N_BATCHES) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [make_batch() for _ in range(n_batches)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fed_config_defaults():
    """FedConfig has correct default values."""
    cfg = FedConfig()
    assert cfg.n_clients == 4
    assert cfg.local_epochs == 2
    assert cfg.local_lr == 1e-3
    assert cfg.mu == 0.01
    assert cfg.aggregation == "fedavg"
    assert cfg.clip_norm is None


def test_fedavg_aggregate_uniform():
    """Averaging two identical state dicts with uniform weights returns the same values."""
    model = tiny_model()
    sd = model.state_dict()
    sd_copy = copy.deepcopy(sd)

    result = fedavg_aggregate([sd, sd_copy])

    for key in sd:
        assert torch.allclose(result[key].float(), sd[key].float(), atol=1e-6), (
            f"Mismatch at key {key}"
        )


def test_fedavg_aggregate_weighted():
    """Weighted aggregation shifts result toward the higher-weight client."""
    model_a = tiny_model()
    model_b = tiny_model()

    # Reinitialize model_b with different weights
    with torch.no_grad():
        for p in model_b.parameters():
            p.fill_(1.0)

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    # Weight heavily toward model_b (weight=9 vs 1)
    result = fedavg_aggregate([sd_a, sd_b], weights=[1.0, 9.0])

    # Pick a parameter key that is a float tensor and not tied
    key = "embed.weight"
    # Result should be closer to sd_b (1.0) than sd_a
    dist_to_b = (result[key].float() - sd_b[key].float()).abs().mean().item()
    dist_to_a = (result[key].float() - sd_a[key].float()).abs().mean().item()
    assert dist_to_b < dist_to_a, (
        "Weighted average should be closer to the higher-weight client"
    )


def test_fedprox_loss_zero_same_model():
    """FedProx loss is zero when local and global models are identical."""
    model = tiny_model()
    global_model = copy.deepcopy(model)

    loss = fedprox_loss(model, global_model, mu=0.01)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_fedprox_loss_positive_different():
    """FedProx loss is positive when local and global models differ."""
    local_model = tiny_model()
    global_model = copy.deepcopy(local_model)

    # Perturb local model parameters
    with torch.no_grad():
        for p in local_model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    loss = fedprox_loss(local_model, global_model, mu=0.01)
    assert loss.item() > 0.0


def test_federated_client_local_train_keys():
    """local_train returns a dict with 'train_loss' and 'n_samples'."""
    model = tiny_model()
    cfg = tiny_fed_config()
    client = FederatedClient(client_id=0, model=model, config=cfg)

    data = make_data_loader()
    result = client.local_train(data)

    assert "train_loss" in result
    assert "n_samples" in result
    assert isinstance(result["train_loss"], float)
    assert isinstance(result["n_samples"], int)
    assert result["n_samples"] > 0


def test_federated_client_state_dict_sync():
    """set_state_dict / get_state_dict roundtrip preserves weights exactly."""
    model = tiny_model()
    cfg = tiny_fed_config()
    client = FederatedClient(client_id=0, model=model, config=cfg)

    # Create a reference model with different init
    ref_model = tiny_model()
    with torch.no_grad():
        for p in ref_model.parameters():
            p.fill_(0.42)

    ref_sd = ref_model.state_dict()
    client.set_state_dict(ref_sd)
    retrieved_sd = client.get_state_dict()

    for key in ref_sd:
        assert torch.allclose(
            retrieved_sd[key].float(), ref_sd[key].float(), atol=1e-6
        ), f"Mismatch at key {key}"


def test_federated_server_register():
    """Server correctly tracks the number of registered clients."""
    model = tiny_model()
    cfg = tiny_fed_config()
    server = FederatedServer(global_model=model, config=cfg)

    for i in range(N_CLIENTS):
        client = FederatedClient(client_id=i, model=tiny_model(), config=cfg)
        server.register_client(client)

    assert len(server.clients) == N_CLIENTS


def test_federated_server_broadcast():
    """After broadcast, all clients share the same weights as the global model."""
    global_model = tiny_model()
    # Set global model to a known constant
    with torch.no_grad():
        for p in global_model.parameters():
            p.fill_(0.7)

    cfg = tiny_fed_config()
    server = FederatedServer(global_model=global_model, config=cfg)

    for i in range(N_CLIENTS):
        client = FederatedClient(client_id=i, model=tiny_model(), config=cfg)
        server.register_client(client)

    server.broadcast()

    global_sd = global_model.state_dict()
    for client in server.clients:
        client_sd = client.get_state_dict()
        for key in global_sd:
            assert torch.allclose(
                client_sd[key].float(), global_sd[key].float(), atol=1e-6
            ), f"Client {client.client_id} mismatch at key {key}"


def test_federated_round_keys():
    """federated_round returns dict with 'round', 'mean_client_loss', 'n_clients'."""
    global_model = tiny_model()
    cfg = tiny_fed_config()
    server = FederatedServer(global_model=global_model, config=cfg)

    for i in range(N_CLIENTS):
        client = FederatedClient(client_id=i, model=tiny_model(), config=cfg)
        server.register_client(client)

    client_data = [make_data_loader() for _ in range(N_CLIENTS)]
    result = server.federated_round(client_data)

    assert "round" in result
    assert "mean_client_loss" in result
    assert "n_clients" in result
    assert result["round"] == 1
    assert result["n_clients"] == N_CLIENTS
    assert isinstance(result["mean_client_loss"], float)


def test_simulate_data_heterogeneity_shape():
    """simulate_data_heterogeneity returns correct number of clients and batch shapes."""
    data = simulate_data_heterogeneity(
        n_clients=N_CLIENTS,
        vocab_size=256,
        seq_len=SEQ_LEN,
        n_batches=N_BATCHES,
        seed=42,
    )

    assert len(data) == N_CLIENTS

    for client_batches in data:
        assert len(client_batches) == N_BATCHES
        for input_ids, labels in client_batches:
            assert input_ids.shape == (1, SEQ_LEN), f"Expected (1, {SEQ_LEN}), got {input_ids.shape}"
            assert labels.shape == (1, SEQ_LEN), f"Expected (1, {SEQ_LEN}), got {labels.shape}"
