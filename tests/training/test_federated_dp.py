"""
test_federated_dp.py — 16+ tests for federated_dp.py

Uses tiny inline models: d_model=8, vocab_size=8, n_clients=3, B=2, T=4.
Pure PyTorch only.
"""

import copy
import math

import pytest
import torch
import torch.nn as nn

from src.training.federated_dp import (
    DifferentiallyPrivateAggregator,
    FedAvgServer,
    FedDPConfig,
    FederatedClient,
    FederatedDPTrainer,
    FedProxServer,
    SecureAggregationSimulator,
)

# ---------------------------------------------------------------------------
# Shared tiny model factory
# ---------------------------------------------------------------------------

D_MODEL = 8
VOCAB_SIZE = 8
N_CLIENTS = 3
B, T = 2, 4


def tiny_model() -> nn.Module:
    """Embedding + linear head, accepts (B, T) long tensor, outputs (B, T, V)."""

    class _TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
            self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.embed(x))

    return _TinyLM()


def make_batch():
    data = [torch.randint(0, VOCAB_SIZE, (B, T))]
    labels = [torch.randint(0, VOCAB_SIZE, (B, T))]
    return data, labels


def make_clients(global_model: nn.Module, n: int = N_CLIENTS) -> list:
    return [
        FederatedClient(
            client_id=i,
            model=global_model,
            lr=1e-2,
            n_local_steps=3,
        )
        for i in range(n)
    ]


# ===========================================================================
# FederatedClient tests
# ===========================================================================


def test_client_local_update_returns_dict():
    """local_update must return a dict keyed by parameter names."""
    model = tiny_model()
    client = FederatedClient(0, model, lr=1e-2, n_local_steps=2)
    data, labels = make_batch()
    delta = client.local_update(data, labels)
    assert isinstance(delta, dict)
    assert len(delta) > 0
    for k, v in delta.items():
        assert isinstance(k, str)
        assert isinstance(v, torch.Tensor)


def test_client_local_update_delta_nonzero():
    """delta_w should be non-zero after at least one gradient step."""
    model = tiny_model()
    client = FederatedClient(0, model, lr=1e-1, n_local_steps=3)
    data, labels = make_batch()
    delta = client.local_update(data, labels)
    total_norm = sum(v.pow(2).sum().item() for v in delta.values())
    assert total_norm > 0.0, "Expected non-zero delta_w after local update"


def test_client_local_update_keys_match_params():
    """Keys in delta_w must exactly match model parameter names."""
    model = tiny_model()
    client = FederatedClient(0, model, lr=1e-2, n_local_steps=2)
    data, labels = make_batch()
    delta = client.local_update(data, labels)
    expected_keys = set(n for n, _ in model.named_parameters())
    assert set(delta.keys()) == expected_keys


def test_client_fedprox_update_returns_dict():
    """fedprox_update must return a dict of parameter tensors."""
    model = tiny_model()
    client = FederatedClient(0, model, lr=1e-2, n_local_steps=2)
    data, labels = make_batch()
    global_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    delta = client.fedprox_update(data, labels, global_params, mu=0.01)
    assert isinstance(delta, dict)
    assert len(delta) > 0
    for k, v in delta.items():
        assert isinstance(v, torch.Tensor)


def test_client_fedprox_update_keys_match_params():
    """fedprox_update keys must match model parameter names."""
    model = tiny_model()
    client = FederatedClient(0, model, lr=1e-2, n_local_steps=2)
    data, labels = make_batch()
    global_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    delta = client.fedprox_update(data, labels, global_params, mu=0.01)
    expected_keys = set(n for n, _ in model.named_parameters())
    assert set(delta.keys()) == expected_keys


# ===========================================================================
# DifferentiallyPrivateAggregator tests
# ===========================================================================


def _make_delta_w(model: nn.Module, scale: float = 1.0) -> dict:
    return {
        n: torch.randn_like(p.detach().float()) * scale
        for n, p in model.named_parameters()
    }


def test_dp_clip_reduces_norm():
    """clip_update must ensure global L2 norm <= max_norm."""
    model = tiny_model()
    agg = DifferentiallyPrivateAggregator(noise_multiplier=0.0, max_norm=1.0)
    # Create update with large norm
    delta = _make_delta_w(model, scale=100.0)
    clipped = agg.clip_update(delta)
    total_norm = math.sqrt(sum(v.pow(2).sum().item() for v in clipped.values()))
    assert total_norm <= 1.0 + 1e-5, f"Norm after clipping: {total_norm:.6f}"


def test_dp_clip_does_not_amplify_small_update():
    """An update already within max_norm should not be amplified."""
    model = tiny_model()
    agg = DifferentiallyPrivateAggregator(noise_multiplier=0.0, max_norm=1.0)
    # Create update with tiny norm
    delta = _make_delta_w(model, scale=1e-6)
    original_norm = math.sqrt(sum(v.pow(2).sum().item() for v in delta.values()))
    clipped = agg.clip_update(delta)
    clipped_norm = math.sqrt(sum(v.pow(2).sum().item() for v in clipped.values()))
    assert clipped_norm <= original_norm + 1e-8


def test_dp_add_noise_changes_update():
    """add_noise must produce a different update than the input."""
    model = tiny_model()
    agg = DifferentiallyPrivateAggregator(noise_multiplier=1.0, max_norm=1.0)
    delta = _make_delta_w(model, scale=1.0)
    noised = agg.add_noise(delta)
    # At least one tensor should differ
    any_diff = any(
        not torch.allclose(delta[k], noised[k]) for k in delta
    )
    assert any_diff, "add_noise must modify the update"


def test_dp_aggregate_returns_dict():
    """aggregate must return a dict with the same keys as updates."""
    model = tiny_model()
    agg = DifferentiallyPrivateAggregator(noise_multiplier=0.1, max_norm=1.0)
    updates = [_make_delta_w(model) for _ in range(3)]
    result = agg.aggregate(updates, n_clients_total=5)
    assert isinstance(result, dict)
    expected_keys = set(n for n, _ in model.named_parameters())
    assert set(result.keys()) == expected_keys


def test_dp_aggregate_correct_shape():
    """aggregate output tensors must have the same shape as the parameters."""
    model = tiny_model()
    agg = DifferentiallyPrivateAggregator(noise_multiplier=0.0, max_norm=1.0)
    updates = [_make_delta_w(model) for _ in range(2)]
    result = agg.aggregate(updates, n_clients_total=3)
    for n, p in model.named_parameters():
        assert result[n].shape == p.shape, f"Shape mismatch for {n}"


# ===========================================================================
# FedAvgServer tests
# ===========================================================================


def test_server_select_clients_correct_fraction():
    """select_clients must return ceil/round of client_fraction * n_clients."""
    model = tiny_model()
    server = FedAvgServer(model, n_clients=10, client_fraction=0.4)
    selected = server.select_clients(round_id=0)
    expected_k = max(1, round(0.4 * 10))
    assert len(selected) == expected_k


def test_server_select_clients_ids_in_range():
    """All selected client IDs must be in [0, n_clients)."""
    model = tiny_model()
    server = FedAvgServer(model, n_clients=8, client_fraction=0.5)
    selected = server.select_clients(round_id=1)
    assert all(0 <= cid < 8 for cid in selected)


def test_server_aggregate_updates_weighted_average():
    """aggregate_updates with equal weights should be a plain average."""
    model = tiny_model()
    server = FedAvgServer(model, n_clients=N_CLIENTS)
    u1 = {n: torch.ones_like(p.detach().float()) for n, p in model.named_parameters()}
    u2 = {n: torch.ones_like(p.detach().float()) * 3.0 for n, p in model.named_parameters()}
    # Equal weights → average should be 2.0 everywhere
    agg = server.aggregate_updates([u1, u2], weights=[1.0, 1.0])
    for k, v in agg.items():
        assert torch.allclose(v, torch.full_like(v, 2.0), atol=1e-5), \
            f"Expected 2.0, got {v.mean().item():.4f}"


def test_server_update_global_changes_params():
    """update_global must modify the global model's parameters."""
    model = tiny_model()
    server = FedAvgServer(model, n_clients=N_CLIENTS)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    # Large delta so the change is easily detectable
    delta = {n: torch.ones_like(p.float()) * 5.0 for n, p in model.named_parameters()}
    server.update_global(delta)
    after = {n: p.detach().clone() for n, p in model.named_parameters()}
    any_changed = any(
        not torch.allclose(before[n], after[n]) for n in before
    )
    assert any_changed, "update_global must change at least one parameter"


def test_server_broadcast_returns_current_params():
    """broadcast must return a dict matching current global_params."""
    model = tiny_model()
    server = FedAvgServer(model, n_clients=N_CLIENTS)
    broadcasted = server.broadcast()
    assert isinstance(broadcasted, dict)
    for n, p in model.named_parameters():
        assert torch.allclose(
            broadcasted[n].float(), p.detach().float(), atol=1e-6
        ), f"broadcast mismatch for {n}"


# ===========================================================================
# SecureAggregationSimulator tests
# ===========================================================================


def test_secure_agg_generate_masks_produces_pairs():
    """generate_masks must produce an entry for each client."""
    sim = SecureAggregationSimulator(n_clients=3)
    client_ids = [0, 1, 2]
    masks = sim.generate_masks(client_ids)
    assert set(masks.keys()) == set(client_ids)


def test_secure_agg_pairwise_cancellation():
    """Masks must satisfy r_ij + r_ji = 0 (pairwise cancellation)."""
    model = tiny_model()
    sim = SecureAggregationSimulator(n_clients=3)
    client_ids = [0, 1, 2]
    masks = sim.generate_masks(client_ids)
    data, labels = make_batch()
    delta_w = _make_delta_w(model)

    # Get masked updates for every client
    masked_updates = {}
    for cid in client_ids:
        masked_updates[cid] = sim.mask_update(cid, delta_w, masks)

    # Sum of all masks across clients should be ~0 per key
    for k in delta_w:
        total_mask = sum(masks[cid][k] for cid in client_ids)
        assert torch.allclose(
            total_mask, torch.zeros_like(total_mask), atol=1e-5
        ), f"Masks do not cancel for key {k}: norm={total_mask.norm().item()}"


def test_secure_agg_verify_cancellation_returns_true():
    """verify_cancellation must return True for a valid set of masked updates."""
    model = tiny_model()
    sim = SecureAggregationSimulator(n_clients=3)
    client_ids = [0, 1, 2]
    masks = sim.generate_masks(client_ids)
    delta_w = _make_delta_w(model)
    masked_list = [
        sim.mask_update(cid, delta_w, masks) for cid in client_ids
    ]
    result = sim.verify_cancellation(masked_list)
    assert result is True


# ===========================================================================
# FederatedDPTrainer tests
# ===========================================================================


def test_trainer_train_round_returns_finite_loss():
    """train_round must return a finite avg_loss."""
    model = tiny_model()
    server = FedAvgServer(model, n_clients=N_CLIENTS, client_fraction=1.0)
    agg = DifferentiallyPrivateAggregator(noise_multiplier=0.5, max_norm=1.0)
    trainer = FederatedDPTrainer(server, agg, n_rounds=5)
    clients = make_clients(model)
    avg_loss, epsilon = trainer.train_round(clients, round_id=0)
    assert math.isfinite(avg_loss), f"avg_loss not finite: {avg_loss}"


def test_trainer_train_round_returns_positive_epsilon():
    """train_round must return a positive privacy budget epsilon."""
    model = tiny_model()
    server = FedAvgServer(model, n_clients=N_CLIENTS, client_fraction=1.0)
    agg = DifferentiallyPrivateAggregator(noise_multiplier=1.0, max_norm=1.0, delta=1e-5)
    trainer = FederatedDPTrainer(server, agg, n_rounds=5)
    clients = make_clients(model)
    _, epsilon = trainer.train_round(clients, round_id=0)
    assert epsilon > 0.0, f"Expected positive epsilon, got {epsilon}"


def test_trainer_privacy_budget_positive():
    """privacy_budget must return a positive float for valid inputs."""
    model = tiny_model()
    server = FedAvgServer(model, n_clients=10)
    agg = DifferentiallyPrivateAggregator(noise_multiplier=1.0, max_norm=1.0, delta=1e-5)
    trainer = FederatedDPTrainer(server, agg, n_rounds=10)
    eps = trainer.privacy_budget(n_rounds=5, n_clients_selected=3, n_total=10)
    assert eps > 0.0, f"Expected positive privacy budget, got {eps}"


def test_trainer_privacy_budget_increases_with_rounds():
    """Privacy budget should increase with more rounds."""
    model = tiny_model()
    server = FedAvgServer(model, n_clients=10)
    agg = DifferentiallyPrivateAggregator(noise_multiplier=1.0, max_norm=1.0, delta=1e-5)
    trainer = FederatedDPTrainer(server, agg, n_rounds=10)
    eps_fewer = trainer.privacy_budget(2, 5, 10)
    eps_more = trainer.privacy_budget(8, 5, 10)
    assert eps_more >= eps_fewer, "More rounds should not decrease epsilon"


# ===========================================================================
# FedProxServer tests
# ===========================================================================


def test_fedprox_server_inherits_fedavg_interface():
    """FedProxServer must expose all FedAvgServer methods."""
    model = tiny_model()
    server = FedProxServer(model, n_clients=N_CLIENTS, client_fraction=1.0, mu=0.01)
    assert hasattr(server, "select_clients")
    assert hasattr(server, "aggregate_updates")
    assert hasattr(server, "update_global")
    assert hasattr(server, "broadcast")
    assert isinstance(server, FedAvgServer)


def test_fedprox_server_aggregate_updates_runs():
    """FedProxServer.aggregate_updates must return a valid dict."""
    model = tiny_model()
    server = FedProxServer(model, n_clients=N_CLIENTS, mu=0.05)
    u1 = {n: torch.ones_like(p.detach().float()) for n, p in model.named_parameters()}
    u2 = {n: torch.ones_like(p.detach().float()) * 2.0 for n, p in model.named_parameters()}
    result = server.aggregate_updates([u1, u2], weights=[1.0, 1.0])
    assert isinstance(result, dict)
    for k, v in result.items():
        assert isinstance(v, torch.Tensor)


# ===========================================================================
# FedDPConfig tests
# ===========================================================================


def test_feddpconfig_defaults():
    """FedDPConfig must have the correct default field values."""
    cfg = FedDPConfig()
    assert cfg.n_clients == 5
    assert cfg.client_fraction == 0.6
    assert cfg.n_local_steps == 3
    assert cfg.noise_multiplier == 1.0
    assert cfg.max_norm == 1.0
    assert cfg.delta == 1e-5
    assert cfg.mu == 0.01
    assert cfg.n_rounds == 5
    assert cfg.lr == 1e-3


def test_feddpconfig_custom_values():
    """FedDPConfig must accept custom overrides."""
    cfg = FedDPConfig(n_clients=10, lr=5e-4, n_rounds=20)
    assert cfg.n_clients == 10
    assert cfg.lr == 5e-4
    assert cfg.n_rounds == 20
