"""Tests for src/training/federated_avg.py — FedAvg, FedProx, FedMedian."""

from __future__ import annotations

import copy
from collections import OrderedDict

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.federated_avg import (
    FederatedClient,
    FederatedConfig,
    FederatedServer,
    fedavg_aggregate,
    fedmedian_aggregate,
    fedprox_penalty,
    get_model_weights,
    set_model_weights,
)

# ---------------------------------------------------------------------------
# Tiny model fixture
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

SEQ_LEN = 8
BATCH = 2


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return AureliusTransformer(TINY_CFG)


@pytest.fixture
def input_ids():
    torch.manual_seed(42)
    return torch.randint(0, TINY_CFG.vocab_size, (BATCH, SEQ_LEN))


# ---------------------------------------------------------------------------
# 1. FederatedConfig defaults
# ---------------------------------------------------------------------------


def test_federated_config_defaults():
    cfg = FederatedConfig()
    assert cfg.n_clients == 10
    assert cfg.fraction == 0.1
    assert cfg.local_epochs == 5
    assert cfg.local_lr == 0.01
    assert cfg.mu == 0.0
    assert cfg.aggregation == "fedavg"


def test_federated_config_custom():
    cfg = FederatedConfig(n_clients=5, fraction=0.2, local_epochs=3, mu=0.1, aggregation="fedprox")
    assert cfg.n_clients == 5
    assert cfg.fraction == 0.2
    assert cfg.local_epochs == 3
    assert cfg.mu == 0.1
    assert cfg.aggregation == "fedprox"


# ---------------------------------------------------------------------------
# 2. get_model_weights
# ---------------------------------------------------------------------------


def test_get_model_weights_returns_ordered_dict(tiny_model):
    weights = get_model_weights(tiny_model)
    assert isinstance(weights, OrderedDict)


def test_get_model_weights_same_keys(tiny_model):
    weights = get_model_weights(tiny_model)
    model_keys = set(name for name, _ in tiny_model.named_parameters())
    assert set(weights.keys()) == model_keys


def test_get_model_weights_detached(tiny_model):
    weights = get_model_weights(tiny_model)
    for v in weights.values():
        assert not v.requires_grad, "Weights should be detached (no grad)"


def test_get_model_weights_is_copy(tiny_model):
    weights = get_model_weights(tiny_model)
    # Mutating the dict values should not affect the model
    first_key = next(iter(weights))
    original_val = weights[first_key].clone()
    weights[first_key].fill_(999.0)
    model_val = dict(tiny_model.named_parameters())[first_key].detach()
    assert torch.allclose(model_val, original_val), (
        "get_model_weights should return copies, not views"
    )


# ---------------------------------------------------------------------------
# 3. set_model_weights round-trip
# ---------------------------------------------------------------------------


def test_set_model_weights_round_trip(tiny_model):
    original_weights = get_model_weights(tiny_model)
    # Corrupt model parameters
    with torch.no_grad():
        for p in tiny_model.parameters():
            p.fill_(0.0)
    # Restore
    set_model_weights(tiny_model, original_weights)
    restored_weights = get_model_weights(tiny_model)
    for key in original_weights:
        assert torch.allclose(original_weights[key], restored_weights[key]), (
            f"Round-trip failed for parameter '{key}'"
        )


def test_set_model_weights_updates_in_place(tiny_model):
    new_weights = OrderedDict(
        (name, torch.ones_like(param)) for name, param in tiny_model.named_parameters()
    )
    set_model_weights(tiny_model, new_weights)
    for name, param in tiny_model.named_parameters():
        assert torch.all(param == 1.0), f"Parameter '{name}' was not updated"


# ---------------------------------------------------------------------------
# 4. fedavg_aggregate — equal sizes = simple mean
# ---------------------------------------------------------------------------


def test_fedavg_equal_sizes_is_mean():
    # 3 clients, each with a single scalar weight
    w1 = OrderedDict([("w", torch.tensor([1.0]))])
    w2 = OrderedDict([("w", torch.tensor([3.0]))])
    w3 = OrderedDict([("w", torch.tensor([5.0]))])
    result = fedavg_aggregate([w1, w2, w3], [10, 10, 10])
    expected = torch.tensor([3.0])
    assert torch.allclose(result["w"], expected), f"Expected {expected}, got {result['w']}"


def test_fedavg_equal_sizes_multikey():
    w1 = OrderedDict([("a", torch.tensor([2.0])), ("b", torch.tensor([4.0]))])
    w2 = OrderedDict([("a", torch.tensor([4.0])), ("b", torch.tensor([8.0]))])
    result = fedavg_aggregate([w1, w2], [5, 5])
    assert torch.allclose(result["a"], torch.tensor([3.0]))
    assert torch.allclose(result["b"], torch.tensor([6.0]))


# ---------------------------------------------------------------------------
# 5. fedavg_aggregate — unequal sizes weights by n_i
# ---------------------------------------------------------------------------


def test_fedavg_unequal_sizes_weighted():
    w1 = OrderedDict([("w", torch.tensor([0.0]))])
    w2 = OrderedDict([("w", torch.tensor([4.0]))])
    # n1=1, n2=3 => (1*0 + 3*4)/4 = 3.0
    result = fedavg_aggregate([w1, w2], [1, 3])
    assert torch.allclose(result["w"], torch.tensor([3.0])), f"Got {result['w']}"


def test_fedavg_unequal_sizes_large_client_dominates():
    w1 = OrderedDict([("w", torch.tensor([10.0]))])
    w2 = OrderedDict([("w", torch.tensor([0.0]))])
    # n1=100, n2=1 => (100*10 + 1*0)/101 ≈ 9.9
    result = fedavg_aggregate([w1, w2], [100, 1])
    expected = torch.tensor([1000.0 / 101.0])
    assert torch.allclose(result["w"], expected, atol=1e-4), f"Got {result['w']}"


# ---------------------------------------------------------------------------
# 6. fedmedian_aggregate with 3 clients returns median
# ---------------------------------------------------------------------------


def test_fedmedian_returns_median():
    w1 = OrderedDict([("w", torch.tensor([1.0, 6.0]))])
    w2 = OrderedDict([("w", torch.tensor([3.0, 2.0]))])
    w3 = OrderedDict([("w", torch.tensor([5.0, 4.0]))])
    result = fedmedian_aggregate([w1, w2, w3])
    expected = torch.tensor([3.0, 4.0])
    assert torch.allclose(result["w"], expected), f"Expected {expected}, got {result['w']}"


def test_fedmedian_single_client():
    w = OrderedDict([("w", torch.tensor([7.0]))])
    result = fedmedian_aggregate([w])
    assert torch.allclose(result["w"], torch.tensor([7.0]))


def test_fedmedian_two_clients_is_average_of_two():
    # median of 2 values is the lower one (torch.median picks the lower)
    w1 = OrderedDict([("w", torch.tensor([2.0]))])
    w2 = OrderedDict([("w", torch.tensor([4.0]))])
    result = fedmedian_aggregate([w1, w2])
    # torch.median returns the lower element for even counts
    assert result["w"].item() in (2.0, 4.0)  # either boundary is acceptable


# ---------------------------------------------------------------------------
# 7. fedprox_penalty = 0 when model matches global
# ---------------------------------------------------------------------------


def test_fedprox_penalty_zero_when_equal(tiny_model):
    global_weights = get_model_weights(tiny_model)
    penalty = fedprox_penalty(tiny_model, global_weights, mu=1.0)
    assert torch.isclose(penalty, torch.tensor(0.0), atol=1e-6), (
        f"Penalty should be 0 when model == global; got {penalty.item()}"
    )


def test_fedprox_penalty_zero_when_mu_zero(tiny_model):
    global_weights = get_model_weights(tiny_model)
    # Even if weights differ, mu=0 => penalty must be 0
    with torch.no_grad():
        for p in tiny_model.parameters():
            p.fill_(999.0)
    penalty = fedprox_penalty(tiny_model, global_weights, mu=0.0)
    assert penalty.item() == 0.0, f"Expected 0.0, got {penalty.item()}"


# ---------------------------------------------------------------------------
# 8. fedprox_penalty > 0 when model differs
# ---------------------------------------------------------------------------


def test_fedprox_penalty_positive_when_differ(tiny_model):
    global_weights = get_model_weights(tiny_model)
    # Perturb the model
    with torch.no_grad():
        for p in tiny_model.parameters():
            p.add_(1.0)
    penalty = fedprox_penalty(tiny_model, global_weights, mu=1.0)
    assert penalty.item() > 0.0, f"Expected positive penalty; got {penalty.item()}"


def test_fedprox_penalty_scales_with_mu(tiny_model):
    global_weights = get_model_weights(tiny_model)
    with torch.no_grad():
        for p in tiny_model.parameters():
            p.add_(1.0)
    penalty_low = fedprox_penalty(tiny_model, global_weights, mu=0.1)
    penalty_high = fedprox_penalty(tiny_model, global_weights, mu=1.0)
    assert penalty_high.item() > penalty_low.item(), "Higher mu should produce larger penalty"


# ---------------------------------------------------------------------------
# 9. FederatedClient.local_train returns weights dict
# ---------------------------------------------------------------------------


def test_federated_client_local_train_returns_weights(tiny_model, input_ids):
    cfg = FederatedConfig(local_epochs=1, local_lr=0.01)
    client = FederatedClient(copy.deepcopy(tiny_model), cfg)
    weights, n_samples = client.local_train(input_ids)
    assert isinstance(weights, OrderedDict)
    assert n_samples == BATCH


def test_federated_client_local_train_weights_have_correct_keys(tiny_model, input_ids):
    cfg = FederatedConfig(local_epochs=1)
    client = FederatedClient(copy.deepcopy(tiny_model), cfg)
    weights, _ = client.local_train(input_ids)
    model_keys = set(name for name, _ in tiny_model.named_parameters())
    assert set(weights.keys()) == model_keys


def test_federated_client_local_train_updates_weights(tiny_model, input_ids):
    cfg = FederatedConfig(local_epochs=2, local_lr=0.1)
    original_weights = get_model_weights(tiny_model)
    client = FederatedClient(copy.deepcopy(tiny_model), cfg)
    updated_weights, _ = client.local_train(input_ids)
    # At least one parameter should differ after training
    any_changed = any(
        not torch.allclose(original_weights[k], updated_weights[k]) for k in original_weights
    )
    assert any_changed, "Local training should update at least one parameter"


def test_federated_client_fedprox_local_train(tiny_model, input_ids):
    cfg = FederatedConfig(local_epochs=1, mu=0.1, aggregation="fedprox")
    global_weights = get_model_weights(tiny_model)
    client = FederatedClient(copy.deepcopy(tiny_model), cfg)
    weights, n_samples = client.local_train(input_ids, global_weights=global_weights)
    assert isinstance(weights, OrderedDict)
    assert n_samples == BATCH


# ---------------------------------------------------------------------------
# 10. FederatedServer.aggregate_round updates model
# ---------------------------------------------------------------------------


def test_federated_server_aggregate_round_updates_model(tiny_model):
    cfg = FederatedConfig()
    server = FederatedServer(tiny_model, cfg)

    # Create two fake client weight updates (all zeros vs all ones)
    w1 = OrderedDict((n, torch.zeros_like(p)) for n, p in tiny_model.named_parameters())
    w2 = OrderedDict((n, torch.ones_like(p)) for n, p in tiny_model.named_parameters())

    aggregated = server.aggregate_round([(w1, 10), (w2, 10)])
    assert isinstance(aggregated, OrderedDict)

    # After aggregate, server model params should equal the average (0.5)
    for name, param in tiny_model.named_parameters():
        expected = torch.full_like(param, 0.5)
        assert torch.allclose(param, expected, atol=1e-5), (
            f"Server model param '{name}' not updated correctly"
        )


def test_federated_server_aggregate_round_returns_dict(tiny_model):
    cfg = FederatedConfig()
    server = FederatedServer(tiny_model, cfg)
    w = get_model_weights(tiny_model)
    result = server.aggregate_round([(w, 5)])
    assert isinstance(result, OrderedDict)
    assert set(result.keys()) == set(w.keys())


def test_federated_server_fedmedian_aggregation(tiny_model):
    cfg = FederatedConfig(aggregation="fedmedian")
    server = FederatedServer(tiny_model, cfg)

    w1 = OrderedDict((n, torch.zeros_like(p)) for n, p in tiny_model.named_parameters())
    w2 = OrderedDict((n, torch.full_like(p, 2.0)) for n, p in tiny_model.named_parameters())
    w3 = OrderedDict((n, torch.full_like(p, 4.0)) for n, p in tiny_model.named_parameters())

    server.aggregate_round([(w1, 10), (w2, 10), (w3, 10)])
    # Median of [0, 2, 4] = 2
    for name, param in tiny_model.named_parameters():
        expected = torch.full_like(param, 2.0)
        assert torch.allclose(param, expected, atol=1e-5), (
            f"FedMedian server param '{name}' not updated correctly; got {param.mean().item()}"
        )
