"""Tests for src/training/federated.py — FedAvg + DP-SGD federated learning simulation."""

from __future__ import annotations

import copy
import random

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.federated import (
    FederatedConfig,
    ClientSimulator,
    FederatedTrainer,
    add_dp_noise,
    clip_gradients_dp,
    fedavg_aggregate,
    fedprox_loss,
)


# ---------------------------------------------------------------------------
# Shared fixtures
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
BATCH_SIZE = 2


def tiny_model() -> AureliusTransformer:
    return AureliusTransformer(TINY_CFG)


def tiny_config(**kwargs) -> FederatedConfig:
    defaults = dict(
        n_clients=10,
        fraction_clients=0.3,
        local_steps=2,
        local_lr=1e-3,
        noise_multiplier=1.0,
        clip_norm=1.0,
        aggregation="fedavg",
        mu=0.01,
    )
    defaults.update(kwargs)
    return FederatedConfig(**defaults)


def random_input(batch: int = BATCH_SIZE, seq: int = SEQ_LEN) -> torch.Tensor:
    return torch.randint(0, 256, (batch, seq))


def make_client_data(n_clients: int, batch: int = BATCH_SIZE) -> dict[int, torch.Tensor]:
    return {i: random_input(batch) for i in range(n_clients)}


# ---------------------------------------------------------------------------
# Test 1: FederatedConfig defaults
# ---------------------------------------------------------------------------

def test_federated_config_defaults():
    """FederatedConfig has the correct default values."""
    cfg = FederatedConfig()
    assert cfg.n_clients == 10
    assert cfg.fraction_clients == 0.3
    assert cfg.local_steps == 5
    assert cfg.local_lr == 1e-3
    assert cfg.noise_multiplier == 1.0
    assert cfg.clip_norm == 1.0
    assert cfg.aggregation == "fedavg"
    assert cfg.mu == 0.01


# ---------------------------------------------------------------------------
# Test 2: clip_gradients_dp clips grad norm
# ---------------------------------------------------------------------------

def test_clip_gradients_dp_clips_norm():
    """clip_gradients_dp ensures the total gradient norm is <= clip_norm."""
    model = tiny_model()
    # Set all gradients to large values
    for param in model.parameters():
        param.grad = torch.ones_like(param) * 100.0

    clip_norm = 1.0
    clip_gradients_dp(model, clip_norm)

    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    assert total_norm <= clip_norm + 1e-5, (
        f"Expected grad norm <= {clip_norm}, got {total_norm}"
    )


# ---------------------------------------------------------------------------
# Test 3: add_dp_noise changes parameter gradients
# ---------------------------------------------------------------------------

def test_add_dp_noise_changes_gradients():
    """add_dp_noise modifies the gradients in-place by adding Gaussian noise."""
    model = tiny_model()
    for param in model.parameters():
        param.grad = torch.zeros_like(param)

    # Record original grad values (all zeros)
    original_grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    add_dp_noise(model, noise_multiplier=1.0, clip_norm=1.0, n_samples=10)

    changed = False
    for name, param in model.named_parameters():
        if param.grad is not None and name in original_grads:
            if not torch.allclose(param.grad, original_grads[name]):
                changed = True
                break

    assert changed, "add_dp_noise should modify at least one gradient"


# ---------------------------------------------------------------------------
# Test 4: fedavg_aggregate returns correct mean
# ---------------------------------------------------------------------------

def test_fedavg_aggregate_returns_mean():
    """fedavg_aggregate with uniform weights returns the element-wise mean."""
    model = tiny_model()
    global_params = {k: v.detach().clone() for k, v in model.named_parameters()}

    params_a = {k: torch.zeros_like(v) for k, v in global_params.items()}
    params_b = {k: torch.ones_like(v) * 2.0 for k, v in global_params.items()}

    result = fedavg_aggregate(global_params, [params_a, params_b])

    for key in global_params:
        expected = (params_a[key].float() + params_b[key].float()) / 2.0
        assert torch.allclose(result[key].float(), expected, atol=1e-5), (
            f"Mean mismatch at {key}"
        )


# ---------------------------------------------------------------------------
# Test 5: fedavg_aggregate with weights upweights specified client
# ---------------------------------------------------------------------------

def test_fedavg_aggregate_weighted_upweights_client():
    """fedavg_aggregate with custom weights shifts result toward the higher-weight client."""
    model = tiny_model()
    global_params = {k: v.detach().clone() for k, v in model.named_parameters()}

    params_a = {k: torch.zeros_like(v) for k, v in global_params.items()}
    params_b = {k: torch.ones_like(v) * 10.0 for k, v in global_params.items()}

    # Weight 9:1 toward params_b
    result = fedavg_aggregate(global_params, [params_a, params_b], weights=[1.0, 9.0])

    # Result should be closer to params_b (10.0) than params_a (0.0)
    key = list(global_params.keys())[0]
    dist_to_a = (result[key].float() - params_a[key].float()).abs().mean().item()
    dist_to_b = (result[key].float() - params_b[key].float()).abs().mean().item()
    assert dist_to_b < dist_to_a, (
        "Weighted avg should be closer to the high-weight client"
    )


# ---------------------------------------------------------------------------
# Test 6: fedprox_loss is zero when params are identical
# ---------------------------------------------------------------------------

def test_fedprox_loss_zero_identical_params():
    """fedprox_loss returns 0 when model_params equals global_params."""
    model = tiny_model()
    params = {k: v.detach().clone() for k, v in model.named_parameters()}

    loss = fedprox_loss(params, params, mu=0.01)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 7: fedprox_loss > 0 when params differ
# ---------------------------------------------------------------------------

def test_fedprox_loss_positive_when_params_differ():
    """fedprox_loss is positive when model_params differ from global_params."""
    model = tiny_model()
    global_p = {k: v.detach().clone() for k, v in model.named_parameters()}
    local_p = {k: v.detach().clone() + 1.0 for k, v in model.named_parameters()}

    loss = fedprox_loss(local_p, global_p, mu=0.01)
    assert loss.item() > 0.0, "fedprox_loss should be positive when params differ"


# ---------------------------------------------------------------------------
# Test 8: ClientSimulator local_train returns param dict
# ---------------------------------------------------------------------------

def test_client_simulator_local_train_returns_param_dict():
    """local_train returns a dict of parameter tensors."""
    model = tiny_model()
    cfg = tiny_config(local_steps=1)
    client = ClientSimulator(client_id=0, model=model, config=cfg)

    data = random_input()
    global_params = {k: v.detach().clone() for k, v in model.named_parameters()}
    result = client.local_train(data, global_params)

    assert isinstance(result, dict), "local_train must return a dict"
    assert len(result) > 0, "Returned param dict must not be empty"
    for key, val in result.items():
        assert isinstance(val, torch.Tensor), f"Value for {key} must be a Tensor"


# ---------------------------------------------------------------------------
# Test 9: ClientSimulator get/set params round-trips correctly
# ---------------------------------------------------------------------------

def test_client_simulator_get_set_params_roundtrip():
    """set_params followed by get_params returns the same parameter values."""
    model = tiny_model()
    cfg = tiny_config()
    client = ClientSimulator(client_id=0, model=model, config=cfg)

    # Build reference params with a known constant
    ref_params = {k: torch.full_like(v, 0.42) for k, v in model.named_parameters()}

    client.set_params(ref_params)
    retrieved = client.get_params()

    for key in ref_params:
        assert torch.allclose(retrieved[key].float(), ref_params[key].float(), atol=1e-6), (
            f"Round-trip mismatch at {key}"
        )


# ---------------------------------------------------------------------------
# Test 10: FederatedTrainer select_clients returns correct count
# ---------------------------------------------------------------------------

def test_federated_trainer_select_clients_count():
    """select_clients returns floor(fraction_clients * n_clients) >= 1 clients."""
    cfg = tiny_config(n_clients=10, fraction_clients=0.3)
    model = tiny_model()
    trainer = FederatedTrainer(global_model=model, config=cfg)

    rng = random.Random(42)
    selected = trainer.select_clients(rng)

    expected_count = max(1, int(cfg.fraction_clients * cfg.n_clients))
    assert len(selected) == expected_count, (
        f"Expected {expected_count} clients, got {len(selected)}"
    )
    # All IDs in valid range
    for cid in selected:
        assert 0 <= cid < cfg.n_clients


# ---------------------------------------------------------------------------
# Test 11: FederatedTrainer train_round returns correct keys
# ---------------------------------------------------------------------------

def test_federated_trainer_train_round_returns_correct_keys():
    """train_round returns a dict with 'round_loss', 'n_clients_selected', 'noise_scale'."""
    cfg = tiny_config(n_clients=5, fraction_clients=0.4, local_steps=1)
    model = tiny_model()
    trainer = FederatedTrainer(global_model=model, config=cfg)

    client_data = make_client_data(cfg.n_clients)
    rng = random.Random(0)
    result = trainer.train_round(client_data, rng)

    assert "round_loss" in result, "Missing 'round_loss'"
    assert "n_clients_selected" in result, "Missing 'n_clients_selected'"
    assert "noise_scale" in result, "Missing 'noise_scale'"
    assert isinstance(result["round_loss"], float)
    assert isinstance(result["n_clients_selected"], int)
    assert isinstance(result["noise_scale"], float)


# ---------------------------------------------------------------------------
# Test 12: FederatedTrainer train_round updates global model params
# ---------------------------------------------------------------------------

def test_federated_trainer_train_round_updates_global_model():
    """train_round modifies the global model parameters."""
    cfg = tiny_config(n_clients=5, fraction_clients=0.4, local_steps=2)
    model = tiny_model()
    trainer = FederatedTrainer(global_model=model, config=cfg)

    # Capture original params
    original_params = {
        k: v.detach().clone() for k, v in model.named_parameters()
    }

    client_data = make_client_data(cfg.n_clients)
    rng = random.Random(7)
    trainer.train_round(client_data, rng)

    # At least some parameters should have changed
    changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param.data, original_params[name], atol=1e-8):
            changed = True
            break

    assert changed, "train_round should update global model parameters"


# ---------------------------------------------------------------------------
# Test 13: fedavg_aggregate uniform weights gives exactly the mean
# ---------------------------------------------------------------------------

def test_fedavg_aggregate_uniform_exact_mean():
    """fedavg_aggregate with uniform weights produces the exact arithmetic mean."""
    model = tiny_model()
    global_params = {k: v.detach().clone() for k, v in model.named_parameters()}

    values = [0.0, 2.0, 4.0, 6.0]
    client_param_list = [
        {k: torch.full_like(v, val) for k, v in global_params.items()}
        for val in values
    ]

    result = fedavg_aggregate(global_params, client_param_list)
    expected_mean = sum(values) / len(values)  # 3.0

    for key in global_params:
        actual = result[key].float().mean().item()
        assert abs(actual - expected_mean) < 1e-5, (
            f"Expected mean {expected_mean}, got {actual} at {key}"
        )
