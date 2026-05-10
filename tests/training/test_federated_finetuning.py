"""Tests for src/training/federated_finetuning.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from aurelius.training.federated_finetuning import (
    ClientDataShard,
    FedAggregator,
    FedLoRAAdapter,
    FedProxClient,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

D_IN = 16
D_OUT = 8
RANK = 2
N_LOCAL = 20
BATCH_SIZE = 4


def make_shard(n: int = N_LOCAL) -> ClientDataShard:
    data = torch.randn(n, D_IN)
    labels = torch.randn(n, D_OUT)
    return ClientDataShard(client_id=0, data=data, labels=labels)


def make_simple_model() -> nn.Module:
    """Tiny linear model for testing."""
    return nn.Linear(D_IN, D_OUT, bias=False)


# ---------------------------------------------------------------------------
# 1. ClientDataShard.sample_batch returns correct shapes
# ---------------------------------------------------------------------------


def test_sample_batch_shapes():
    shard = make_shard()
    data_b, labels_b = shard.sample_batch(BATCH_SIZE)
    assert data_b.shape == (BATCH_SIZE, D_IN)
    assert labels_b.shape == (BATCH_SIZE, D_OUT)


# ---------------------------------------------------------------------------
# 2. sample_batch with batch_size > N_local still works (wraps)
# ---------------------------------------------------------------------------


def test_sample_batch_wraps():
    shard = make_shard(n=5)
    # Request more than available — should not raise
    data_b, labels_b = shard.sample_batch(20)
    assert data_b.shape == (20, D_IN)
    assert labels_b.shape == (20, D_OUT)


# ---------------------------------------------------------------------------
# 3. FedProxClient.set_global_params loads model correctly
# ---------------------------------------------------------------------------


def test_set_global_params_loads_model():
    model = make_simple_model()
    client = FedProxClient(client_id=0, model=model)

    # Create a distinct global state with known values
    new_state = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
    client.set_global_params(new_state)

    for k, v in client.model.state_dict().items():
        assert torch.allclose(v, torch.ones_like(v)), f"Param {k} not loaded correctly"


# ---------------------------------------------------------------------------
# 4. proximal_term = 0 when model unchanged after set_global_params
# ---------------------------------------------------------------------------


def test_proximal_term_zero_after_set():
    model = make_simple_model()
    client = FedProxClient(client_id=0, model=model)
    client.set_global_params(model.state_dict())
    prox = client.proximal_term()
    assert prox.item() == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# 5. proximal_term > 0 after model params changed
# ---------------------------------------------------------------------------


def test_proximal_term_positive_after_change():
    model = make_simple_model()
    client = FedProxClient(client_id=0, model=model, mu=0.01)
    client.set_global_params(model.state_dict())

    # Perturb the model's parameters
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param))

    prox = client.proximal_term()
    assert prox.item() > 0.0


# ---------------------------------------------------------------------------
# 6. local_train returns expected keys
# ---------------------------------------------------------------------------


def test_local_train_keys():
    model = make_simple_model()
    client = FedProxClient(client_id=0, model=model)
    client.set_global_params(model.state_dict())
    shard = make_shard()

    loss_fn = nn.MSELoss()
    result = client.local_train(shard, loss_fn, n_steps=3, lr=0.01)

    assert set(result.keys()) == {"avg_task_loss", "avg_proximal", "n_steps"}


# ---------------------------------------------------------------------------
# 7. local_train avg_task_loss is finite
# ---------------------------------------------------------------------------


def test_local_train_finite_loss():
    model = make_simple_model()
    client = FedProxClient(client_id=0, model=model)
    client.set_global_params(model.state_dict())
    shard = make_shard()

    loss_fn = nn.MSELoss()
    result = client.local_train(shard, loss_fn, n_steps=5, lr=0.01)

    assert torch.isfinite(torch.tensor(result["avg_task_loss"]))
    assert result["n_steps"] == 5


# ---------------------------------------------------------------------------
# 8. get_update returns dict with same keys as model state_dict
# ---------------------------------------------------------------------------


def test_get_update_keys_match_state_dict():
    model = make_simple_model()
    client = FedProxClient(client_id=0, model=model)
    client.set_global_params(model.state_dict())

    update = client.get_update()
    assert set(update.keys()) == set(model.state_dict().keys())


# ---------------------------------------------------------------------------
# 9. FedLoRAAdapter.forward output shape (B, d_out)
# ---------------------------------------------------------------------------


def test_lora_forward_shape():
    adapter = FedLoRAAdapter(d_in=D_IN, d_out=D_OUT, rank=RANK)
    x = torch.randn(BATCH_SIZE, D_IN)
    out = adapter(x)
    assert out.shape == (BATCH_SIZE, D_OUT)


# ---------------------------------------------------------------------------
# 10. LoRA output is zero when B is initialized to zeros
# ---------------------------------------------------------------------------


def test_lora_output_zero_at_init():
    adapter = FedLoRAAdapter(d_in=D_IN, d_out=D_OUT, rank=RANK)
    # B is zero-initialized by spec; effective_weight = (alpha/rank) * B @ A = 0
    x = torch.randn(BATCH_SIZE, D_IN)
    out = adapter(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


# ---------------------------------------------------------------------------
# 11. get_adapter_params returns A and B
# ---------------------------------------------------------------------------


def test_get_adapter_params_keys():
    adapter = FedLoRAAdapter(d_in=D_IN, d_out=D_OUT, rank=RANK)
    params = adapter.get_adapter_params()
    assert "A" in params and "B" in params
    assert params["A"].shape == (RANK, D_IN)
    assert params["B"].shape == (D_OUT, RANK)


# ---------------------------------------------------------------------------
# 12. communication_bytes < full model bytes (rank << d)
# ---------------------------------------------------------------------------


def test_communication_bytes_less_than_full():
    adapter = FedLoRAAdapter(d_in=D_IN, d_out=D_OUT, rank=RANK)
    lora_bytes = adapter.communication_bytes()
    # Full weight matrix would be D_IN * D_OUT * 4 bytes
    full_bytes = D_IN * D_OUT * 4
    assert lora_bytes < full_bytes, (
        f"LoRA bytes ({lora_bytes}) should be less than full ({full_bytes})"
    )
    # Verify formula: (rank * d_in + d_out * rank) * 4
    expected = (RANK * D_IN + D_OUT * RANK) * 4
    assert lora_bytes == expected


# ---------------------------------------------------------------------------
# 13. FedAggregator.fedavg returns dict with correct keys
# ---------------------------------------------------------------------------


def test_fedavg_returns_correct_keys():
    agg = FedAggregator(n_clients=3)
    states = [make_simple_model().state_dict() for _ in range(3)]
    result = agg.fedavg(states)
    assert set(result.keys()) == set(states[0].keys())


# ---------------------------------------------------------------------------
# 14. fedavg with uniform weights averages correctly
# ---------------------------------------------------------------------------


def test_fedavg_uniform_average():
    agg = FedAggregator(n_clients=2)

    model_a = make_simple_model()
    model_b = make_simple_model()
    # Set known values
    with torch.no_grad():
        for p in model_a.parameters():
            p.fill_(0.0)
        for p in model_b.parameters():
            p.fill_(4.0)

    result = agg.fedavg([model_a.state_dict(), model_b.state_dict()])
    for key, val in result.items():
        assert torch.allclose(val, torch.full_like(val, 2.0), atol=1e-6), (
            f"Expected 2.0 for key {key}, got {val}"
        )


# ---------------------------------------------------------------------------
# 15. aggregate_lora returns dict with A and B keys
# ---------------------------------------------------------------------------


def test_aggregate_lora_keys():
    agg = FedAggregator(n_clients=3)
    adapters = [FedLoRAAdapter(d_in=D_IN, d_out=D_OUT, rank=RANK) for _ in range(3)]
    params_list = [a.get_adapter_params() for a in adapters]
    result = agg.aggregate_lora(params_list)
    assert "A" in result and "B" in result
    assert result["A"].shape == (RANK, D_IN)
    assert result["B"].shape == (D_OUT, RANK)
