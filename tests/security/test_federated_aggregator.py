"""Tests for FederatedAggregator."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.security.federated_aggregator import ClientUpdate, FederatedAggregator


@pytest.fixture
def model():
    torch.manual_seed(0)
    return nn.Linear(8, 4)


@pytest.fixture
def aggregator(model):
    return FederatedAggregator(model, n_clients=3, noise_sigma=0.0)


def _make_updates(model: nn.Module, n_clients: int = 3, perturb: bool = False):
    """Create client updates, optionally with random perturbations."""
    updates = []
    for i in range(n_clients):
        state = copy.deepcopy(model.state_dict())
        if perturb:
            with torch.no_grad():
                for key in state:
                    state[key] = (
                        state[key]
                        + torch.randn_like(state[key].float()).to(state[key].dtype) * (i + 1) * 0.1
                    )
        updates.append(ClientUpdate(client_id=i, state_dict=state, n_samples=10 * (i + 1)))
    return updates


# 1. FederatedAggregator instantiates
def test_instantiation(model):
    agg = FederatedAggregator(model, n_clients=3, noise_sigma=0.01)
    assert agg.model is model
    assert agg.n_clients == 3
    assert agg.noise_sigma == 0.01


# 2. _weighted_average returns dict with same keys as model
def test_weighted_average_keys(aggregator, model):
    updates = _make_updates(model)
    result = aggregator._weighted_average(updates)
    assert set(result.keys()) == set(model.state_dict().keys())


# 3. Weighted average of identical updates equals that update
def test_weighted_average_identical_updates(aggregator, model):
    updates = _make_updates(model, perturb=False)
    result = aggregator._weighted_average(updates)
    expected = model.state_dict()
    for key in expected:
        assert torch.allclose(result[key].float(), expected[key].float(), atol=1e-5), (
            f"Mismatch on key {key}"
        )


# 4. aggregate returns a state dict (dict with tensor values)
def test_aggregate_returns_state_dict(aggregator, model):
    updates = _make_updates(model)
    result = aggregator.aggregate(updates)
    assert isinstance(result, dict)
    assert all(isinstance(v, torch.Tensor) for v in result.values())


# 5. All tensors in aggregated dict have correct shapes
def test_aggregate_correct_shapes(aggregator, model):
    updates = _make_updates(model, perturb=True)
    result = aggregator.aggregate(updates)
    expected_state = model.state_dict()
    for key in expected_state:
        assert result[key].shape == expected_state[key].shape, (
            f"Shape mismatch on key {key}: {result[key].shape} vs {expected_state[key].shape}"
        )


# 6. Aggregated update differs from any single client when updates differ
def test_aggregate_differs_from_single_client(aggregator, model):
    updates = _make_updates(model, perturb=True)
    result = aggregator.aggregate(updates)
    for update in updates:
        all_equal = all(
            torch.allclose(result[k].float(), update.state_dict[k].float(), atol=1e-5)
            for k in result
        )
        assert not all_equal, f"Aggregated result should differ from client {update.client_id}"


# 7. apply changes model parameters
def test_apply_changes_model(aggregator, model):
    original_params = {k: v.clone() for k, v in model.state_dict().items()}
    new_state = copy.deepcopy(model.state_dict())
    for key in new_state:
        new_state[key] = new_state[key] + 1.0
    aggregator.apply(new_state)
    updated_params = model.state_dict()
    for key in original_params:
        assert not torch.allclose(updated_params[key].float(), original_params[key].float()), (
            f"Parameter {key} was not updated"
        )


# 8. compute_update_norm returns float >= 0
def test_compute_update_norm_returns_nonneg_float(aggregator, model):
    updates = _make_updates(model, perturb=True)
    global_state = model.state_dict()
    norm = aggregator.compute_update_norm(updates[0], global_state)
    assert isinstance(norm, float)
    assert norm >= 0.0


# 9. Norm is 0 when client update equals global model
def test_compute_update_norm_zero_when_equal(aggregator, model):
    global_state = model.state_dict()
    identical_update = ClientUpdate(
        client_id=0,
        state_dict=copy.deepcopy(global_state),
        n_samples=10,
    )
    norm = aggregator.compute_update_norm(identical_update, global_state)
    assert norm == pytest.approx(0.0, abs=1e-6)


# 10. clip_update returns a ClientUpdate
def test_clip_update_returns_client_update(aggregator, model):
    updates = _make_updates(model, perturb=True)
    clipped = aggregator.clip_update(updates[0], max_norm=0.1)
    assert isinstance(clipped, ClientUpdate)


# 11. Clipped update norm <= max_norm (with tolerance 1e-5)
def test_clip_update_norm_bounded(aggregator, model):
    updates = _make_updates(model, perturb=True)
    max_norm = 0.05
    global_state = model.state_dict()
    for update in updates:
        clipped = aggregator.clip_update(update, max_norm=max_norm)
        norm = aggregator.compute_update_norm(clipped, global_state)
        assert norm <= max_norm + 1e-5, f"Norm {norm} exceeds max_norm {max_norm}"


# 12. _add_aggregation_noise with sigma=0 returns identical values
def test_add_noise_sigma_zero_unchanged(aggregator, model):
    state = copy.deepcopy(model.state_dict())
    result = aggregator._add_aggregation_noise(state, sigma=0.0)
    for key in state:
        assert torch.allclose(result[key].float(), state[key].float(), atol=1e-7), (
            f"Values changed for key {key} with sigma=0"
        )


# 13. _add_aggregation_noise with sigma > 0 changes values
def test_add_noise_sigma_positive_changes_values(aggregator, model):
    torch.manual_seed(42)
    state = copy.deepcopy(model.state_dict())
    result = aggregator._add_aggregation_noise(state, sigma=1.0)
    any_changed = any(not torch.allclose(result[k].float(), state[k].float()) for k in state)
    assert any_changed, "Noise with sigma=1.0 should change at least some tensor values"
