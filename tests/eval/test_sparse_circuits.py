"""Tests for sparse_circuits.py — Sparse Feature Circuits."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.eval.sparse_circuits import (
    Circuit,
    CircuitFinder,
    FeatureAnalyzer,
    SparseAutoencoder,
    SparseFeature,
    compute_feature_correlation,
    train_sae_step,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INPUT_DIM = 16
N_FEATURES = 64
B = 8
N = 32


@pytest.fixture
def sae():
    return SparseAutoencoder(input_dim=INPUT_DIM, n_features=N_FEATURES, l1_coef=1e-3)


@pytest.fixture
def activations():
    torch.manual_seed(42)
    return torch.randn(B, INPUT_DIM)


@pytest.fixture
def large_activations():
    torch.manual_seed(42)
    return torch.randn(N, INPUT_DIM)


@pytest.fixture
def dummy_model():
    return nn.Linear(INPUT_DIM, INPUT_DIM)


@pytest.fixture
def circuit_finder(sae, dummy_model):
    return CircuitFinder(sae=sae, model=dummy_model, layer_name="hidden")


@pytest.fixture
def feature_analyzer(sae):
    return FeatureAnalyzer(sae=sae)


# ---------------------------------------------------------------------------
# Test 1: SparseAutoencoder forward returns tuple of 3
# ---------------------------------------------------------------------------


def test_sae_forward_returns_three_tensors(sae, activations):
    result = sae(activations)
    assert isinstance(result, tuple)
    assert len(result) == 3, "forward() must return (reconstruction, features, loss)"


# ---------------------------------------------------------------------------
# Test 2: Reconstruction shape == input shape
# ---------------------------------------------------------------------------


def test_sae_reconstruction_shape(sae, activations):
    reconstruction, features, loss = sae(activations)
    assert reconstruction.shape == activations.shape, (
        f"Reconstruction shape {reconstruction.shape} != input shape {activations.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: Features shape == (B, n_features)
# ---------------------------------------------------------------------------


def test_sae_features_shape(sae, activations):
    reconstruction, features, loss = sae(activations)
    assert features.shape == (B, N_FEATURES), (
        f"Features shape {features.shape} != expected ({B}, {N_FEATURES})"
    )


# ---------------------------------------------------------------------------
# Test 4: Features are non-negative (ReLU)
# ---------------------------------------------------------------------------


def test_sae_features_nonnegative(sae, activations):
    reconstruction, features, loss = sae(activations)
    assert (features >= 0).all(), "All feature activations must be non-negative (ReLU output)"


# ---------------------------------------------------------------------------
# Test 5: Features are sparse (mean active per sample < 0.5 * n_features)
# ---------------------------------------------------------------------------


def test_sae_features_sparse(sae):
    torch.manual_seed(0)
    # Use random activations with a reasonably initialized SAE
    x = torch.randn(B, INPUT_DIM)
    _, features, _ = sae(x)
    mean_active = (features > 0).float().sum(dim=1).mean().item()
    assert mean_active < 0.5 * N_FEATURES, (
        f"Mean active features {mean_active:.1f} is not sparse (threshold: {0.5 * N_FEATURES})"
    )


# ---------------------------------------------------------------------------
# Test 6: train_sae_step returns dict with 'loss' key
# ---------------------------------------------------------------------------


def test_train_sae_step_returns_dict_with_loss(sae, activations):
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)
    result = train_sae_step(sae, activations, optimizer)
    assert isinstance(result, dict), "train_sae_step must return a dict"
    assert "loss" in result, "Result dict must contain 'loss' key"


# ---------------------------------------------------------------------------
# Test 7: train_sae_step reduces loss over 10 steps
# ---------------------------------------------------------------------------


def test_train_sae_step_reduces_loss(sae):
    torch.manual_seed(123)
    x = torch.randn(B, INPUT_DIM)
    optimizer = optim.Adam(sae.parameters(), lr=1e-2)

    first_loss = train_sae_step(sae, x, optimizer)["loss"]
    for _ in range(9):
        last_loss = train_sae_step(sae, x, optimizer)["loss"]

    assert last_loss < first_loss, (
        f"Loss did not decrease over 10 steps: {first_loss:.4f} -> {last_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 8: FeatureAnalyzer.analyze_batch returns list of SparseFeature
# ---------------------------------------------------------------------------


def test_analyze_batch_returns_list_of_sparse_features(feature_analyzer, large_activations):
    result = feature_analyzer.analyze_batch(large_activations)
    assert isinstance(result, list), "analyze_batch must return a list"
    assert len(result) > 0, "Should find at least some active features"
    assert all(isinstance(f, SparseFeature) for f in result), (
        "All items must be SparseFeature instances"
    )


# ---------------------------------------------------------------------------
# Test 9: SparseFeature.activation_freq in [0, 1]
# ---------------------------------------------------------------------------


def test_sparse_feature_activation_freq_in_range(feature_analyzer, large_activations):
    features = feature_analyzer.analyze_batch(large_activations)
    for f in features:
        assert 0.0 <= f.activation_freq <= 1.0, (
            f"activation_freq {f.activation_freq} out of [0, 1] for feature {f.feature_id}"
        )


# ---------------------------------------------------------------------------
# Test 10: FeatureAnalyzer.find_top_features returns list of (int, float) tuples
# ---------------------------------------------------------------------------


def test_find_top_features_returns_tuples(feature_analyzer, activations):
    result = feature_analyzer.find_top_features(activations, top_k=10)
    assert isinstance(result, list), "find_top_features must return a list"
    for item in result:
        assert isinstance(item, tuple) and len(item) == 2, "Each item must be a 2-tuple"
        fid, val = item
        assert isinstance(fid, int), f"Feature id must be int, got {type(fid)}"
        assert isinstance(val, float), f"Activation value must be float, got {type(val)}"


# ---------------------------------------------------------------------------
# Test 11: find_top_features length <= top_k
# ---------------------------------------------------------------------------


def test_find_top_features_length(feature_analyzer, activations):
    top_k = 10
    result = feature_analyzer.find_top_features(activations, top_k=top_k)
    assert len(result) <= top_k, (
        f"find_top_features returned {len(result)} items, expected <= {top_k}"
    )


# ---------------------------------------------------------------------------
# Test 12: CircuitFinder.find_circuit_greedy returns Circuit
# ---------------------------------------------------------------------------


def test_find_circuit_greedy_returns_circuit(circuit_finder, activations):
    result = circuit_finder.find_circuit_greedy(activations, max_features=5)
    assert isinstance(result, Circuit), (
        f"find_circuit_greedy must return a Circuit, got {type(result)}"
    )


# ---------------------------------------------------------------------------
# Test 13: Circuit.faithfulness_score in [0, 1]
# ---------------------------------------------------------------------------


def test_circuit_faithfulness_score_in_range(circuit_finder, activations):
    circuit = circuit_finder.find_circuit_greedy(activations, max_features=5)
    assert 0.0 <= circuit.faithfulness_score <= 1.0, (
        f"faithfulness_score {circuit.faithfulness_score} out of [0, 1]"
    )


# ---------------------------------------------------------------------------
# Test 14: CircuitFinder.ablate_features returns tensor of same shape as input
# ---------------------------------------------------------------------------


def test_ablate_features_shape(circuit_finder, activations):
    # Use a few arbitrary feature ids
    feature_ids = [0, 1, 2]
    result = circuit_finder.ablate_features(activations, feature_ids, mode="zero")
    assert isinstance(result, torch.Tensor), "ablate_features must return a Tensor"
    assert result.shape == activations.shape, (
        f"Ablated tensor shape {result.shape} != input shape {activations.shape}"
    )


# ---------------------------------------------------------------------------
# Test 15: compute_feature_correlation returns dict
# ---------------------------------------------------------------------------


def test_compute_feature_correlation_returns_dict():
    torch.manual_seed(0)
    feat_acts = torch.rand(N, N_FEATURES)
    result = compute_feature_correlation(feat_acts, top_k=5)
    assert isinstance(result, dict), "compute_feature_correlation must return a dict"
    assert len(result) == N_FEATURES, f"Dict should have {N_FEATURES} keys, got {len(result)}"


# ---------------------------------------------------------------------------
# Test 16: Feature correlation values in [-1, 1]
# ---------------------------------------------------------------------------


def test_feature_correlation_values_in_range():
    torch.manual_seed(1)
    feat_acts = torch.rand(N, N_FEATURES)
    result = compute_feature_correlation(feat_acts, top_k=5)
    for fid, pairs in result.items():
        for other_fid, corr in pairs:
            assert -1.0 <= corr <= 1.0, (
                f"Correlation {corr:.4f} for feature {fid}->{other_fid} out of [-1, 1]"
            )
