"""Tests for src/eval/probing_suite.py — 15 tests covering all public APIs."""

import pytest
import torch

from src.eval.probing_suite import (
    LayerwiseProber,
    LayerwiseProbingResults,
    MLPProbe,
    ProbingClassifier,
    ProbingConfig,
    ProbingDataset,
    extract_layer_representations,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def small_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def input_ids():
    """Short sequence: B=4, T=4."""
    return torch.randint(0, 256, (4, 4))


@pytest.fixture
def binary_labels():
    return torch.tensor([0, 1, 0, 1], dtype=torch.long)


# ---------------------------------------------------------------------------
# Test 1: ProbingConfig defaults
# ---------------------------------------------------------------------------


def test_probing_config_defaults():
    cfg = ProbingConfig()
    assert cfg.probe_type == "linear"
    assert cfg.n_epochs == 10
    assert cfg.lr == 1e-3
    assert cfg.hidden_dim == 128
    assert cfg.dropout == 0.1


# ---------------------------------------------------------------------------
# Test 2: MLPProbe output shape (B, n_classes)
# ---------------------------------------------------------------------------


def test_mlp_probe_output_shape():
    probe = MLPProbe(input_dim=64, n_classes=3, hidden_dim=32, dropout=0.0)
    x = torch.randn(8, 64)
    out = probe(x)
    assert out.shape == (8, 3), f"Expected (8, 3), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 3: extract_layer_representations returns dict with correct keys
# ---------------------------------------------------------------------------


def test_extract_layer_representations_keys(small_model, input_ids):
    layer_indices = [0, 1]
    result = extract_layer_representations(small_model, input_ids, layer_indices)
    assert set(result.keys()) == {0, 1}


# ---------------------------------------------------------------------------
# Test 4: extract_layer_representations shape (B, T, D) per layer
# ---------------------------------------------------------------------------


def test_extract_layer_representations_shape(small_model, input_ids):
    result = extract_layer_representations(small_model, input_ids, [0])
    hidden = result[0]
    B, T = input_ids.shape
    D = 64  # d_model from small_cfg
    assert hidden.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {hidden.shape}"


# ---------------------------------------------------------------------------
# Test 5: ProbingDataset.__len__ returns N
# ---------------------------------------------------------------------------


def test_probing_dataset_len():
    reps = torch.randn(20, 32)
    labels = torch.randint(0, 2, (20,))
    ds = ProbingDataset(reps, labels)
    assert len(ds) == 20


# ---------------------------------------------------------------------------
# Test 6: ProbingDataset.split correct sizes
# ---------------------------------------------------------------------------


def test_probing_dataset_split_sizes():
    N = 10
    reps = torch.randn(N, 16)
    labels = torch.randint(0, 2, (N,))
    ds = ProbingDataset(reps, labels)
    train_ds, val_ds = ds.split(ratio=0.8)
    assert len(train_ds) == 8
    assert len(val_ds) == 2


# ---------------------------------------------------------------------------
# Test 7: ProbingClassifier.fit returns list of floats
# ---------------------------------------------------------------------------


def test_probing_classifier_fit_returns_loss_history():
    cfg = ProbingConfig(n_epochs=5, probe_type="linear")
    reps = torch.randn(16, 32)
    labels = torch.randint(0, 2, (16,))
    ds = ProbingDataset(reps, labels)
    clf = ProbingClassifier(config=cfg, input_dim=32, n_classes=2)
    history = clf.fit(ds)
    assert isinstance(history, list)
    assert len(history) == 5
    assert all(isinstance(v, float) for v in history)


# ---------------------------------------------------------------------------
# Test 8: ProbingClassifier.evaluate returns required keys
# ---------------------------------------------------------------------------


def test_probing_classifier_evaluate_keys():
    cfg = ProbingConfig(n_epochs=3)
    reps = torch.randn(12, 32)
    labels = torch.randint(0, 2, (12,))
    ds = ProbingDataset(reps, labels)
    clf = ProbingClassifier(config=cfg, input_dim=32, n_classes=2)
    clf.fit(ds)
    result = clf.evaluate(ds)
    assert "accuracy" in result
    assert "loss" in result


# ---------------------------------------------------------------------------
# Test 9: ProbingClassifier.evaluate accuracy in [0, 1]
# ---------------------------------------------------------------------------


def test_probing_classifier_evaluate_accuracy_range():
    cfg = ProbingConfig(n_epochs=3)
    reps = torch.randn(12, 32)
    labels = torch.randint(0, 3, (12,))
    ds = ProbingDataset(reps, labels)
    clf = ProbingClassifier(config=cfg, input_dim=32, n_classes=3)
    clf.fit(ds)
    result = clf.evaluate(ds)
    assert 0.0 <= result["accuracy"] <= 1.0, f"accuracy {result['accuracy']} out of [0,1]"


# ---------------------------------------------------------------------------
# Test 10: ProbingClassifier.predict returns correct shape
# ---------------------------------------------------------------------------


def test_probing_classifier_predict_shape():
    cfg = ProbingConfig(n_epochs=2)
    reps = torch.randn(10, 32)
    labels = torch.randint(0, 2, (10,))
    ds = ProbingDataset(reps, labels)
    clf = ProbingClassifier(config=cfg, input_dim=32, n_classes=2)
    clf.fit(ds)
    x = torch.randn(6, 32)
    preds = clf.predict(x)
    assert preds.shape == (6,), f"Expected (6,), got {preds.shape}"


# ---------------------------------------------------------------------------
# Test 11: LayerwiseProber.probe_all_layers returns LayerwiseProbingResults
# ---------------------------------------------------------------------------


def test_layerwise_prober_returns_results(small_model, input_ids, binary_labels):
    cfg = ProbingConfig(n_epochs=2)
    prober = LayerwiseProber(small_model, cfg)
    results = prober.probe_all_layers(input_ids, binary_labels, task_name="test_task")
    assert isinstance(results, LayerwiseProbingResults)
    assert results.task_name == "test_task"
    assert isinstance(results.layer_accuracies, dict)


# ---------------------------------------------------------------------------
# Test 12: LayerwiseProbingResults.best_layer is valid layer index
# ---------------------------------------------------------------------------


def test_layerwise_results_best_layer_valid(small_model, input_ids, binary_labels):
    cfg = ProbingConfig(n_epochs=2)
    prober = LayerwiseProber(small_model, cfg)
    results = prober.probe_all_layers(input_ids, binary_labels, task_name="validity")
    # n_layers=2 means valid layer indices are 0 and 1
    assert results.best_layer in results.layer_accuracies
    assert results.best_layer in {0, 1}


# ---------------------------------------------------------------------------
# Test 13: LayerwiseProber.mutual_information_estimate returns non-negative float
# ---------------------------------------------------------------------------


def test_mutual_information_estimate_non_negative(small_model):
    cfg = ProbingConfig()
    prober = LayerwiseProber(small_model, cfg)
    reps = torch.randn(50, 64)
    labels = torch.randint(0, 2, (50,))
    mi = prober.mutual_information_estimate(reps, labels)
    assert isinstance(mi, float)
    assert mi >= 0.0, f"MI should be non-negative, got {mi}"


# ---------------------------------------------------------------------------
# Test 14: LayerwiseProber.rank_layers returns sorted list (descending)
# ---------------------------------------------------------------------------


def test_rank_layers_sorted_descending(small_model, input_ids, binary_labels):
    cfg = ProbingConfig(n_epochs=2)
    prober = LayerwiseProber(small_model, cfg)
    results = prober.probe_all_layers(input_ids, binary_labels, task_name="rank_test")
    ranked = prober.rank_layers(results)
    assert isinstance(ranked, list)
    assert len(ranked) == len(results.layer_accuracies)
    # Check descending order
    accuracies = [acc for _, acc in ranked]
    assert accuracies == sorted(accuracies, reverse=True), (
        f"Layers not sorted by descending accuracy: {ranked}"
    )


# ---------------------------------------------------------------------------
# Test 15: ProbingDataset.split sizes sum to original
# ---------------------------------------------------------------------------


def test_probing_dataset_split_sizes_sum_to_original():
    N = 25
    reps = torch.randn(N, 16)
    labels = torch.randint(0, 3, (N,))
    ds = ProbingDataset(reps, labels)
    train_ds, val_ds = ds.split(ratio=0.7)
    assert len(train_ds) + len(val_ds) == N, f"Train ({len(train_ds)}) + Val ({len(val_ds)}) != {N}"
