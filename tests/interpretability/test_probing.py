"""
Tests for src/interpretability/probing.py

Uses tiny configs (D=16, N_CLASSES=3, N=40, N_EPOCHS=5, B_SIZE=20) so the
suite runs quickly without GPU.
"""

from __future__ import annotations

import math
import pytest
import torch

from src.interpretability.probing import (
    ProbingConfig,
    LinearProbe,
    train_probe,
    evaluate_probe,
    compute_mutual_information_proxy,
    LayerProbingResults,
    MultiLayerProber,
)

# ---------------------------------------------------------------------------
# Shared tiny constants
# ---------------------------------------------------------------------------

D = 16
N_CLASSES = 3
N = 40
N_EPOCHS = 5
B_SIZE = 20

torch.manual_seed(0)


def _make_random_data():
    """Random hiddens + random labels."""
    hiddens = torch.randn(N, D)
    labels = torch.randint(0, N_CLASSES, (N,))
    return hiddens, labels


def _make_separable_data():
    """
    Linearly-separable hiddens: each class cluster is far from others along
    the first dimension, so MI proxy > 0 is guaranteed.
    """
    hiddens = torch.randn(N, D) * 0.1
    labels = torch.arange(N) % N_CLASSES
    for cls in range(N_CLASSES):
        mask = labels == cls
        hiddens[mask, 0] += cls * 10.0  # class-discriminative first dim
    return hiddens, labels


def _tiny_config() -> ProbingConfig:
    return ProbingConfig(
        n_classes=N_CLASSES,
        d_hidden=D,
        n_epochs=N_EPOCHS,
        lr=1e-2,
        l2_reg=1e-4,
        batch_size=B_SIZE,
    )


# ---------------------------------------------------------------------------
# 1. ProbingConfig defaults
# ---------------------------------------------------------------------------

def test_probing_config_defaults():
    cfg = ProbingConfig()
    assert cfg.n_classes == 2
    assert cfg.d_hidden == 512
    assert cfg.n_epochs == 100
    assert math.isclose(cfg.lr, 1e-3)
    assert math.isclose(cfg.l2_reg, 1e-4)
    assert cfg.batch_size == 64


# ---------------------------------------------------------------------------
# 2. LinearProbe output shape
# ---------------------------------------------------------------------------

def test_linear_probe_output_shape():
    probe = LinearProbe(D, N_CLASSES)
    h = torch.randn(N, D)
    out = probe(h)
    assert out.shape == (N, N_CLASSES), f"Expected ({N}, {N_CLASSES}), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. train_probe returns list of length n_epochs
# ---------------------------------------------------------------------------

def test_train_probe_returns_correct_length():
    cfg = _tiny_config()
    hiddens, labels = _make_random_data()
    probe = LinearProbe(D, N_CLASSES)
    losses = train_probe(probe, hiddens, labels, cfg)
    assert isinstance(losses, list)
    assert len(losses) == N_EPOCHS


# ---------------------------------------------------------------------------
# 4. train_probe loss decreases (first epoch loss > last epoch loss)
# ---------------------------------------------------------------------------

def test_train_probe_loss_decreases():
    cfg = ProbingConfig(
        n_classes=N_CLASSES,
        d_hidden=D,
        n_epochs=50,
        lr=5e-2,
        l2_reg=0.0,
        batch_size=N,
    )
    hiddens, labels = _make_separable_data()
    probe = LinearProbe(D, N_CLASSES)
    losses = train_probe(probe, hiddens, labels, cfg)
    assert losses[0] > losses[-1], (
        f"Expected loss to decrease, first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. evaluate_probe returns accuracy and loss keys
# ---------------------------------------------------------------------------

def test_evaluate_probe_keys():
    cfg = _tiny_config()
    hiddens, labels = _make_random_data()
    probe = LinearProbe(D, N_CLASSES)
    train_probe(probe, hiddens, labels, cfg)
    metrics = evaluate_probe(probe, hiddens, labels)
    assert "accuracy" in metrics
    assert "loss" in metrics


# ---------------------------------------------------------------------------
# 6. accuracy is in [0, 1]
# ---------------------------------------------------------------------------

def test_evaluate_probe_accuracy_range():
    cfg = _tiny_config()
    hiddens, labels = _make_random_data()
    probe = LinearProbe(D, N_CLASSES)
    train_probe(probe, hiddens, labels, cfg)
    metrics = evaluate_probe(probe, hiddens, labels)
    acc = metrics["accuracy"]
    assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"


# ---------------------------------------------------------------------------
# 7. loss is positive
# ---------------------------------------------------------------------------

def test_evaluate_probe_loss_positive():
    cfg = _tiny_config()
    hiddens, labels = _make_random_data()
    probe = LinearProbe(D, N_CLASSES)
    train_probe(probe, hiddens, labels, cfg)
    metrics = evaluate_probe(probe, hiddens, labels)
    assert metrics["loss"] > 0.0, f"Expected positive loss, got {metrics['loss']}"


# ---------------------------------------------------------------------------
# 8. compute_mutual_information_proxy returns non-negative float
# ---------------------------------------------------------------------------

def test_mi_proxy_non_negative():
    hiddens, labels = _make_random_data()
    mi = compute_mutual_information_proxy(hiddens, labels)
    assert isinstance(mi, float)
    assert mi >= 0.0, f"MI should be non-negative, got {mi}"


# ---------------------------------------------------------------------------
# 9. MI > 0 for linearly separable data
# ---------------------------------------------------------------------------

def test_mi_proxy_positive_for_separable_data():
    hiddens, labels = _make_separable_data()
    mi = compute_mutual_information_proxy(hiddens, labels, n_bins=10)
    assert mi > 0.0, f"Expected MI > 0 for separable data, got {mi}"


# ---------------------------------------------------------------------------
# 10. MultiLayerProber.probe_layer returns LayerProbingResults
# ---------------------------------------------------------------------------

def test_probe_layer_returns_correct_type():
    cfg = _tiny_config()
    prober = MultiLayerProber(cfg)
    hiddens, labels = _make_random_data()
    result = prober.probe_layer(hiddens, labels, layer_idx=3)
    assert isinstance(result, LayerProbingResults)


# ---------------------------------------------------------------------------
# 11. probe_all_layers length equals number of layers
# ---------------------------------------------------------------------------

def test_probe_all_layers_length():
    cfg = _tiny_config()
    prober = MultiLayerProber(cfg)
    n_layers = 4
    all_hiddens = [torch.randn(N, D) for _ in range(n_layers)]
    labels = torch.randint(0, N_CLASSES, (N,))
    results = prober.probe_all_layers(all_hiddens, labels)
    assert len(results) == n_layers


# ---------------------------------------------------------------------------
# 12. get_best_layer returns a valid layer index
# ---------------------------------------------------------------------------

def test_get_best_layer_valid_index():
    cfg = _tiny_config()
    prober = MultiLayerProber(cfg)
    n_layers = 3
    all_hiddens = [torch.randn(N, D) for _ in range(n_layers)]
    labels = torch.randint(0, N_CLASSES, (N,))
    results = prober.probe_all_layers(all_hiddens, labels)
    best = prober.get_best_layer(results)
    valid_indices = {r.layer_idx for r in results}
    assert best in valid_indices, f"best={best} not in {valid_indices}"


# ---------------------------------------------------------------------------
# 13. LayerProbingResults has required fields
# ---------------------------------------------------------------------------

def test_layer_probing_results_fields():
    result = LayerProbingResults(
        layer_idx=2,
        accuracy=0.75,
        loss=0.5,
        train_losses=[1.0, 0.8, 0.6],
    )
    assert result.layer_idx == 2
    assert result.accuracy == 0.75
    assert result.loss == 0.5
    assert result.train_losses == [1.0, 0.8, 0.6]


# ---------------------------------------------------------------------------
# 14. probe_all_layers results are sorted by layer_idx
# ---------------------------------------------------------------------------

def test_probe_all_layers_sorted():
    cfg = _tiny_config()
    prober = MultiLayerProber(cfg)
    n_layers = 5
    all_hiddens = [torch.randn(N, D) for _ in range(n_layers)]
    labels = torch.randint(0, N_CLASSES, (N,))
    results = prober.probe_all_layers(all_hiddens, labels)
    indices = [r.layer_idx for r in results]
    assert indices == sorted(indices), f"Results not sorted: {indices}"


# ---------------------------------------------------------------------------
# 15. train_probe losses are all finite floats
# ---------------------------------------------------------------------------

def test_train_probe_losses_finite():
    cfg = _tiny_config()
    hiddens, labels = _make_random_data()
    probe = LinearProbe(D, N_CLASSES)
    losses = train_probe(probe, hiddens, labels, cfg)
    for i, loss in enumerate(losses):
        assert math.isfinite(loss), f"Loss at epoch {i} is not finite: {loss}"
