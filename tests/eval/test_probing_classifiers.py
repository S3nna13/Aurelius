"""Tests for src/eval/probing_classifiers.py."""

from __future__ import annotations

import torch
from aurelius.eval.probing_classifiers import (
    LinearProbe,
    MLPProbe,
    ProbingEvaluator,
    ProbingResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _separable_data(n: int = 200, d: int = 32, n_classes: int = 2):
    """Generate linearly separable features and labels."""
    torch.manual_seed(0)
    features = torch.randn(n, d)
    labels = torch.zeros(n, dtype=torch.long)
    half = n // n_classes
    for c in range(n_classes):
        start = c * half
        end = start + half
        features[start:end] += c * 5.0  # large class separation
        labels[start:end] = c
    return features, labels


def _random_labels(n: int = 200, d: int = 32, n_classes: int = 2):
    """Generate random features with random labels."""
    torch.manual_seed(1)
    features = torch.randn(n, d)
    labels = torch.randint(0, n_classes, (n,))
    return features, labels


# ---------------------------------------------------------------------------
# LinearProbe tests
# ---------------------------------------------------------------------------


def test_linear_probe_output_shape_2d():
    """LinearProbe on (B, d) input returns (B, n_classes)."""
    probe = LinearProbe(d_model=32, n_classes=5)
    x = torch.randn(8, 32)
    out = probe(x)
    assert out.shape == (8, 5)


def test_linear_probe_output_shape_3d_mean_pool():
    """LinearProbe on (B, T, d) input mean-pools T and returns (B, n_classes)."""
    probe = LinearProbe(d_model=32, n_classes=4)
    x = torch.randn(8, 10, 32)
    out = probe(x)
    assert out.shape == (8, 4)


def test_linear_probe_fit_returns_self():
    """fit() returns the probe instance itself."""
    probe = LinearProbe(d_model=16, n_classes=2)
    features = torch.randn(50, 16)
    labels = torch.randint(0, 2, (50,))
    result = probe.fit(features, labels, n_epochs=5)
    assert result is probe


def test_linear_probe_accuracy_in_range():
    """accuracy() returns a float in [0, 1]."""
    probe = LinearProbe(d_model=16, n_classes=2)
    features = torch.randn(50, 16)
    labels = torch.randint(0, 2, (50,))
    probe.fit(features, labels, n_epochs=10)
    acc = probe.accuracy(features, labels)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_linear_probe_separable_accuracy():
    """LinearProbe achieves >0.9 accuracy on linearly separable data."""
    features, labels = _separable_data(n=200, d=32, n_classes=2)
    probe = LinearProbe(d_model=32, n_classes=2)
    probe.fit(features, labels, n_epochs=200, lr=0.05)
    acc = probe.accuracy(features, labels)
    assert acc > 0.9, f"Expected >0.9, got {acc:.4f}"


def test_linear_probe_random_labels_chance():
    """LinearProbe trained on random labels should not exceed chance on held-out data."""
    torch.manual_seed(42)
    n_classes = 2
    # Train on first 160, evaluate on held-out 40
    features, labels = _random_labels(n=200, d=32, n_classes=n_classes)
    train_feat, train_lab = features[:160], labels[:160]
    val_feat, val_lab = features[160:], labels[160:]
    probe = LinearProbe(d_model=32, n_classes=n_classes)
    probe.fit(train_feat, train_lab, n_epochs=100)
    acc = probe.accuracy(val_feat, val_lab)
    chance = 1.0 / n_classes
    assert acc < chance + 0.2, f"Expected near-chance val accuracy, got {acc:.4f}"


# ---------------------------------------------------------------------------
# MLPProbe tests
# ---------------------------------------------------------------------------


def test_mlp_probe_output_shape():
    """MLPProbe on (B, d) input returns (B, n_classes)."""
    probe = MLPProbe(d_model=32, hidden_dim=64, n_classes=3)
    x = torch.randn(8, 32)
    out = probe(x)
    assert out.shape == (8, 3)


def test_mlp_probe_separable_accuracy():
    """MLPProbe achieves >0.9 accuracy on linearly separable data."""
    features, labels = _separable_data(n=200, d=32, n_classes=2)
    probe = MLPProbe(d_model=32, hidden_dim=64, n_classes=2)
    probe.fit(features, labels, n_epochs=200, lr=0.05)
    acc = probe.accuracy(features, labels)
    assert acc > 0.9, f"Expected >0.9, got {acc:.4f}"


def test_mlp_probe_with_dropout():
    """MLPProbe with dropout=0.5 runs without error and returns valid shape."""
    probe = MLPProbe(d_model=32, hidden_dim=64, n_classes=2, dropout=0.5)
    x = torch.randn(16, 32)
    labels = torch.randint(0, 2, (16,))
    probe.fit(x, labels, n_epochs=10)
    out = probe(x)
    assert out.shape == (16, 2)
    acc = probe.accuracy(x, labels)
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# ProbingEvaluator tests
# ---------------------------------------------------------------------------


def test_probing_evaluator_returns_correct_keys():
    """evaluate_layer returns dict with 'train_acc', 'val_acc', 'n_params'."""
    features, labels = _separable_data(n=100, d=16, n_classes=2)
    evaluator = ProbingEvaluator(probe_cls=LinearProbe, n_classes=2)
    result = evaluator.evaluate_layer(features, labels, n_epochs=10)
    assert set(result.keys()) == {"train_acc", "val_acc", "n_params"}


def test_probing_evaluator_val_acc_in_range():
    """evaluate_layer val_acc is in [0, 1]."""
    features, labels = _separable_data(n=100, d=16, n_classes=2)
    evaluator = ProbingEvaluator(probe_cls=LinearProbe, n_classes=2)
    result = evaluator.evaluate_layer(features, labels, n_epochs=10)
    assert 0.0 <= result["val_acc"] <= 1.0


def test_probing_evaluator_n_params_positive():
    """evaluate_layer n_params is > 0."""
    features, labels = _separable_data(n=100, d=16, n_classes=2)
    evaluator = ProbingEvaluator(probe_cls=LinearProbe, n_classes=2)
    result = evaluator.evaluate_layer(features, labels, n_epochs=10)
    assert result["n_params"] > 0


def test_probing_evaluator_all_layers_length():
    """evaluate_all_layers returns one result per layer."""
    torch.manual_seed(7)
    n_layers = 4
    layer_features = [torch.randn(100, 16) for _ in range(n_layers)]
    labels = torch.randint(0, 2, (100,))
    evaluator = ProbingEvaluator(probe_cls=LinearProbe, n_classes=2)
    results = evaluator.evaluate_all_layers(layer_features, labels)
    assert len(results) == n_layers


# ---------------------------------------------------------------------------
# ProbingResult tests
# ---------------------------------------------------------------------------


def test_probing_result_is_significant_threshold_logic():
    """is_significant uses val_acc > chance_level + threshold."""
    # Exactly at boundary → not significant
    r = ProbingResult(layer_idx=0, train_acc=0.9, val_acc=0.6, n_params=100)
    # 0.6 > 0.5 + 0.1 is False (equal, not greater)
    assert r.is_significant(chance_level=0.5, threshold=0.1) is False


def test_probing_result_significant_high_val_acc():
    """val_acc=0.95 with chance=0.5, threshold=0.1 is significant."""
    r = ProbingResult(layer_idx=2, train_acc=0.98, val_acc=0.95, n_params=500)
    assert r.is_significant(chance_level=0.5, threshold=0.1) is True


def test_probing_result_not_significant_low_val_acc():
    """val_acc=0.52 with chance=0.5, threshold=0.1 is NOT significant."""
    r = ProbingResult(layer_idx=1, train_acc=0.55, val_acc=0.52, n_params=100)
    assert r.is_significant(chance_level=0.5, threshold=0.1) is False
