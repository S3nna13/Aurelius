"""Tests for the safety classifier."""
from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.alignment.safety_classifier import (
    SafetyClassifier,
    SafetyConfig,
    SafetyTrainer,
    safety_filter,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def backbone(small_cfg):
    torch.manual_seed(42)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def safety_cfg():
    return SafetyConfig(d_model=64, freeze_backbone=True)


@pytest.fixture
def classifier(backbone, safety_cfg):
    return SafetyClassifier(backbone, safety_cfg)


def _make_dataloader(batch_size: int = 4, seq_len: int = 8, n_samples: int = 8):
    """Build a small DataLoader yielding {"input_ids", "labels"} dicts."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, 256, (n_samples, seq_len))
    labels = torch.randint(0, 2, (n_samples,)).float()

    class DictDataset(torch.utils.data.Dataset):
        def __len__(self):
            return n_samples

        def __getitem__(self, idx):
            return {"input_ids": input_ids[idx], "labels": labels[idx]}

    return DataLoader(DictDataset(), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_safety_classifier_forward_shape(classifier):
    """Logits must have shape (B,)."""
    input_ids = torch.randint(0, 256, (3, 8))
    logits = classifier(input_ids)
    assert logits.shape == (3,), f"Expected (3,), got {logits.shape}"


def test_safety_classifier_backbone_frozen(backbone, safety_cfg):
    """When freeze_backbone=True, all backbone params must have requires_grad=False."""
    clf = SafetyClassifier(backbone, SafetyConfig(d_model=64, freeze_backbone=True))
    for name, param in clf.backbone.named_parameters():
        assert not param.requires_grad, f"Backbone param {name} should be frozen"


def test_safety_head_trainable(classifier):
    """safety_head params must have requires_grad=True."""
    for name, param in classifier.safety_head.named_parameters():
        assert param.requires_grad, f"safety_head param {name} should be trainable"


def test_safety_loss_shape(classifier):
    """safety_loss must return a scalar tensor."""
    logits = torch.tensor([0.5, -0.3, 1.2])
    labels = torch.tensor([1.0, 0.0, 1.0])
    loss = classifier.safety_loss(logits, labels)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"


def test_safety_loss_range(classifier):
    """BCE loss must be strictly positive."""
    logits = torch.tensor([0.5, -0.3, 1.2])
    labels = torch.tensor([1.0, 0.0, 1.0])
    loss = classifier.safety_loss(logits, labels)
    assert loss.item() > 0.0, "Loss should be > 0"


def test_predict_returns_tuple(classifier):
    """predict() must return a 2-tuple."""
    input_ids = torch.randint(0, 256, (2, 8))
    result = classifier.predict(input_ids)
    assert isinstance(result, tuple) and len(result) == 2


def test_predict_probabilities_in_range(classifier):
    """All predicted probabilities must lie in [0, 1]."""
    input_ids = torch.randint(0, 256, (4, 8))
    is_unsafe, probs = classifier.predict(input_ids)
    assert is_unsafe.dtype == torch.bool, "is_unsafe should be BoolTensor"
    assert probs.dtype == torch.float32, "probs should be FloatTensor"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probs must be in [0, 1]"


def test_safety_filter_safe_mask_shape(classifier):
    """safety_filter must return safe_mask of shape (B,)."""
    B = 3
    input_ids = torch.randint(0, 256, (B, 8))
    generated_ids = torch.randint(0, 256, (B, 4))
    safe_mask, probs = safety_filter(classifier, input_ids, generated_ids)
    assert safe_mask.shape == (B,), f"Expected ({B},), got {safe_mask.shape}"
    assert safe_mask.dtype == torch.bool


def test_trainer_epoch_returns_float(classifier):
    """train_epoch must return a finite Python float."""
    trainer = SafetyTrainer(classifier)
    loader = _make_dataloader()
    mean_loss = trainer.train_epoch(loader)
    assert isinstance(mean_loss, float), "train_epoch should return float"
    assert math.isfinite(mean_loss), "Loss must be finite"


def test_trainer_updates_safety_head(backbone, safety_cfg):
    """safety_head weights must change after a training step; backbone stays frozen."""
    clf = SafetyClassifier(backbone, safety_cfg)
    trainer = SafetyTrainer(clf)

    weight_before = clf.safety_head.weight.clone()
    # Capture a backbone weight to verify it does NOT change
    backbone_weight_before = next(clf.backbone.parameters()).clone()

    loader = _make_dataloader()
    trainer.train_epoch(loader)

    assert not torch.equal(clf.safety_head.weight, weight_before), (
        "safety_head weight should have changed after training"
    )
    assert torch.equal(next(clf.backbone.parameters()), backbone_weight_before), (
        "Frozen backbone weight should not have changed"
    )


def test_evaluate_returns_metrics(classifier):
    """evaluate() must return a dict with the required keys."""
    trainer = SafetyTrainer(classifier)
    loader = _make_dataloader()
    metrics = trainer.evaluate(loader)
    required_keys = {"loss", "accuracy", "precision", "recall"}
    assert required_keys == set(metrics.keys()), (
        f"Missing keys: {required_keys - set(metrics.keys())}"
    )
    for key, val in metrics.items():
        assert isinstance(val, float), f"metrics[{key}] should be float, got {type(val)}"
        assert math.isfinite(val), f"metrics[{key}] = {val} is not finite"
