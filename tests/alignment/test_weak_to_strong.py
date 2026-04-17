"""Tests for Weak-to-Strong Generalization (Burns et al. 2023)."""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alignment.weak_to_strong import (
    WeakSupervisor,
    WeakToStrongDataset,
    WeakToStrongLoss,
    WeakToStrongTrainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D_IN = 16
NUM_CLASSES = 4
BATCH_SIZE = 8


def _make_linear_model(d_in: int = D_IN, num_classes: int = NUM_CLASSES) -> nn.Linear:
    """Minimal single-layer linear model used as weak/strong stand-in."""
    torch.manual_seed(0)
    return nn.Linear(d_in, num_classes)


def _make_inputs(n: int = BATCH_SIZE, d: int = D_IN) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(n, d)


def _make_supervisor(num_classes: int = NUM_CLASSES) -> WeakSupervisor:
    return WeakSupervisor(_make_linear_model(num_classes=num_classes), num_classes=num_classes)


# ---------------------------------------------------------------------------
# WeakSupervisor tests
# ---------------------------------------------------------------------------

def test_supervisor_freezes_all_params():
    """WeakSupervisor must freeze every parameter in the wrapped model."""
    model = _make_linear_model()
    supervisor = WeakSupervisor(model, num_classes=NUM_CLASSES)
    for p in supervisor.model.parameters():
        assert not p.requires_grad, "All parameters should be frozen"


def test_supervisor_get_soft_labels_shape():
    """get_soft_labels must return (B, num_classes)."""
    supervisor = _make_supervisor()
    x = _make_inputs()
    out = supervisor.get_soft_labels(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"Expected ({BATCH_SIZE}, {NUM_CLASSES}), got {out.shape}"
    )


def test_supervisor_get_soft_labels_sum_to_one():
    """Each row of soft labels must sum to 1 (valid probability distribution)."""
    supervisor = _make_supervisor()
    x = _make_inputs()
    out = supervisor.get_soft_labels(x)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(BATCH_SIZE), atol=1e-5), (
        f"Row sums not close to 1: {row_sums}"
    )


def test_supervisor_get_soft_labels_no_grad():
    """get_soft_labels must not accumulate gradients."""
    supervisor = _make_supervisor()
    x = _make_inputs()
    out = supervisor.get_soft_labels(x)
    assert not out.requires_grad, "Output should not require grad (no_grad context)"


# ---------------------------------------------------------------------------
# WeakToStrongDataset tests
# ---------------------------------------------------------------------------

def test_dataset_len():
    """Dataset length must equal the number of input samples."""
    inputs = _make_inputs(n=12)
    supervisor = _make_supervisor()
    ds = WeakToStrongDataset(inputs, supervisor)
    assert len(ds) == 12


def test_dataset_getitem_returns_pair():
    """__getitem__ must return a (input, soft_label) 2-tuple."""
    inputs = _make_inputs()
    supervisor = _make_supervisor()
    ds = WeakToStrongDataset(inputs, supervisor)
    item = ds[0]
    assert isinstance(item, tuple) and len(item) == 2, (
        "Expected a 2-tuple (input, soft_label)"
    )


def test_dataset_soft_labels_shape():
    """Precomputed soft labels must have shape (N, num_classes)."""
    inputs = _make_inputs()
    supervisor = _make_supervisor()
    ds = WeakToStrongDataset(inputs, supervisor)
    assert ds.soft_labels.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"Expected soft_labels shape ({BATCH_SIZE}, {NUM_CLASSES}), got {ds.soft_labels.shape}"
    )


# ---------------------------------------------------------------------------
# WeakToStrongLoss tests
# ---------------------------------------------------------------------------

def _make_random_logits_and_labels(b: int = BATCH_SIZE, c: int = NUM_CLASSES):
    torch.manual_seed(42)
    student_logits = torch.randn(b, c)
    # Produce valid soft labels via softmax of random logits
    weak_logits = torch.randn(b, c)
    soft_labels = F.softmax(weak_logits, dim=-1)
    return student_logits, soft_labels


def test_loss_no_confidence_weighting_is_scalar():
    """WeakToStrongLoss without confidence weighting must return a scalar."""
    loss_fn = WeakToStrongLoss(confidence_weighting=False)
    student_logits, soft_labels = _make_random_logits_and_labels()
    loss = loss_fn(student_logits, soft_labels)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


def test_loss_is_non_negative():
    """KL divergence is always >= 0."""
    loss_fn = WeakToStrongLoss(confidence_weighting=False)
    student_logits, soft_labels = _make_random_logits_and_labels()
    loss = loss_fn(student_logits, soft_labels)
    assert loss.item() >= -1e-6, f"Loss should be non-negative, got {loss.item()}"


def test_loss_identical_student_teacher_near_zero():
    """When student replicates teacher exactly, KL divergence is ~0."""
    loss_fn = WeakToStrongLoss(confidence_weighting=False)
    torch.manual_seed(7)
    soft_labels = F.softmax(torch.randn(BATCH_SIZE, NUM_CLASSES), dim=-1)
    # student_logits whose softmax equals soft_labels: use log(soft_labels) as logits
    student_logits = soft_labels.log()
    loss = loss_fn(student_logits, soft_labels)
    assert loss.item() < 1e-5, f"Expected ~0 loss for identical distributions, got {loss.item()}"


def test_loss_confidence_weighting_returns_scalar():
    """WeakToStrongLoss with confidence_weighting=True must return a scalar."""
    loss_fn = WeakToStrongLoss(confidence_weighting=True, confidence_threshold=0.3)
    student_logits, soft_labels = _make_random_logits_and_labels()
    loss = loss_fn(student_logits, soft_labels)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


def test_loss_confidence_threshold_filters_samples():
    """High threshold should zero out uncertain samples, changing the loss."""
    torch.manual_seed(99)
    # Craft soft labels where some samples are peaked, some are diffuse
    b, c = 8, NUM_CLASSES
    # All-uniform soft labels: max prob = 1/c (low confidence)
    uniform = torch.full((b, c), 1.0 / c)
    torch.manual_seed(3)
    student_logits = torch.randn(b, c)

    loss_no_thresh = WeakToStrongLoss(confidence_weighting=True, confidence_threshold=0.0)
    loss_high_thresh = WeakToStrongLoss(confidence_weighting=True, confidence_threshold=0.9)

    val_no_thresh = loss_no_thresh(student_logits, uniform)
    val_high_thresh = loss_high_thresh(student_logits, uniform)

    # With threshold=0.9 and uniform labels (max=0.25 < 0.9), all samples filtered -> 0
    assert val_high_thresh.item() == 0.0, (
        f"Expected 0 loss when all samples filtered, got {val_high_thresh.item()}"
    )
    # Without threshold there should be nonzero loss (uniform != student softmax)
    assert val_no_thresh.item() >= 0.0


# ---------------------------------------------------------------------------
# WeakToStrongTrainer tests
# ---------------------------------------------------------------------------

def _make_trainer(confidence_weighting: bool = False, threshold: float = 0.0):
    torch.manual_seed(10)
    strong_model = _make_linear_model()
    optimizer = torch.optim.SGD(strong_model.parameters(), lr=1e-3)
    loss_fn = WeakToStrongLoss(
        confidence_weighting=confidence_weighting,
        confidence_threshold=threshold,
    )
    return WeakToStrongTrainer(strong_model, optimizer, loss_fn)


def test_trainer_train_step_returns_correct_keys():
    """train_step must return a dict with exactly 'loss', 'mean_confidence', 'frac_above_threshold'."""
    trainer = _make_trainer()
    x = _make_inputs()
    supervisor = _make_supervisor()
    soft_labels = supervisor.get_soft_labels(x)
    metrics = trainer.train_step(x, soft_labels)
    assert set(metrics.keys()) == {"loss", "mean_confidence", "frac_above_threshold"}, (
        f"Unexpected keys: {metrics.keys()}"
    )


def test_trainer_train_step_loss_is_finite():
    """train_step loss must be a finite float."""
    trainer = _make_trainer()
    x = _make_inputs()
    supervisor = _make_supervisor()
    soft_labels = supervisor.get_soft_labels(x)
    metrics = trainer.train_step(x, soft_labels)
    assert math.isfinite(metrics["loss"]), f"Loss is not finite: {metrics['loss']}"


def test_trainer_train_step_mean_confidence_in_unit_interval():
    """mean_confidence must lie in [0, 1]."""
    trainer = _make_trainer()
    x = _make_inputs()
    supervisor = _make_supervisor()
    soft_labels = supervisor.get_soft_labels(x)
    metrics = trainer.train_step(x, soft_labels)
    c = metrics["mean_confidence"]
    assert 0.0 <= c <= 1.0, f"mean_confidence out of [0,1]: {c}"
