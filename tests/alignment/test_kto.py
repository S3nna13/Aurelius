"""Tests for KTO utilities."""

import pytest
import torch

from src.alignment.kto import kto_accuracy, kto_loss, kto_metrics, prospect_value


def test_prospect_value_handles_gains_and_losses():
    values = torch.tensor([4.0, -4.0])
    transformed = prospect_value(values, gain_exponent=0.5, loss_exponent=0.5, loss_aversion=2.0)
    assert transformed[0].item() == pytest.approx(2.0)
    assert transformed[1].item() == pytest.approx(-4.0)


def test_kto_desirable_positive_log_ratio_has_lower_loss():
    better = kto_loss(torch.tensor([1.0]), torch.tensor([1]))
    worse = kto_loss(torch.tensor([-1.0]), torch.tensor([1]))
    assert better.item() < worse.item()


def test_kto_undesirable_negative_log_ratio_has_lower_loss():
    better = kto_loss(torch.tensor([-1.0]), torch.tensor([0]))
    worse = kto_loss(torch.tensor([1.0]), torch.tensor([0]))
    assert better.item() < worse.item()


def test_kto_none_reduction_matches_shape():
    losses = kto_loss(torch.tensor([0.5, -0.3, 1.2]), torch.tensor([1, 0, 1]), reduction="none")
    assert losses.shape == (3,)


def test_kto_accuracy_uses_sign_threshold():
    acc = kto_accuracy(torch.tensor([0.4, -0.8, 0.1]), torch.tensor([1, 0, 0]))
    assert acc.item() == pytest.approx(2.0 / 3.0)


def test_kto_backward_produces_gradients():
    ratios = torch.tensor([0.2, -0.4, 0.5], requires_grad=True)
    labels = torch.tensor([1, 0, 1])
    loss = kto_loss(ratios, labels)
    loss.backward()
    assert ratios.grad is not None


def test_kto_metrics_exposes_signed_utilities():
    metrics = kto_metrics(torch.tensor([1.0, -1.0]), torch.tensor([1, 0]), beta=1.0)
    assert metrics.signed_utilities[0].item() > 0
    assert metrics.signed_utilities[1].item() > 0

