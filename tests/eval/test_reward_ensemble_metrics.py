"""Tests for reward ensemble metrics."""

import pytest
import torch

from src.eval.reward_ensemble_metrics import (
    ensemble_disagreement,
    ensemble_mean,
    ensemble_std,
    reward_ensemble_report,
)


def make_scores():
    return torch.tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])


def test_ensemble_mean_reduces_over_models():
    mean = ensemble_mean(make_scores())
    assert torch.allclose(mean, torch.tensor([1.0, 3.0, 4.0]))


def test_ensemble_std_zero_for_identical_models():
    std = ensemble_std(torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    assert torch.allclose(std, torch.zeros(2))


def test_ensemble_disagreement_positive_when_models_differ():
    disagreement = ensemble_disagreement(make_scores())
    assert disagreement.item() > 0.0


def test_reward_ensemble_report_bundles_metrics():
    report = reward_ensemble_report(make_scores())
    assert report.mean.shape == (3,)
    assert report.std.shape == (3,)


def test_ensemble_mean_rejects_bad_rank():
    with pytest.raises(ValueError):
        ensemble_mean(torch.tensor([1.0, 2.0]))


def test_ensemble_std_rejects_bad_rank():
    with pytest.raises(ValueError):
        ensemble_std(torch.tensor([1.0, 2.0]))


def test_reward_ensemble_report_scalar_disagreement():
    report = reward_ensemble_report(make_scores())
    assert report.disagreement.ndim == 0
