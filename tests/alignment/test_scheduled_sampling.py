"""Tests for src/alignment/scheduled_sampling.py"""

import pytest
import torch

from src.alignment.scheduled_sampling import ScheduledSampler


def test_scheduled_sampler_instantiates():
    s = ScheduledSampler()
    assert s is not None
    assert s.schedule == "linear"


def test_teacher_forcing_prob_linear_step0():
    s = ScheduledSampler(schedule="linear")
    assert s.teacher_forcing_prob(0, 100) == pytest.approx(1.0)


def test_teacher_forcing_prob_linear_end():
    s = ScheduledSampler(schedule="linear")
    assert s.teacher_forcing_prob(100, 100) == pytest.approx(0.0)


def test_exponential_schedule_decreases():
    s = ScheduledSampler(schedule="exponential", k=0.99)
    p0 = s.teacher_forcing_prob(0, 1000)
    p100 = s.teacher_forcing_prob(100, 1000)
    p500 = s.teacher_forcing_prob(500, 1000)
    assert p0 >= p100 >= p500


def test_sigmoid_schedule_decreases():
    s = ScheduledSampler(schedule="sigmoid", k=1.0)
    p0 = s.teacher_forcing_prob(0, 1000)
    p500 = s.teacher_forcing_prob(500, 1000)
    p1000 = s.teacher_forcing_prob(1000, 1000)
    assert p0 > p500 > p1000


def test_sample_input_returns_long_tensor_same_shape():
    s = ScheduledSampler()
    gt = torch.randint(0, 50, (2, 10))
    logits = torch.randn(2, 10, 50)
    out = s.sample_input(gt, logits, step=50, total_steps=100)
    assert out.dtype == torch.long
    assert out.shape == gt.shape


def test_sample_input_prob1_equals_ground_truth():
    s = ScheduledSampler(schedule="linear")
    gt = torch.randint(0, 50, (4, 8))
    logits = torch.randn(4, 8, 50)
    # p=1.0 at step=0 means all tokens should be ground truth
    torch.manual_seed(0)
    out = s.sample_input(gt, logits, step=0, total_steps=100)
    assert torch.all(out == gt)


def test_mixing_loss_returns_scalar():
    s = ScheduledSampler()
    logits = torch.randn(2, 10, 50)
    targets = torch.randint(0, 50, (2, 10))
    loss = s.mixing_loss(logits, targets, step=50, total_steps=100)
    assert loss.ndim == 0


def test_loss_is_finite_and_nonneg():
    s = ScheduledSampler()
    logits = torch.randn(2, 10, 50)
    targets = torch.randint(0, 50, (2, 10))
    loss = s.mixing_loss(logits, targets, step=30, total_steps=100)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_mixing_loss_prob0_still_valid():
    s = ScheduledSampler(schedule="linear")
    logits = torch.randn(2, 10, 50)
    targets = torch.randint(0, 50, (2, 10))
    # step == total_steps => p=0, loss should be 0
    loss = s.mixing_loss(logits, targets, step=100, total_steps=100)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
