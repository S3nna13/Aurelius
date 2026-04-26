"""Unit tests for Step-DPO."""

from __future__ import annotations

import math

import pytest
import torch

from src.alignment.step_dpo import (
    StepDPOTrainer,
    StepPreferenceExample,
    step_dpo_loss,
)


def _mkex(c=0.0, r=0.0, cr=0.0, rr=0.0, prefix=0.0, requires_grad=False):
    def t(v):
        return torch.tensor(float(v), requires_grad=requires_grad)

    return StepPreferenceExample(
        prefix_logprobs=t(prefix),
        chosen_step_logprobs=t(c),
        rejected_step_logprobs=t(r),
        chosen_step_ref_logprobs=t(cr),
        rejected_step_ref_logprobs=t(rr),
    )


def test_loss_returns_scalar_finite():
    c = torch.tensor([0.1, -0.2, 0.3])
    r = torch.tensor([-0.1, 0.2, -0.4])
    cr = torch.zeros(3)
    rr = torch.zeros(3)
    loss = step_dpo_loss(c, r, cr, rr, beta=0.1)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gradient_flow():
    c = torch.tensor([0.3, 0.1], requires_grad=True)
    r = torch.tensor([-0.2, 0.05], requires_grad=True)
    cr = torch.zeros(2)
    rr = torch.zeros(2)
    loss = step_dpo_loss(c, r, cr, rr, beta=0.5)
    loss.backward()
    assert c.grad is not None and r.grad is not None
    assert torch.all(torch.isfinite(c.grad))
    assert torch.all(torch.isfinite(r.grad))
    # chosen should want to increase -> negative gradient
    assert (c.grad <= 0).all()
    # rejected should want to decrease -> positive gradient
    assert (r.grad >= 0).all()


def test_loss_equals_log2_when_policy_equals_ref():
    c = torch.tensor([0.7, -0.3])
    r = torch.tensor([-0.1, 0.5])
    # ref equals policy -> margin = 0 -> softplus(0) = log(2)
    loss = step_dpo_loss(c, r, c.clone(), r.clone(), beta=0.1)
    assert math.isclose(loss.item(), math.log(2.0), abs_tol=1e-6)


def test_loss_goes_to_zero_when_policy_prefers_chosen_strongly():
    c = torch.tensor([20.0])
    r = torch.tensor([-20.0])
    cr = torch.zeros(1)
    rr = torch.zeros(1)
    loss = step_dpo_loss(c, r, cr, rr, beta=1.0)
    assert loss.item() < 1e-8


def test_loss_increases_when_policy_prefers_rejected():
    base_c, base_r = torch.tensor([0.0]), torch.tensor([0.0])
    cr, rr = torch.zeros(1), torch.zeros(1)
    base_loss = step_dpo_loss(base_c, base_r, cr, rr, beta=0.5)
    bad_loss = step_dpo_loss(torch.tensor([-1.0]), torch.tensor([1.0]), cr, rr, beta=0.5)
    assert bad_loss.item() > base_loss.item()


def test_beta_zero_gives_log2_always():
    c = torch.tensor([5.0, -3.0])
    r = torch.tensor([-5.0, 3.0])
    cr = torch.tensor([1.0, 2.0])
    rr = torch.tensor([-1.0, -2.0])
    loss = step_dpo_loss(c, r, cr, rr, beta=0.0)
    assert math.isclose(loss.item(), math.log(2.0), abs_tol=1e-6)


def test_shape_mismatch_raises():
    c = torch.zeros(3)
    r = torch.zeros(2)
    cr = torch.zeros(3)
    rr = torch.zeros(3)
    with pytest.raises(ValueError):
        step_dpo_loss(c, r, cr, rr)


def test_step_preference_example_validates_fields():
    with pytest.raises(TypeError):
        StepPreferenceExample(
            prefix_logprobs=0.0,  # not a tensor
            chosen_step_logprobs=torch.tensor(0.0),
            rejected_step_logprobs=torch.tensor(0.0),
            chosen_step_ref_logprobs=torch.tensor(0.0),
            rejected_step_ref_logprobs=torch.tensor(0.0),
        )
    # valid construction works
    ex = _mkex()
    assert isinstance(ex.chosen_step_logprobs, torch.Tensor)


def test_compute_loss_aggregates_over_batch():
    trainer = StepDPOTrainer(beta=0.1)
    batch = [
        _mkex(c=0.5, r=-0.5, cr=0.0, rr=0.0),
        _mkex(c=0.2, r=-0.2, cr=0.0, rr=0.0),
        _mkex(c=1.0, r=-1.0, cr=0.0, rr=0.0),
    ]
    loss = trainer.compute_loss(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    # compare vs manual
    c = torch.tensor([0.5, 0.2, 1.0])
    r = torch.tensor([-0.5, -0.2, -1.0])
    z = torch.zeros(3)
    expected = step_dpo_loss(c, r, z, z, beta=0.1)
    assert torch.allclose(loss, expected)


def test_trainer_step_returns_metrics_dict():
    theta_c = torch.tensor([0.1, 0.0], requires_grad=True)
    theta_r = torch.tensor([0.0, 0.0], requires_grad=True)

    # build examples that share these params via a closure
    batch = [
        StepPreferenceExample(
            prefix_logprobs=torch.tensor(0.0),
            chosen_step_logprobs=theta_c[0],
            rejected_step_logprobs=theta_r[0],
            chosen_step_ref_logprobs=torch.tensor(0.0),
            rejected_step_ref_logprobs=torch.tensor(0.0),
        ),
        StepPreferenceExample(
            prefix_logprobs=torch.tensor(0.0),
            chosen_step_logprobs=theta_c[1],
            rejected_step_logprobs=theta_r[1],
            chosen_step_ref_logprobs=torch.tensor(0.0),
            rejected_step_ref_logprobs=torch.tensor(0.0),
        ),
    ]
    trainer = StepDPOTrainer(beta=0.1)
    opt = torch.optim.SGD([theta_c, theta_r], lr=0.1)
    metrics = trainer.step(opt, batch)
    assert "loss" in metrics and "reward_margin" in metrics
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["reward_margin"])


def test_determinism():
    torch.manual_seed(0)
    c = torch.randn(5)
    r = torch.randn(5)
    cr = torch.randn(5)
    rr = torch.randn(5)
    l1 = step_dpo_loss(c, r, cr, rr, beta=0.2)
    l2 = step_dpo_loss(c.clone(), r.clone(), cr.clone(), rr.clone(), beta=0.2)
    assert torch.allclose(l1, l2)


def test_nan_safe_on_extreme_logprobs():
    c = torch.tensor([1e4, -1e4])
    r = torch.tensor([-1e4, 1e4])
    cr = torch.zeros(2)
    rr = torch.zeros(2)
    loss = step_dpo_loss(c, r, cr, rr, beta=1.0)
    assert torch.isfinite(loss)


def test_invalid_beta_raises():
    c = torch.zeros(1)
    with pytest.raises(ValueError):
        step_dpo_loss(c, c, c, c, beta=-0.1)
    with pytest.raises(ValueError):
        StepDPOTrainer(beta=-1.0)


def test_batch_of_one_works():
    trainer = StepDPOTrainer(beta=0.1)
    batch = [_mkex(c=0.5, r=-0.5)]
    loss = trainer.compute_loss(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_empty_batch_raises():
    trainer = StepDPOTrainer(beta=0.1)
    with pytest.raises(ValueError):
        trainer.compute_loss([])
