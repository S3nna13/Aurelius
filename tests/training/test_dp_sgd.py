from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.training.dp_sgd import DPSGDConfig, DPSGDOptimizer


def make_dp_opt(
    lr: float = 0.01,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.1,
    batch_size: int = 256,
    delta: float = 1e-5,
):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    config = DPSGDConfig(
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        batch_size=batch_size,
        delta=delta,
    )
    dp_opt = DPSGDOptimizer(optimizer, config)
    return model, dp_opt


def run_forward_backward(model):
    x = torch.randn(4, 4)
    y = torch.randn(4, 2)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()


# --- DPSGDConfig tests ---


def test_config_defaults():
    cfg = DPSGDConfig()
    assert cfg.max_grad_norm == 1.0
    assert cfg.noise_multiplier == 1.1
    assert cfg.batch_size == 256
    assert cfg.delta == 1e-5


def test_config_custom():
    cfg = DPSGDConfig(max_grad_norm=0.5, noise_multiplier=2.0, batch_size=64, delta=1e-6)
    assert cfg.max_grad_norm == 0.5
    assert cfg.batch_size == 64


# --- Construction validation ---


def test_invalid_max_grad_norm():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="max_grad_norm"):
        DPSGDOptimizer(opt, DPSGDConfig(max_grad_norm=0.0))


def test_invalid_noise_multiplier():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="noise_multiplier"):
        DPSGDOptimizer(opt, DPSGDConfig(noise_multiplier=-1.0))


def test_invalid_batch_size():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="batch_size"):
        DPSGDOptimizer(opt, DPSGDConfig(batch_size=0))


def test_invalid_delta():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="delta"):
        DPSGDOptimizer(opt, DPSGDConfig(delta=1.5))


# --- _clip_grad ---


def test_clip_grad_returns_float():
    model, dp_opt = make_dp_opt()
    run_forward_backward(model)
    norm = dp_opt._clip_grad(list(model.parameters()))
    assert isinstance(norm, float)
    assert norm >= 0.0


def test_clip_grad_clips_large_grads():
    model, dp_opt = make_dp_opt(max_grad_norm=0.01)
    run_forward_backward(model)
    dp_opt._clip_grad(list(model.parameters()))
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    assert math.sqrt(total) <= 0.01 + 1e-6


def test_clip_grad_no_op_when_small():
    model, dp_opt = make_dp_opt(max_grad_norm=1000.0)
    run_forward_backward(model)
    grads_before = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    dp_opt._clip_grad(list(model.parameters()))
    grads_after = [p.grad for p in model.parameters() if p.grad is not None]
    for g_b, g_a in zip(grads_before, grads_after):
        assert torch.allclose(g_b, g_a, atol=1e-6)


# --- _add_noise ---


def test_add_noise_changes_grads():
    model, dp_opt = make_dp_opt()
    run_forward_backward(model)
    grads_before = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    dp_opt._add_noise(list(model.parameters()))
    changed = any(
        not torch.equal(g_b, p.grad)
        for g_b, p in zip(grads_before, model.parameters())
        if p.grad is not None
    )
    assert changed


def test_add_noise_zero_multiplier_no_change():
    model, dp_opt = make_dp_opt(noise_multiplier=0.0)
    run_forward_backward(model)
    grads_before = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    dp_opt._add_noise(list(model.parameters()))
    for g_b, p in zip(grads_before, model.parameters()):
        if p.grad is not None:
            assert torch.allclose(g_b, p.grad, atol=1e-6)


# --- step ---


def test_step_returns_dict():
    model, dp_opt = make_dp_opt()
    run_forward_backward(model)
    result = dp_opt.step(list(model.parameters()))
    assert "mean_grad_norm" in result
    assert "noise_std" in result


def test_step_noise_std_formula():
    model, dp_opt = make_dp_opt(noise_multiplier=1.5, max_grad_norm=2.0, batch_size=32)
    run_forward_backward(model)
    result = dp_opt.step(list(model.parameters()))
    expected_std = 1.5 * 2.0 / 32
    assert abs(result["noise_std"] - expected_std) < 1e-9


def test_step_updates_params():
    model, dp_opt = make_dp_opt()
    params_before = [p.data.clone() for p in model.parameters()]
    run_forward_backward(model)
    dp_opt.step(list(model.parameters()))
    updated = any(not torch.equal(pb, p.data) for pb, p in zip(params_before, model.parameters()))
    assert updated


def test_zero_grad_clears_grads():
    model, dp_opt = make_dp_opt()
    run_forward_backward(model)
    dp_opt.zero_grad()
    for p in model.parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0.0


# --- get_privacy_spent ---


def test_privacy_spent_keys():
    _, dp_opt = make_dp_opt()
    spent = dp_opt.get_privacy_spent(100)
    assert set(spent.keys()) == {"epsilon", "delta", "steps"}


def test_privacy_spent_increases_with_steps():
    _, dp_opt = make_dp_opt()
    e1 = dp_opt.get_privacy_spent(100)["epsilon"]
    e2 = dp_opt.get_privacy_spent(200)["epsilon"]
    assert e2 > e1


def test_privacy_spent_delta_matches_config():
    _, dp_opt = make_dp_opt(delta=1e-6)
    spent = dp_opt.get_privacy_spent(50)
    assert spent["delta"] == 1e-6


def test_privacy_spent_steps_echoed():
    _, dp_opt = make_dp_opt()
    spent = dp_opt.get_privacy_spent(77)
    assert spent["steps"] == 77


def test_privacy_spent_formula():
    cfg = DPSGDConfig(noise_multiplier=1.1, max_grad_norm=1.0, batch_size=256, delta=1e-5)
    model = nn.Linear(4, 2)
    dp_opt = DPSGDOptimizer(torch.optim.SGD(model.parameters(), lr=0.01), cfg)
    steps = 1000
    expected = (1.1**-2) * 2.0 * math.log(1.25 / 1e-5) * steps / 256
    result = dp_opt.get_privacy_spent(steps)["epsilon"]
    assert abs(result - expected) < 1e-9
