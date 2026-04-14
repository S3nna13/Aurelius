"""Tests for the RAdam optimizer."""

from __future__ import annotations

import pytest
import torch

from src.optimizers.radam import RAdam, radam_rectification, radam_rho_inf, radam_rho_t


def test_rho_inf_matches_formula():
    assert radam_rho_inf(0.999) == pytest.approx(1999.0)


def test_rho_t_is_positive_and_grows():
    rho_1 = radam_rho_t(1, 0.999)
    rho_10 = radam_rho_t(10, 0.999)
    assert rho_1 > 0.0
    assert rho_10 > rho_1


def test_rectification_is_zero_in_warmup():
    assert radam_rectification(1, 0.999) == 0.0


def test_rectification_turns_on_later():
    assert radam_rectification(100, 0.999) > 0.0


def test_radam_first_step_matches_momentum_sgd():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = RAdam([param], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    param.grad = torch.ones_like(param)
    opt.step()
    assert torch.allclose(param, torch.tensor([0.9, 1.9]), atol=1e-6)


def test_radam_state_tracks_rectification():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = RAdam([param], lr=0.1)
    for _ in range(5):
        param.grad = torch.ones_like(param)
        opt.step()
    state = opt.state[param]
    assert state["step"] == 5
    assert "exp_avg" in state
    assert "exp_avg_sq" in state
    assert state["rho_t"] == pytest.approx(radam_rho_t(5, 0.999))


def test_radam_state_dict_round_trip():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = RAdam([param], lr=0.1)
    param.grad = torch.ones_like(param)
    opt.step()
    state = opt.state_dict()

    clone = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    clone_opt = RAdam([clone], lr=0.1)
    clone_opt.load_state_dict(state)
    assert clone_opt.state[clone]["step"] == opt.state[param]["step"]
    assert torch.allclose(clone_opt.state[clone]["exp_avg"], opt.state[param]["exp_avg"])
