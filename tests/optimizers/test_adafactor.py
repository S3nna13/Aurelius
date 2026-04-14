"""Tests for the Adafactor optimizer."""

from __future__ import annotations

import pytest
import torch

from src.optimizers.adafactor import (
    Adafactor,
    adafactor_factorized_second_moment,
    adafactor_reconstruct_second_moment,
)


def test_factorized_second_moment_shapes():
    grad_sq = torch.arange(1.0, 13.0).view(3, 4)
    row, col = adafactor_factorized_second_moment(grad_sq)
    assert row.shape == (3,)
    assert col.shape == (4,)


def test_factorized_second_moment_round_trip_shape():
    grad_sq = torch.arange(1.0, 13.0).view(3, 4)
    row, col = adafactor_factorized_second_moment(grad_sq)
    recon = adafactor_reconstruct_second_moment(row, col)
    assert recon.shape == grad_sq.shape


def test_factorized_second_moment_preserves_mean_scale():
    grad_sq = torch.arange(1.0, 13.0).view(3, 4)
    row, col = adafactor_factorized_second_moment(grad_sq)
    recon = adafactor_reconstruct_second_moment(row, col)
    assert recon.mean().item() == pytest.approx(grad_sq.mean().item(), rel=0.5)


def test_adafactor_uses_factorized_state_for_matrices():
    param = torch.nn.Parameter(torch.ones(3, 4))
    opt = Adafactor([param], lr=0.1)
    param.grad = torch.full_like(param, 0.5)
    opt.step()
    state = opt.state[param]
    assert state["exp_avg_sq_row"].shape == (3,)
    assert state["exp_avg_sq_col"].shape == (4,)


def test_adafactor_uses_full_state_for_vectors():
    param = torch.nn.Parameter(torch.ones(5))
    opt = Adafactor([param], lr=0.1)
    param.grad = torch.full_like(param, 0.5)
    opt.step()
    state = opt.state[param]
    assert state["exp_avg_sq"].shape == (5,)


def test_adafactor_step_updates_parameters():
    param = torch.nn.Parameter(torch.ones(2, 2))
    opt = Adafactor([param], lr=0.1, beta2=0.9)
    param.grad = torch.ones_like(param)
    before = param.detach().clone()
    opt.step()
    assert not torch.allclose(param, before)


def test_adafactor_zero_grad_is_no_op():
    param = torch.nn.Parameter(torch.ones(2, 2))
    opt = Adafactor([param], lr=0.1)
    before = param.detach().clone()
    opt.step()
    assert torch.allclose(param, before)


def test_adafactor_state_dict_round_trip():
    param = torch.nn.Parameter(torch.ones(2, 2))
    opt = Adafactor([param], lr=0.1)
    param.grad = torch.ones_like(param)
    opt.step()
    state = opt.state_dict()

    clone = torch.nn.Parameter(torch.ones(2, 2))
    clone_opt = Adafactor([clone], lr=0.1)
    clone_opt.load_state_dict(state)

    assert clone_opt.state[clone]["step"] == opt.state[param]["step"]
    assert torch.allclose(
        clone_opt.state[clone]["exp_avg_sq_row"], opt.state[param]["exp_avg_sq_row"]
    )
