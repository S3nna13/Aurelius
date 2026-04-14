"""Tests for the LAMB optimizer."""

from __future__ import annotations

import pytest
import torch

from src.optimizers.lamb import LAMB, lamb_trust_ratio


def test_trust_ratio_defaults_to_one_for_zero_norms():
    assert lamb_trust_ratio(torch.zeros(2), torch.ones(2)) == 1.0
    assert lamb_trust_ratio(torch.ones(2), torch.zeros(2)) == 1.0


def test_trust_ratio_matches_norm_ratio():
    param = torch.tensor([3.0, 4.0])
    update = torch.tensor([1.0, 2.0])
    ratio = lamb_trust_ratio(param, update, eps=0.0, trust_clip=None)
    assert ratio == pytest.approx(param.norm().item() / update.norm().item())


def test_lamb_first_step_scales_by_trust_ratio():
    param = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    opt = LAMB([param], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    param.grad = torch.ones_like(param)
    opt.step()
    expected = torch.tensor([3.0, 4.0]) - 0.1 * (torch.tensor([1.0, 1.0]) * (5.0 / 2.0**0.5))
    assert torch.allclose(param.detach(), expected, atol=1e-5)


def test_lamb_state_tracks_trust_ratio():
    param = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    opt = LAMB([param], lr=0.1)
    param.grad = torch.ones_like(param)
    opt.step()
    state = opt.state[param]
    assert state["step"] == 1
    assert state["trust_ratio"] > 0.0
    assert "exp_avg" in state and "exp_avg_sq" in state


def test_lamb_weight_decay_changes_update():
    param = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    opt = LAMB([param], lr=0.1, weight_decay=0.1)
    param.grad = torch.zeros_like(param)
    before = param.detach().clone()
    opt.step()
    assert not torch.allclose(param.detach(), before)


def test_lamb_state_dict_round_trip():
    param = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    opt = LAMB([param], lr=0.1)
    param.grad = torch.ones_like(param)
    opt.step()
    state = opt.state_dict()

    clone = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    clone_opt = LAMB([clone], lr=0.1)
    clone_opt.load_state_dict(state)
    assert clone_opt.state[clone]["step"] == opt.state[param]["step"]
    assert clone_opt.state[clone]["trust_ratio"] == pytest.approx(opt.state[param]["trust_ratio"])
