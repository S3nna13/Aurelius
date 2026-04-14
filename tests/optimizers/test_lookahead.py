"""Tests for the Lookahead optimizer wrapper."""

from __future__ import annotations

import torch

from src.optimizers.lookahead import Lookahead


def test_lookahead_initializes_slow_weights():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    base = torch.optim.SGD([param], lr=0.1)
    opt = Lookahead(base, k=2, alpha=0.5)
    assert torch.allclose(opt.slow_params[0], param.detach())


def test_lookahead_does_not_sync_before_k_steps():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    base = torch.optim.SGD([param], lr=0.1)
    opt = Lookahead(base, k=2, alpha=0.5)
    param.grad = torch.ones_like(param)
    opt.step()
    assert torch.allclose(opt.slow_params[0], torch.tensor([1.0, 2.0]))


def test_lookahead_syncs_on_kth_step():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    base = torch.optim.SGD([param], lr=0.1)
    opt = Lookahead(base, k=2, alpha=0.5)
    param.grad = torch.ones_like(param)
    opt.step()
    param.grad = torch.ones_like(param)
    opt.step()
    assert torch.allclose(param.detach(), torch.tensor([0.9, 1.9]))


def test_lookahead_alpha_one_copies_fast_weights():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    base = torch.optim.SGD([param], lr=0.1)
    opt = Lookahead(base, k=1, alpha=1.0)
    param.grad = torch.ones_like(param)
    opt.step()
    assert torch.allclose(opt.slow_params[0], param.detach())


def test_lookahead_zero_grad_delegates():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    base = torch.optim.SGD([param], lr=0.1)
    opt = Lookahead(base, k=2, alpha=0.5)
    param.grad = torch.ones_like(param)
    opt.zero_grad(set_to_none=True)
    assert param.grad is None


def test_lookahead_state_dict_round_trip():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    base = torch.optim.SGD([param], lr=0.1)
    opt = Lookahead(base, k=2, alpha=0.5)
    param.grad = torch.ones_like(param)
    opt.step()
    state = opt.state_dict()

    clone = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    clone_base = torch.optim.SGD([clone], lr=0.1)
    clone_opt = Lookahead(clone_base, k=2, alpha=0.5)
    clone_opt.load_state_dict(state)
    assert clone_opt.k == opt.k
    assert clone_opt.alpha == opt.alpha
    assert torch.allclose(clone_opt.slow_params[0], opt.slow_params[0])
