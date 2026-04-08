"""Tests for the SOAP optimizer."""

import pytest
import torch
import torch.nn as nn

from src.training.soap import SOAP


def test_soap_updates_linear_parameters():
    model = nn.Linear(4, 3)
    optimizer = SOAP(model.parameters(), lr=1e-2)
    before = model.weight.detach().clone()
    loss = model(torch.randn(8, 4)).pow(2).mean()
    loss.backward()
    optimizer.step()
    assert not torch.allclose(model.weight, before)


def test_soap_closure_returns_loss():
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = SOAP([param], lr=1e-2)

    def closure():
        optimizer.zero_grad()
        loss = (param ** 2).sum()
        loss.backward()
        return loss

    result = optimizer.step(closure)
    assert result is not None
    assert result.item() == pytest.approx(1.0)


def test_soap_initializes_matrix_state():
    param = nn.Parameter(torch.randn(3, 4))
    optimizer = SOAP([param], lr=1e-2)
    param.grad = torch.randn_like(param)
    optimizer.step()
    state = optimizer.state[param]
    assert state["row_cov"].shape == (3, 3)
    assert state["col_cov"].shape == (4, 4)


def test_soap_vector_parameter_uses_matrix_fallback():
    param = nn.Parameter(torch.randn(5))
    optimizer = SOAP([param], lr=1e-2)
    before = param.detach().clone()
    param.grad = torch.randn_like(param)
    optimizer.step()
    assert not torch.allclose(param, before)


def test_soap_precondition_frequency_delays_basis_refresh():
    param = nn.Parameter(torch.randn(2, 2))
    optimizer = SOAP([param], lr=1e-2, precondition_frequency=2)
    param.grad = torch.randn_like(param)
    optimizer.step()
    first = optimizer.state[param]["row_inv_root"].clone()
    param.grad = torch.randn_like(param)
    optimizer.step()
    second = optimizer.state[param]["row_inv_root"].clone()
    assert torch.allclose(first, torch.eye(2), atol=1e-6)
    assert not torch.allclose(second, torch.eye(2), atol=1e-6)


def test_soap_weight_decay_changes_update():
    param_a = nn.Parameter(torch.ones(2, 2))
    param_b = nn.Parameter(torch.ones(2, 2))
    grad = torch.full((2, 2), 0.25)
    opt_a = SOAP([param_a], lr=1e-2, weight_decay=0.0)
    opt_b = SOAP([param_b], lr=1e-2, weight_decay=0.1)
    param_a.grad = grad.clone()
    param_b.grad = grad.clone()
    opt_a.step()
    opt_b.step()
    assert not torch.allclose(param_a, param_b)


def test_soap_rejects_invalid_hparams():
    with pytest.raises(ValueError):
        SOAP([nn.Parameter(torch.randn(()))], lr=0.0)
    with pytest.raises(ValueError):
        SOAP([nn.Parameter(torch.randn(()))], precondition_frequency=0)
