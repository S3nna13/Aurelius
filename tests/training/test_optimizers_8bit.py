"""Tests for 8-bit optimizer wrappers."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.training.muon import Muon
from src.training.optimizers_8bit import Muon8bit, get_8bit_adamw, get_8bit_muon

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_bnb_module() -> ModuleType:
    """Return a mock bitsandbytes module with AdamW8bit and Adam8bit.update_momentum_8bit."""
    mock_bnb = ModuleType("bitsandbytes")
    mock_bnb.optim = MagicMock()
    mock_bnb.optim.AdamW8bit = MagicMock(return_value="adamw8bit_instance")
    mock_bnb.optim.GlobalStateManager = MagicMock()
    mock_bnb.optim.Adam8bit = MagicMock()
    mock_bnb.optim.Adam8bit.update_momentum_8bit = MagicMock(
        side_effect=lambda _p, _g, buf, _m: buf
    )
    return mock_bnb


# ---------------------------------------------------------------------------
# get_8bit_adamw
# ---------------------------------------------------------------------------


def test_get_8bit_adamw_returns_8bit_when_bnb_and_cuda():
    """When bitsandbytes is available and CUDA is present, return 8-bit AdamW."""
    mock_bnb = _mock_bnb_module()
    params = [nn.Parameter(torch.randn(10, 10))]

    with patch.dict(sys.modules, {"bitsandbytes": mock_bnb}):
        with patch("src.training.optimizers_8bit.torch.cuda.is_available", return_value=True):
            opt = get_8bit_adamw(params, lr=1e-3, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.01)

    assert opt == "adamw8bit_instance"
    mock_bnb.optim.AdamW8bit.assert_called_once_with(
        params, lr=1e-3, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.01
    )


def test_get_8bit_adamw_fallback_no_bnb():
    """When bitsandbytes is absent, fall back to standard torch AdamW."""
    params = [nn.Parameter(torch.randn(10, 10))]

    with patch.dict(sys.modules, {"bitsandbytes": None}):
        opt = get_8bit_adamw(params, lr=5e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1)

    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 5e-4
    assert opt.param_groups[0]["betas"] == (0.9, 0.99)
    assert opt.param_groups[0]["eps"] == 1e-8
    assert opt.param_groups[0]["weight_decay"] == 0.1


def test_get_8bit_adamw_fallback_no_cuda():
    """When bitsandbytes is present but CUDA is unavailable, fall back to standard AdamW."""
    mock_bnb = _mock_bnb_module()
    params = [nn.Parameter(torch.randn(10, 10))]

    with patch.dict(sys.modules, {"bitsandbytes": mock_bnb}):
        with patch("src.training.optimizers_8bit.torch.cuda.is_available", return_value=False):
            opt = get_8bit_adamw(params, lr=2e-4, betas=(0.85, 0.9), eps=1e-7, weight_decay=0.05)

    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 2e-4
    assert opt.param_groups[0]["betas"] == (0.85, 0.9)
    assert opt.param_groups[0]["eps"] == 1e-7
    assert opt.param_groups[0]["weight_decay"] == 0.05
    mock_bnb.optim.AdamW8bit.assert_not_called()


def test_get_8bit_adamw_hyperparameter_defaults():
    """Default hyperparameters are passed through correctly on fallback."""
    params = [nn.Parameter(torch.randn(10, 10))]

    with patch.dict(sys.modules, {"bitsandbytes": None}):
        opt = get_8bit_adamw(params)

    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 3e-4
    assert opt.param_groups[0]["betas"] == (0.9, 0.95)
    assert opt.param_groups[0]["eps"] == 1e-8
    assert opt.param_groups[0]["weight_decay"] == 0.1


# ---------------------------------------------------------------------------
# get_8bit_muon
# ---------------------------------------------------------------------------


def test_get_8bit_muon_returns_muon8bit_when_bnb_and_cuda():
    """When bitsandbytes is available and CUDA is present, return Muon8bit."""
    mock_bnb = _mock_bnb_module()
    params = [nn.Parameter(torch.randn(10, 10))]

    with patch.dict(sys.modules, {"bitsandbytes": mock_bnb}):
        with patch("src.training.optimizers_8bit.torch.cuda.is_available", return_value=True):
            opt = get_8bit_muon(params, lr=0.02, momentum=0.95, weight_decay=0.1)

    assert isinstance(opt, Muon8bit)
    assert opt.param_groups[0]["lr"] == 0.02
    assert opt.param_groups[0]["momentum"] == 0.95
    assert opt.param_groups[0]["weight_decay"] == 0.1


def test_get_8bit_muon_fallback_no_bnb():
    """When bitsandbytes is absent, fall back to standard Muon."""
    params = [nn.Parameter(torch.randn(10, 10))]

    with patch.dict(sys.modules, {"bitsandbytes": None}):
        opt = get_8bit_muon(params, lr=0.01, momentum=0.9, weight_decay=0.05)

    assert isinstance(opt, Muon)
    assert opt.param_groups[0]["lr"] == 0.01
    assert opt.param_groups[0]["momentum"] == 0.9
    assert opt.param_groups[0]["weight_decay"] == 0.05


def test_get_8bit_muon_fallback_no_cuda():
    """When bitsandbytes is present but CUDA is unavailable, fall back to standard Muon."""
    mock_bnb = _mock_bnb_module()
    params = [nn.Parameter(torch.randn(10, 10))]

    with patch.dict(sys.modules, {"bitsandbytes": mock_bnb}):
        with patch("src.training.optimizers_8bit.torch.cuda.is_available", return_value=False):
            opt = get_8bit_muon(params, lr=0.03, momentum=0.85, weight_decay=0.2)

    assert isinstance(opt, Muon)
    assert opt.param_groups[0]["lr"] == 0.03
    assert opt.param_groups[0]["momentum"] == 0.85
    assert opt.param_groups[0]["weight_decay"] == 0.2


def test_get_8bit_muon_hyperparameter_defaults():
    """Default hyperparameters are passed through correctly on fallback."""
    params = [nn.Parameter(torch.randn(10, 10))]

    with patch.dict(sys.modules, {"bitsandbytes": None}):
        opt = get_8bit_muon(params)

    assert isinstance(opt, Muon)
    assert opt.param_groups[0]["lr"] == 0.02
    assert opt.param_groups[0]["momentum"] == 0.95
    assert opt.param_groups[0]["weight_decay"] == 0.1


# ---------------------------------------------------------------------------
# Muon8bit
# ---------------------------------------------------------------------------


def test_muon8bit_requires_bitsandbytes():
    """Muon8bit raises ImportError when bitsandbytes is not installed."""
    params = [nn.Parameter(torch.randn(10, 10))]

    with patch.dict(sys.modules, {"bitsandbytes": None}):
        with pytest.raises(ImportError, match="bitsandbytes required"):
            Muon8bit(params)


def test_muon8bit_step_updates_parameters():
    """Muon8bit step must change parameter values when bitsandbytes is mocked."""
    mock_bnb = _mock_bnb_module()
    linear = nn.Linear(32, 64, bias=False)
    initial = linear.weight.data.clone()

    with patch.dict(sys.modules, {"bitsandbytes": mock_bnb}):
        optimizer = Muon8bit([linear.weight], lr=0.02, momentum=0.95)

    x = torch.randn(4, 32)
    loss = linear(x).sum()
    loss.backward()
    optimizer.step()

    assert not torch.allclose(linear.weight.data, initial), "Muon8bit did not update weights"


def test_muon8bit_zero_lr_no_update():
    """lr=0 should leave weights unchanged."""
    mock_bnb = _mock_bnb_module()
    linear = nn.Linear(32, 64, bias=False)
    initial = linear.weight.data.clone()

    with patch.dict(sys.modules, {"bitsandbytes": mock_bnb}):
        optimizer = Muon8bit([linear.weight], lr=0.0, momentum=0.95)

    x = torch.randn(4, 32)
    loss = linear(x).sum()
    loss.backward()
    optimizer.step()

    assert torch.allclose(linear.weight.data, initial), "Muon8bit changed weights with lr=0"


def test_muon8bit_weight_decay():
    """Weight decay should shrink parameter magnitude over time."""
    mock_bnb = _mock_bnb_module()
    linear = nn.Linear(32, 64, bias=False)
    nn.init.ones_(linear.weight)

    with patch.dict(sys.modules, {"bitsandbytes": mock_bnb}):
        optimizer = Muon8bit([linear.weight], lr=0.01, momentum=0.0, weight_decay=0.1)

    x = torch.randn(4, 32)
    loss = linear(x).sum()
    loss.backward()
    optimizer.step()

    assert linear.weight.norm() < 32 * 64**0.5
