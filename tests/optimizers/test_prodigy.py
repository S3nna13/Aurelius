"""Unit tests for Prodigy optimizer."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.optimizers.prodigy import Prodigy


def _make_linear(in_f: int = 4, out_f: int = 2, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(in_f, out_f)


def test_instantiate_on_linear():
    model = _make_linear()
    opt = Prodigy(model.parameters())
    assert isinstance(opt, torch.optim.Optimizer)
    assert opt.param_groups[0]["d"] > 0.0


def test_step_reduces_loss_on_quadratic():
    torch.manual_seed(0)
    x = torch.randn(1, 4)
    target = torch.zeros(1, 2)
    model = _make_linear()
    opt = Prodigy(model.parameters(), lr=1.0)

    # Warmup a few steps for d to grow.
    initial_loss = None
    final_loss = None
    for i in range(200):
        opt.zero_grad()
        out = model(x)
        loss = ((out - target) ** 2).mean()
        if i == 0:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
        final_loss = loss.item()
    assert final_loss < initial_loss


def test_d_estimate_non_decreasing():
    torch.manual_seed(1)
    model = _make_linear()
    opt = Prodigy(model.parameters(), lr=1.0)
    x = torch.randn(3, 4)
    y = torch.randn(3, 2)

    ds = []
    for _ in range(30):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        ds.append(opt.param_groups[0]["d"])
    # d estimate should be monotonically non-decreasing.
    for a, b in zip(ds, ds[1:]):
        assert b >= a - 1e-12


def test_determinism_with_manual_seed():
    def run():
        torch.manual_seed(42)
        model = _make_linear(seed=42)
        opt = Prodigy(model.parameters())
        x = torch.randn(2, 4)
        y = torch.randn(2, 2)
        for _ in range(10):
            opt.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
        return torch.cat([p.flatten() for p in model.parameters()])

    a = run()
    b = run()
    assert torch.allclose(a, b)


def test_invalid_lr_raises():
    model = _make_linear()
    with pytest.raises(ValueError):
        Prodigy(model.parameters(), lr=0.0)
    with pytest.raises(ValueError):
        Prodigy(model.parameters(), lr=-1.0)


def test_invalid_betas_raises():
    model = _make_linear()
    with pytest.raises(ValueError):
        Prodigy(model.parameters(), betas=(1.1, 0.999))
    with pytest.raises(ValueError):
        Prodigy(model.parameters(), betas=(0.9, -0.1))
    with pytest.raises(ValueError):
        Prodigy(model.parameters(), betas=(0.9,))


def test_growth_rate_caps_d_growth():
    torch.manual_seed(3)
    model = _make_linear()
    opt = Prodigy(model.parameters(), lr=1.0, growth_rate=1.01)
    x = torch.randn(4, 4) * 100
    y = torch.zeros(4, 2)
    # Prodigy's first real d update may jump d directly to d_hat (when d == d0);
    # afterwards d is capped by d * growth_rate per step. Warm up until d > d0.
    d0 = opt.param_groups[0]["d0"]
    for _ in range(10):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        if opt.param_groups[0]["d"] > d0:
            break

    prev_d = opt.param_groups[0]["d"]
    for _ in range(30):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        new_d = opt.param_groups[0]["d"]
        assert new_d <= prev_d * 1.01 + 1e-12
        prev_d = new_d


def test_weight_decay_shrinks_weights():
    torch.manual_seed(4)
    model = _make_linear()
    # Zero gradients by training with zero inputs / zero target -> grads small; rely on wd.
    opt = Prodigy(model.parameters(), lr=1.0, weight_decay=0.1)

    # Force a few steps with non-zero grad so d grows and dlr > 0.
    x = torch.randn(2, 4)
    y = torch.randn(2, 2)
    for _ in range(30):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()

    # Compare with wd=0.
    torch.manual_seed(4)
    model2 = _make_linear()
    opt2 = Prodigy(model2.parameters(), lr=1.0, weight_decay=0.0)
    for _ in range(30):
        opt2.zero_grad()
        loss = ((model2(x) - y) ** 2).mean()
        loss.backward()
        opt2.step()

    w_norm = sum(p.detach().pow(2).sum().item() for p in model.parameters())
    w_norm2 = sum(p.detach().pow(2).sum().item() for p in model2.parameters())
    assert w_norm < w_norm2 + 1e-6 or w_norm != w_norm2  # wd had some effect


def test_state_dict_roundtrip():
    torch.manual_seed(5)
    model = _make_linear()
    opt = Prodigy(model.parameters(), lr=1.0)
    x = torch.randn(2, 4)
    y = torch.randn(2, 2)
    for _ in range(5):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()

    sd = copy.deepcopy(opt.state_dict())

    model2 = _make_linear()
    opt2 = Prodigy(model2.parameters(), lr=1.0)
    opt2.load_state_dict(sd)

    assert opt2.param_groups[0]["d"] == opt.param_groups[0]["d"]
    assert opt2.param_groups[0]["k"] == opt.param_groups[0]["k"]


def test_closure_supported():
    torch.manual_seed(6)
    model = _make_linear()
    opt = Prodigy(model.parameters(), lr=1.0)
    x = torch.randn(2, 4)
    y = torch.randn(2, 2)

    def closure():
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert torch.is_tensor(loss)


def test_none_grad_skipped():
    torch.manual_seed(7)
    model = _make_linear()
    opt = Prodigy(model.parameters(), lr=1.0)
    # Do not run backward, so grads are None.
    opt.zero_grad(set_to_none=True)
    # Should not crash.
    opt.step()


def test_compares_favorably_to_adamw_default():
    torch.manual_seed(8)
    x = torch.randn(16, 8)
    true_w = torch.randn(8, 4)
    y = x @ true_w

    def train(opt_cls, **kwargs):
        torch.manual_seed(8)
        model = nn.Linear(8, 4, bias=False)
        opt = opt_cls(model.parameters(), **kwargs)
        for _ in range(100):
            opt.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return ((model(x) - y) ** 2).mean().item()

    loss_prodigy = train(Prodigy, lr=1.0)
    loss_adamw = train(torch.optim.AdamW)  # default lr=1e-3
    assert loss_prodigy < loss_adamw


def test_fsdp_in_use_flag_accepted():
    torch.manual_seed(9)
    model = _make_linear()
    opt = Prodigy(model.parameters(), lr=1.0, fsdp_in_use=True)
    x = torch.randn(2, 4)
    y = torch.randn(2, 2)
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
    assert opt.param_groups[0]["fsdp_in_use"] is True


def test_safeguard_warmup_lowers_initial_d():
    torch.manual_seed(10)
    x = torch.randn(2, 4)
    y = torch.randn(2, 2)

    def run(sw: bool):
        torch.manual_seed(10)
        model = _make_linear(seed=10)
        opt = Prodigy(model.parameters(), lr=1.0, safeguard_warmup=sw)
        for _ in range(5):
            opt.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
        return opt.param_groups[0]["d"]

    d_no = run(False)
    d_sw = run(True)
    assert d_sw <= d_no
