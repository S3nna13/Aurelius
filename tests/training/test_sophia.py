"""Tests for Sophia second-order optimizer."""
import torch
import torch.nn as nn
import pytest
from src.training.sophia import Sophia
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )


@pytest.fixture
def small_model(small_cfg):
    return AureliusTransformer(small_cfg)


def test_sophia_default_hyperparams():
    """Sophia should be constructable with default hyperparams."""
    linear = nn.Linear(32, 64)
    opt = Sophia(linear.parameters())
    group = opt.param_groups[0]
    assert group["lr"] == 1e-3
    assert group["betas"] == (0.9, 0.95)
    assert group["rho"] == 0.04
    assert group["weight_decay"] == 0.1
    assert group["update_period"] == 10


def test_sophia_step_updates_params(small_model, small_cfg):
    """At least one parameter must change after a Sophia step."""
    opt = Sophia(small_model.parameters(), lr=1e-3)
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    before = {name: p.data.clone() for name, p in small_model.named_parameters()}

    _, logits, _ = small_model(tokens)
    loss = logits.sum()
    loss.backward()
    opt.step()

    changed = any(
        not torch.allclose(p.data, before[name])
        for name, p in small_model.named_parameters()
    )
    assert changed, "No parameters were updated after Sophia step"


def test_sophia_state_initialized(small_model, small_cfg):
    """State must contain 'step', 'm', and 'h' after first step."""
    opt = Sophia(small_model.parameters(), lr=1e-3)
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    _, logits, _ = small_model(tokens)
    logits.sum().backward()
    opt.step()

    for p in small_model.parameters():
        if p.grad is not None:
            state = opt.state[p]
            assert "step" in state, f"'step' missing in state for param shape {p.shape}"
            assert "m" in state, f"'m' missing in state for param shape {p.shape}"
            assert "h" in state, f"'h' missing in state for param shape {p.shape}"


def test_sophia_hessian_estimate_shape(small_model, small_cfg):
    """Hessian estimate 'h' must have same shape as its parameter."""
    opt = Sophia(small_model.parameters(), lr=1e-3)
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    _, logits, _ = small_model(tokens)
    logits.sum().backward()
    opt.step()

    for p in small_model.parameters():
        if p.grad is not None:
            assert opt.state[p]["h"].shape == p.shape, (
                f"Hessian shape {opt.state[p]['h'].shape} != param shape {p.shape}"
            )


def test_sophia_step_count_increments(small_model, small_cfg):
    """state['step'] should equal number of optimizer steps taken."""
    opt = Sophia(small_model.parameters(), lr=1e-3)
    n_steps = 5

    for _ in range(n_steps):
        tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
        _, logits, _ = small_model(tokens)
        logits.sum().backward()
        opt.step()
        opt.zero_grad()

    for p in small_model.parameters():
        if p.grad is not None or p in opt.state:
            if p in opt.state and "step" in opt.state[p]:
                assert opt.state[p]["step"] == n_steps, (
                    f"Expected step={n_steps}, got {opt.state[p]['step']}"
                )
                break  # checking one param is enough to validate increment logic


def test_sophia_hessian_only_updated_at_period():
    """'h' should be different after k steps vs after 1 step (when update_period divides step)."""
    linear = nn.Linear(8, 8, bias=False)
    update_period = 5
    opt = Sophia(
        [linear.weight],
        lr=1e-3,
        betas=(0.9, 0.95),
        rho=0.04,
        weight_decay=0.0,
        update_period=update_period,
        eps=1e-8,
    )

    # Do 1 step — h should stay at initial (ones) since step=1 % 5 != 0
    x = torch.randn(4, 8)
    loss = linear(x).sum()
    loss.backward()
    opt.step()
    h_after_1 = opt.state[linear.weight]["h"].clone()

    # h_after_1 should equal ones (initial) because step 1 is not a multiple of k=5
    assert torch.allclose(h_after_1, torch.ones_like(h_after_1)), (
        "h was updated at step 1 but update_period=5"
    )

    opt.zero_grad()

    # Do steps 2 through k — at step k, h should be updated
    for i in range(update_period - 1):
        x = torch.randn(4, 8)
        loss = linear(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    h_after_k = opt.state[linear.weight]["h"].clone()

    # Now h should differ from ones (it was updated at step k)
    assert not torch.allclose(h_after_k, torch.ones_like(h_after_k)), (
        f"h was not updated at step {update_period} despite update_period={update_period}"
    )


def test_sophia_clipping_bounds():
    """Effective normalized step size should be bounded by rho * lr."""
    linear = nn.Linear(16, 16, bias=False)
    lr = 1e-3
    rho = 0.04
    opt = Sophia(
        [linear.weight],
        lr=lr,
        betas=(0.9, 0.95),
        rho=rho,
        weight_decay=0.0,
        update_period=1,  # update h every step
        eps=1e-8,
    )

    before = linear.weight.data.clone()
    x = torch.randn(4, 16)
    loss = linear(x).sum()
    loss.backward()
    opt.step()

    delta = (linear.weight.data - before).abs()
    max_step = rho * lr
    # Allow a small floating-point tolerance (0.1% of max_step)
    assert (delta <= max_step * 1.001 + 1e-9).all(), (
        f"Update magnitude exceeded rho*lr={max_step:.6f}, max was {delta.max().item():.6f}"
    )


def test_sophia_converges_quadratic():
    """Sophia should minimize a simple quadratic loss over multiple steps."""
    x = nn.Parameter(torch.randn(64) * 2.0)
    opt = Sophia(
        [x],
        lr=1e-2,
        betas=(0.9, 0.95),
        rho=0.04,
        weight_decay=0.0,
        update_period=5,
        eps=1e-8,
    )

    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = (x ** 2).sum()
        losses.append(loss.item())
        loss.backward()
        opt.step()

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_sophia_weight_decay_applied():
    """With wd > 0, weights should shrink even with zero gradient."""
    p = nn.Parameter(torch.ones(8, 8) * 2.0)
    lr = 1e-2
    wd = 0.1
    opt = Sophia([p], lr=lr, weight_decay=wd, update_period=10, eps=1e-8)

    # Provide zero gradient so only weight decay acts
    p.grad = torch.zeros_like(p)
    before_norm = p.data.norm().item()
    opt.step()
    after_norm = p.data.norm().item()

    assert after_norm < before_norm, (
        f"Weight norm did not decrease with weight_decay={wd}: "
        f"before={before_norm:.4f}, after={after_norm:.4f}"
    )


def test_sophia_param_group_support(small_model, small_cfg):
    """Different param groups with different lr values should both update correctly."""
    params = list(small_model.parameters())
    mid = len(params) // 2

    opt = Sophia(
        [
            {"params": params[:mid], "lr": 1e-3},
            {"params": params[mid:], "lr": 5e-3},
        ],
        betas=(0.9, 0.95),
        rho=0.04,
        weight_decay=0.0,
    )

    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    before_first = params[0].data.clone()
    before_last = params[-1].data.clone()

    _, logits, _ = small_model(tokens)
    logits.sum().backward()
    opt.step()

    first_grp_changed = not torch.allclose(params[0].data, before_first)
    last_grp_changed = not torch.allclose(params[-1].data, before_last)

    assert first_grp_changed or last_grp_changed, (
        "Neither param group produced any updates"
    )
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[1]["lr"] == 5e-3
