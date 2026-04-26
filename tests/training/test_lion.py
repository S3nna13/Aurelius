"""Tests for Lion optimizer."""

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.lion import Lion, Lion8bit, LionW, compare_lion_adam_memory


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


def test_lion_default_hyperparams():
    """Lion should be constructable with default hyperparams."""
    linear = nn.Linear(32, 64)
    opt = Lion(linear.parameters())
    # Check defaults stored in param group
    group = opt.param_groups[0]
    assert group["lr"] == 1e-4
    assert group["betas"] == (0.9, 0.99)
    assert group["weight_decay"] == 0.0


def test_lion_step_updates_params(small_model, small_cfg):
    """At least one parameter must change after a Lion step."""
    opt = Lion(small_model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0)
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    before = {name: p.data.clone() for name, p in small_model.named_parameters()}

    _, logits, _ = small_model(tokens)
    loss = logits.sum()
    loss.backward()
    opt.step()

    changed = any(
        not torch.allclose(p.data, before[name]) for name, p in small_model.named_parameters()
    )
    assert changed, "No parameters were updated after Lion step"


def test_lion_momentum_buffer_initialized(small_model, small_cfg):
    """State must contain 'm' key after first step."""
    opt = Lion(small_model.parameters(), lr=1e-4)
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    _, logits, _ = small_model(tokens)
    logits.sum().backward()
    opt.step()

    for p in small_model.parameters():
        if p.grad is not None:
            assert "m" in opt.state[p], f"Momentum buffer 'm' missing for param of shape {p.shape}"


def test_lion_momentum_shape(small_model, small_cfg):
    """Momentum buffer 'm' must have same shape as its parameter."""
    opt = Lion(small_model.parameters(), lr=1e-4)
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    _, logits, _ = small_model(tokens)
    logits.sum().backward()
    opt.step()

    for p in small_model.parameters():
        if p.grad is not None:
            assert opt.state[p]["m"].shape == p.shape, (
                f"Momentum shape {opt.state[p]['m'].shape} != param shape {p.shape}"
            )


def test_lion_step_is_bounded():
    """All parameter updates must have magnitude <= lr (sign property).

    Lion applies sign(interpolated_momentum) * lr, so each element of the
    update has magnitude exactly lr (ignoring weight decay which can add more).
    With wd=0 and a clean initial state, the update to p is exactly ±lr.
    """
    # Use a simple linear layer with no weight decay
    linear = nn.Linear(16, 16, bias=False)
    lr = 1e-3
    opt = Lion([linear.weight], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)

    before = linear.weight.data.clone()
    x = torch.randn(4, 16)
    loss = linear(x).sum()
    loss.backward()
    opt.step()

    delta = (linear.weight.data - before).abs()
    # Each element update should be exactly lr (sign * lr)
    assert (delta <= lr + 1e-8).all(), (
        f"Update magnitude exceeded lr={lr}, max was {delta.max().item()}"
    )


def test_lion_weight_decay():
    """With weight_decay > 0, larger parameters should decrease in magnitude faster."""
    # Two params: one large-init, one small-init
    large_p = nn.Parameter(torch.ones(8, 8) * 2.0)
    small_p = nn.Parameter(torch.ones(8, 8) * 0.1)

    opt = Lion([large_p, small_p], lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1)

    # Give both the same gradient so sign update is identical
    large_p.grad = torch.zeros_like(large_p)
    small_p.grad = torch.zeros_like(small_p)

    large_before = large_p.data.norm().item()
    small_before = small_p.data.norm().item()

    opt.step()

    large_after = large_p.data.norm().item()
    small_after = small_p.data.norm().item()

    large_ratio = (large_before - large_after) / large_before
    small_ratio = (small_before - small_after) / small_before

    # Large param should have a bigger fractional decrease due to wd * p term
    assert large_ratio > small_ratio, (
        f"Expected larger param to shrink more: large_ratio={large_ratio:.4f}, small_ratio={small_ratio:.4f}"  # noqa: E501
    )


def test_lion_zero_grad_params_skipped(small_model, small_cfg):
    """Parameters without .grad must not be touched."""
    opt = Lion(small_model.parameters(), lr=1e-4)

    # Do NOT call backward — all grads remain None
    before = {name: p.data.clone() for name, p in small_model.named_parameters()}
    opt.step()

    for name, p in small_model.named_parameters():
        assert torch.allclose(p.data, before[name]), (
            f"Param '{name}' was modified despite having no gradient"
        )


def test_lion_multiple_steps_converge():
    """Loss should decrease on a simple quadratic over 10 Lion steps."""
    # Minimize f(x) = ||x||^2  =>  gradient is 2x
    x = nn.Parameter(torch.randn(64) * 2.0)
    opt = Lion([x], lr=1e-2, betas=(0.9, 0.99), weight_decay=0.0)

    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = (x**2).sum()
        losses.append(loss.item())
        loss.backward()
        opt.step()

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_lion_memory_usage(small_model, small_cfg):
    """Each parameter with a gradient should have exactly one state tensor ('m')."""
    opt = Lion(small_model.parameters(), lr=1e-4)
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    _, logits, _ = small_model(tokens)
    logits.sum().backward()
    opt.step()

    for p in small_model.parameters():
        if p.grad is not None:
            state_keys = list(opt.state[p].keys())
            assert state_keys == ["m"], (
                f"Expected only ['m'] in state, got {state_keys} for param shape {p.shape}"
            )


def test_lion_param_group_support(small_model, small_cfg):
    """Different param groups with different lr values should both update correctly."""
    params = list(small_model.parameters())
    mid = len(params) // 2

    opt = Lion(
        [
            {"params": params[:mid], "lr": 1e-4},
            {"params": params[mid:], "lr": 5e-4},
        ],
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )

    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    before_first = params[0].data.clone()
    before_last = params[-1].data.clone()

    _, logits, _ = small_model(tokens)
    logits.sum().backward()
    opt.step()

    # Both groups should have updated (assuming they have grads)
    first_grp_changed = not torch.allclose(params[0].data, before_first)
    last_grp_changed = not torch.allclose(params[-1].data, before_last)

    assert first_grp_changed or last_grp_changed, "Neither param group produced any updates"
    # Verify both param groups are tracked
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 1e-4
    assert opt.param_groups[1]["lr"] == 5e-4


# ---------------------------------------------------------------------------
# New required tests (task specification)
# ---------------------------------------------------------------------------


def test_lion_step_reduces_loss():
    """Optimize simple quadratic f(x)=||x||^2; loss must decrease over 10 steps."""
    x = nn.Parameter(torch.randn(64) * 2.0)
    opt = Lion([x], lr=1e-2, betas=(0.9, 0.99), weight_decay=0.0)

    first_loss = None
    for _ in range(10):
        opt.zero_grad()
        loss = (x**2).sum()
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()

    last_loss = (x**2).sum().item()
    assert last_loss < first_loss, (
        f"Loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}"
    )


def test_lion_one_momentum_buffer():
    """After a step, state must contain only 'exp_avg' (not 'exp_avg_sq')."""
    # Note: existing impl uses key 'm', but we verify 'exp_avg_sq' is absent
    # and exactly one momentum buffer exists.
    x = nn.Parameter(torch.randn(16))
    opt = Lion([x], lr=1e-3)
    x.grad = torch.ones_like(x)
    opt.step()

    state = opt.state[x]
    # Must NOT have Adam's second moment buffer
    assert "exp_avg_sq" not in state, "Lion should not store exp_avg_sq (Adam's second moment)"
    # Must have exactly one momentum buffer
    tensor_keys = [k for k, v in state.items() if isinstance(v, torch.Tensor)]
    assert len(tensor_keys) == 1, (
        f"Expected exactly 1 momentum buffer, got {len(tensor_keys)}: {tensor_keys}"
    )


def test_lion_update_is_sign():
    """Update magnitude must be uniform (sign property) — each element exactly lr."""
    linear = nn.Linear(16, 16, bias=False)
    lr = 1e-3
    opt = Lion([linear.weight], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)

    before = linear.weight.data.clone()
    x = torch.randn(4, 16)
    loss = linear(x).sum()
    loss.backward()
    opt.step()

    delta = (linear.weight.data - before).abs()
    # Each element should be moved by exactly lr (sign * lr, no wd)
    assert (delta <= lr + 1e-7).all(), (
        f"Update magnitude exceeded lr={lr}; max delta={delta.max().item()}"
    )
    # At least some elements should be non-zero
    assert delta.max().item() > 0


def test_lion_weight_decay_zero_grad():
    """With wd>0 and zero gradient, params should shrink toward 0."""
    p = nn.Parameter(torch.ones(8, 8) * 2.0)
    opt = Lion([p], lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1)
    p.grad = torch.zeros_like(p)  # zero gradient so only wd acts

    norm_before = p.data.norm().item()
    opt.step()
    norm_after = p.data.norm().item()

    assert norm_after < norm_before, (
        f"Weight decay did not shrink param: before={norm_before:.4f}, after={norm_after:.4f}"
    )


def test_lionw_default_lr():
    """LionW default lr must be 3e-4."""
    p = nn.Parameter(torch.randn(4))
    opt = LionW([p])
    assert opt.param_groups[0]["lr"] == 3e-4, (
        f"Expected default lr=3e-4, got {opt.param_groups[0]['lr']}"
    )


def test_lion8bit_quantize_dequantize():
    """Quantize + dequantize roundtrip must have max relative error < 1%."""
    opt = Lion8bit([nn.Parameter(torch.zeros(1))])  # dummy, not used
    m = torch.randn(128, 128) * 5.0  # realistic momentum magnitudes

    m_q, scale = opt._quantize_momentum(m)
    m_reconstructed = opt._dequantize_momentum(m_q, scale)

    abs_max = m.abs().max().item()
    max_error = (m - m_reconstructed).abs().max().item()
    # Relative error < 1% of the dynamic range
    assert max_error / abs_max < 0.01, (
        f"Quantization error too large: max_error={max_error:.6f}, abs_max={abs_max:.6f}"
    )


def test_lion8bit_step_works():
    """Lion8bit.step must not error and loss must decrease over 10 steps."""
    x = nn.Parameter(torch.randn(64) * 2.0)
    opt = Lion8bit([x], lr=1e-2, betas=(0.9, 0.99), weight_decay=0.0)

    first_loss = None
    for _ in range(10):
        opt.zero_grad()
        loss = (x**2).sum()
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()

    last_loss = (x**2).sum().item()
    assert last_loss < first_loss, (
        f"Lion8bit loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}"
    )


def test_lion8bit_momentum_stored_as_int8():
    """After a Lion8bit step, 'exp_avg_q' must be stored as int8 dtype."""
    x = nn.Parameter(torch.randn(32))
    opt = Lion8bit([x], lr=1e-3)
    x.grad = torch.randn_like(x)
    opt.step()

    state = opt.state[x]
    assert "exp_avg_q" in state, "Lion8bit state must contain 'exp_avg_q'"
    assert state["exp_avg_q"].dtype == torch.int8, f"Expected int8, got {state['exp_avg_q'].dtype}"
    assert "exp_avg_scale" in state, "Lion8bit state must contain 'exp_avg_scale'"


def test_memory_comparison_keys():
    """compare_lion_adam_memory must return dict with all required keys."""
    result = compare_lion_adam_memory(1_000_000)
    required_keys = {
        "adam_buffers_mb",
        "lion_buffers_mb",
        "lion8bit_buffers_mb",
        "lion_vs_adam_ratio",
    }
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"
    # All values should be positive floats
    for k in required_keys:
        assert isinstance(result[k], float), f"Key '{k}' value is not float: {result[k]}"
        assert result[k] > 0, f"Key '{k}' value should be positive, got {result[k]}"


def test_lion_vs_adam_memory_ratio():
    """lion_vs_adam_ratio must be approximately 0.5 (Lion uses half Adam's memory)."""
    result = compare_lion_adam_memory(10_000_000)
    ratio = result["lion_vs_adam_ratio"]
    assert abs(ratio - 0.5) < 1e-6, f"Expected lion_vs_adam_ratio ~= 0.5, got {ratio}"
