"""Tests for GrokFast gradient amplification (≥12 tests).

All imports use the stable ``aurelius.*`` namespace which aliases to ``src.*``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from aurelius.training.grokfast import GrokFastEMA, GrokFastOptimizer, GrokFastSMA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model() -> nn.Linear:
    """A tiny single-layer model for fast tests."""
    torch.manual_seed(0)
    return nn.Linear(4, 4, bias=False)


def _set_grad(model: nn.Module, value: float = 1.0) -> None:
    """Manually assign a constant gradient to every parameter."""
    for param in model.parameters():
        param.grad = torch.full_like(param.data, value)


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.data.norm().item() ** 2
    return total**0.5


# ---------------------------------------------------------------------------
# 1. GrokFastEMA initialises with zero / empty EMA state
# ---------------------------------------------------------------------------


def test_ema_init_state_is_empty():
    model = _tiny_model()
    gf = GrokFastEMA(model, alpha=0.98, lamb=2.0)
    assert len(gf._ema) == 0, "EMA map must be empty before any update()"


# ---------------------------------------------------------------------------
# 2. After update(), EMA is non-zero when grad is non-zero
# ---------------------------------------------------------------------------


def test_ema_update_populates_nonzero():
    model = _tiny_model()
    _set_grad(model, 1.0)
    gf = GrokFastEMA(model, alpha=0.98, lamb=2.0)
    gf.update()

    assert len(gf._ema) > 0, "EMA map must be non-empty after update()"
    for ema_val in gf._ema.values():
        assert ema_val.abs().sum().item() > 0.0, "EMA should be non-zero when grad is non-zero"


# ---------------------------------------------------------------------------
# 3. amplify() increases gradient magnitude (lamb=2 → |grad| larger)
# ---------------------------------------------------------------------------


def test_amplify_increases_grad_magnitude():
    model = _tiny_model()
    _set_grad(model, 1.0)
    gf = GrokFastEMA(model, alpha=0.98, lamb=2.0)

    before = _grad_norm(model)
    gf.update()
    gf.amplify()
    after = _grad_norm(model)

    assert after > before, (
        f"amplify() should increase gradient magnitude: before={before:.4f} after={after:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. step() == update() + amplify() in sequence
# ---------------------------------------------------------------------------


def test_step_equals_update_then_amplify():
    """step() on model_a should produce identical grads to manual update+amplify on model_b."""
    torch.manual_seed(1)
    model_a = nn.Linear(4, 4, bias=False)
    model_b = nn.Linear(4, 4, bias=False)
    # Give both models the same gradient
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        grad = torch.randn_like(pa.data)
        pa.grad = grad.clone()
        pb.grad = grad.clone()

    gf_a = GrokFastEMA(model_a, alpha=0.95, lamb=1.5)
    gf_b = GrokFastEMA(model_b, alpha=0.95, lamb=1.5)

    gf_a.step()
    gf_b.update()
    gf_b.amplify()

    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa.grad, pb.grad), "step() must equal update()+amplify()"


# ---------------------------------------------------------------------------
# 5. state_dict round-trip preserves EMA values
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip():
    model = _tiny_model()
    _set_grad(model, 0.5)
    gf = GrokFastEMA(model, alpha=0.9, lamb=3.0)
    gf.update()

    sd = gf.state_dict()

    # Create a fresh instance and restore
    gf2 = GrokFastEMA(model, alpha=0.0, lamb=0.0)
    gf2.load_state_dict(sd)

    assert gf2.alpha == 0.9
    assert gf2.lamb == 3.0
    assert len(gf2._ema) == len(gf._ema)

    # Check all EMA tensors are equal
    for pid_orig, ema_orig in gf._ema.items():
        # Find matching entry by comparing values
        matched = any(torch.allclose(ema_orig, ema_new) for ema_new in gf2._ema.values())
        assert matched, "EMA tensor mismatch after load_state_dict"


# ---------------------------------------------------------------------------
# 6. GrokFastSMA with window=1 behaves like a single-step EMA
# ---------------------------------------------------------------------------


def test_sma_window1_single_grad():
    model = _tiny_model()
    _set_grad(model, 1.0)
    gf = GrokFastSMA(model, window=1, lamb=2.0)
    gf.update()

    # With window=1 the SMA is just the current gradient
    for pid, buf in gf._buffers.items():
        assert len(buf) == 1
        assert torch.allclose(buf[0], torch.ones_like(buf[0]))


# ---------------------------------------------------------------------------
# 7. SMA with window=3 averages last 3 gradients
# ---------------------------------------------------------------------------


def test_sma_window3_averages_three_grads():
    model = nn.Linear(2, 2, bias=False)
    gf = GrokFastSMA(model, window=3, lamb=0.0)  # lamb=0 → no amplification

    grad_values = [1.0, 2.0, 3.0]
    for v in grad_values:
        _set_grad(model, v)
        gf.update()

    # Each buffer should have exactly 3 entries
    for pid, buf in gf._buffers.items():
        assert len(buf) == 3
        sma = torch.stack(list(buf)).mean(dim=0)
        expected_mean = sum(grad_values) / len(grad_values)
        assert torch.allclose(sma, torch.full_like(sma, expected_mean), atol=1e-5), (
            f"SMA mean {sma.mean().item():.4f} != expected {expected_mean}"
        )


# ---------------------------------------------------------------------------
# 8. GrokFastOptimizer.step() updates model parameters
# ---------------------------------------------------------------------------


def test_grokfast_optimizer_step_updates_params():
    model = _tiny_model()
    inner_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    gf_opt = GrokFastOptimizer(inner_opt, model, alpha=0.98, lamb=2.0)

    params_before = [p.data.clone() for p in model.parameters()]
    _set_grad(model, 1.0)
    gf_opt.step()
    params_after = [p.data.clone() for p in model.parameters()]

    changed = any(not torch.allclose(pb, pa) for pb, pa in zip(params_before, params_after))
    assert changed, "GrokFastOptimizer.step() must update model parameters"


# ---------------------------------------------------------------------------
# 9. GrokFastOptimizer.zero_grad() zeros all parameter gradients
# ---------------------------------------------------------------------------


def test_grokfast_optimizer_zero_grad():
    model = _tiny_model()
    inner_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    gf_opt = GrokFastOptimizer(inner_opt, model, alpha=0.98, lamb=2.0)

    _set_grad(model, 5.0)
    gf_opt.zero_grad()

    for param in model.parameters():
        if param.grad is not None:
            assert param.grad.abs().sum().item() == 0.0, "zero_grad() must zero all gradients"


# ---------------------------------------------------------------------------
# 10. param_groups is accessible via GrokFastOptimizer
# ---------------------------------------------------------------------------


def test_grokfast_optimizer_param_groups():
    model = _tiny_model()
    inner_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gf_opt = GrokFastOptimizer(inner_opt, model)

    pg = gf_opt.param_groups
    assert isinstance(pg, list) and len(pg) > 0, "param_groups must be a non-empty list"
    assert "lr" in pg[0], "param_groups[0] must contain 'lr'"


# ---------------------------------------------------------------------------
# 11. Works with SGD and Adam inner optimizers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "optim_cls,kwargs",
    [
        (torch.optim.SGD, {"lr": 0.01}),
        (torch.optim.Adam, {"lr": 1e-3}),
    ],
)
def test_grokfast_optimizer_sgd_and_adam(optim_cls, kwargs):
    model = _tiny_model()
    inner_opt = optim_cls(model.parameters(), **kwargs)
    gf_opt = GrokFastOptimizer(inner_opt, model, alpha=0.98, lamb=2.0)

    _set_grad(model, 1.0)
    # Must not raise
    gf_opt.step()


# ---------------------------------------------------------------------------
# 12. EMA decays toward zero after many zero-gradient steps
# ---------------------------------------------------------------------------


def test_ema_decays_toward_zero():
    """After initialising EMA with a non-zero gradient and then feeding zero
    gradients for many steps, the EMA should shrink toward zero."""
    model = nn.Linear(2, 2, bias=False)
    gf = GrokFastEMA(model, alpha=0.9, lamb=0.0)  # lamb=0 → no amplification side-effect

    # One warm-up step with a large gradient
    _set_grad(model, 1.0)
    gf.update()

    # Store initial EMA norm
    initial_norm = sum(v.norm().item() ** 2 for v in gf._ema.values()) ** 0.5

    # Many steps with zero gradient
    _set_grad(model, 0.0)
    for _ in range(200):
        gf.update()

    final_norm = sum(v.norm().item() ** 2 for v in gf._ema.values()) ** 0.5

    assert final_norm < initial_norm * 0.01, (
        f"EMA should decay to near-zero: initial={initial_norm:.4f} final={final_norm:.6f}"
    )


# ---------------------------------------------------------------------------
# 13. SMA window respects maxlen — old grads are evicted
# ---------------------------------------------------------------------------


def test_sma_window_evicts_old_entries():
    model = nn.Linear(2, 2, bias=False)
    gf = GrokFastSMA(model, window=2, lamb=0.0)

    for v in [1.0, 2.0, 3.0, 4.0]:  # 4 steps but window=2
        _set_grad(model, v)
        gf.update()

    for pid, buf in gf._buffers.items():
        assert len(buf) == 2, f"Buffer should have exactly window=2 entries, got {len(buf)}"
        # Should contain grads from steps 3.0 and 4.0
        vals = torch.stack(list(buf))
        assert torch.allclose(vals, torch.full_like(vals, 3.0)) or torch.allclose(
            vals.mean(0), torch.full_like(vals.mean(0), 3.5)
        ), "Buffer should contain the last 2 gradient values"


# ---------------------------------------------------------------------------
# 14. GrokFastEMA params with no grad are not added to EMA map
# ---------------------------------------------------------------------------


def test_ema_skips_params_without_grad():
    model = _tiny_model()
    # Do NOT call backward or set grads → all param.grad is None
    gf = GrokFastEMA(model, alpha=0.98, lamb=2.0)
    gf.update()
    assert len(gf._ema) == 0, "update() must skip parameters without .grad"
