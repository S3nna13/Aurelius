"""Tests for GrokFast gradient amplification."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.grokfast import GrokFastConfig, GrokFastEMA, GrokFastMA, apply_grokfast
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


def _make_model() -> AureliusTransformer:
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )
    return AureliusTransformer(cfg)


def _run_backward(model: nn.Module) -> None:
    """Do a simple forward+backward to populate .grad on all parameters."""
    B, S = 2, 16
    input_ids = torch.randint(0, 256, (B, S))
    labels = torch.randint(0, 256, (B, S))
    loss, _logits, _kv = model(input_ids, labels=labels)
    loss.backward()


# ---------------------------------------------------------------------------
# 1. test_grokfast_config_defaults
# ---------------------------------------------------------------------------

def test_grokfast_config_defaults():
    cfg = GrokFastConfig()
    assert cfg.alpha == 0.98
    assert cfg.lamb == 2.0


# ---------------------------------------------------------------------------
# 2. test_grokfast_ema_apply_modifies_gradients
# ---------------------------------------------------------------------------

def test_grokfast_ema_apply_modifies_gradients():
    model = _make_model()
    _run_backward(model)

    # Capture original gradients
    orig_grads = {
        name: param.grad.data.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    grokfast = GrokFastEMA(model)
    grokfast.apply()

    # At least one gradient must have changed
    changed = False
    for name, param in model.named_parameters():
        if param.grad is not None and name in orig_grads:
            if not torch.allclose(param.grad.data, orig_grads[name]):
                changed = True
                break
    assert changed, "apply() must modify at least one gradient"


# ---------------------------------------------------------------------------
# 3. test_grokfast_ema_amplifies_slow_component
# ---------------------------------------------------------------------------

def test_grokfast_ema_amplifies_slow_component():
    """Gradient norm must grow after apply() when lamb > 0."""
    model = _make_model()
    _run_backward(model)

    # Compute original total gradient norm
    orig_norm = sum(
        p.grad.data.norm().item() ** 2
        for p in model.parameters()
        if p.grad is not None
    ) ** 0.5

    grokfast = GrokFastEMA(model, GrokFastConfig(alpha=0.98, lamb=2.0))
    grokfast.apply()

    new_norm = sum(
        p.grad.data.norm().item() ** 2
        for p in model.parameters()
        if p.grad is not None
    ) ** 0.5

    assert new_norm > orig_norm, (
        f"Amplified norm {new_norm:.4f} should exceed original {orig_norm:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. test_grokfast_ema_state_initialized
# ---------------------------------------------------------------------------

def test_grokfast_ema_state_initialized():
    model = _make_model()
    _run_backward(model)

    grokfast = GrokFastEMA(model)
    assert len(grokfast._ema) == 0, "_ema should be empty before first apply()"

    grokfast.apply()

    assert len(grokfast._ema) > 0, "_ema should be populated after apply()"
    # All keys should match parameter names that have gradients
    param_names_with_grad = {
        name for name, p in model.named_parameters() if p.grad is not None
    }
    assert set(grokfast._ema.keys()) == param_names_with_grad


# ---------------------------------------------------------------------------
# 5. test_grokfast_ema_reset_clears_state
# ---------------------------------------------------------------------------

def test_grokfast_ema_reset_clears_state():
    model = _make_model()
    _run_backward(model)

    grokfast = GrokFastEMA(model)
    grokfast.apply()
    assert len(grokfast._ema) > 0

    grokfast.reset()
    assert len(grokfast._ema) == 0, "_ema must be empty after reset()"


# ---------------------------------------------------------------------------
# 6. test_grokfast_ma_apply_runs
# ---------------------------------------------------------------------------

def test_grokfast_ma_apply_runs():
    model = _make_model()
    _run_backward(model)

    grokfast = GrokFastMA(model)
    # Should not raise
    grokfast.apply()


# ---------------------------------------------------------------------------
# 7. test_grokfast_ma_windows_populated
# ---------------------------------------------------------------------------

def test_grokfast_ma_windows_populated():
    model = _make_model()
    _run_backward(model)

    grokfast = GrokFastMA(model)
    assert len(grokfast._windows) == 0

    grokfast.apply()

    assert len(grokfast._windows) > 0, "_windows should be populated after apply()"
    # Each window should have exactly one entry after one call
    for name, window in grokfast._windows.items():
        assert len(window) == 1, f"Window for {name} should have 1 entry, got {len(window)}"


# ---------------------------------------------------------------------------
# 8. test_apply_grokfast_functional
# ---------------------------------------------------------------------------

def test_apply_grokfast_functional():
    model = _make_model()
    _run_backward(model)

    ema_state: dict[str, torch.Tensor] = {}
    returned = apply_grokfast(model, ema_state, alpha=0.98, lamb=2.0)

    # Must return the updated dict
    assert returned is ema_state, "apply_grokfast must return the same ema_state dict"
    assert len(ema_state) > 0, "ema_state should be populated after call"


# ---------------------------------------------------------------------------
# 9. test_apply_grokfast_updates_state
# ---------------------------------------------------------------------------

def test_apply_grokfast_updates_state():
    model = _make_model()
    _run_backward(model)

    ema_state: dict[str, torch.Tensor] = {}
    apply_grokfast(model, ema_state)

    param_names_with_grad = {
        name for name, p in model.named_parameters() if p.grad is not None
    }
    assert set(ema_state.keys()) == param_names_with_grad, (
        "ema_state keys must match parameter names that have gradients"
    )
    # Each EMA tensor must have same shape as corresponding parameter
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert ema_state[name].shape == p.data.shape, (
                f"EMA shape mismatch for {name}: "
                f"{ema_state[name].shape} vs {p.data.shape}"
            )


# ---------------------------------------------------------------------------
# 10. test_grokfast_no_grad_params_skipped
# ---------------------------------------------------------------------------

def test_grokfast_no_grad_params_skipped():
    """Parameters without .grad must not be touched (no entry in _ema)."""
    model = _make_model()
    # Do NOT call backward — no grads set

    grokfast = GrokFastEMA(model)
    grokfast.apply()  # Should not crash

    assert len(grokfast._ema) == 0, (
        "No parameter has .grad, so _ema should remain empty"
    )
