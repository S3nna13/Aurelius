"""Tests for LoRA (Low-Rank Adaptation) implementation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    get_lora_param_count,
    get_lora_params,
    load_lora_weights,
    save_lora_weights,
)

# ---------------------------------------------------------------------------
# Shared constants for all tests
# ---------------------------------------------------------------------------
IN_FEATURES = 16
OUT_FEATURES = 8
R = 4
ALPHA = 8.0
BATCH = 2


@pytest.fixture
def base_linear() -> nn.Linear:
    torch.manual_seed(42)
    return nn.Linear(IN_FEATURES, OUT_FEATURES)


@pytest.fixture
def lora_linear(base_linear: nn.Linear) -> LoRALinear:
    return LoRALinear(base_linear, r=R, alpha=ALPHA)


@pytest.fixture
def simple_model() -> nn.Module:
    """A tiny model with q_proj and v_proj linears to test apply_lora."""
    torch.manual_seed(0)

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(IN_FEATURES, OUT_FEATURES)
            self.v_proj = nn.Linear(IN_FEATURES, OUT_FEATURES)
            self.other = nn.Linear(IN_FEATURES, OUT_FEATURES)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.q_proj(x) + self.v_proj(x) + self.other(x)

    return TinyModel()


@pytest.fixture
def sample_input() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(BATCH, IN_FEATURES)


# ---------------------------------------------------------------------------
# 1. LoRAConfig defaults
# ---------------------------------------------------------------------------
def test_lora_config_defaults() -> None:
    cfg = LoRAConfig()
    assert cfg.r == 8
    assert cfg.alpha == 16.0
    assert cfg.dropout == 0.0
    assert cfg.target_modules == ["q_proj", "v_proj"]
    assert cfg.merge_weights is False


# ---------------------------------------------------------------------------
# 2. LoRALinear output shape matches original linear
# ---------------------------------------------------------------------------
def test_lora_linear_output_shape(lora_linear: LoRALinear, sample_input: torch.Tensor) -> None:
    out = lora_linear(sample_input)
    assert out.shape == (BATCH, OUT_FEATURES)


# ---------------------------------------------------------------------------
# 3. LoRALinear original weight frozen (no grad)
# ---------------------------------------------------------------------------
def test_lora_linear_original_weight_frozen(lora_linear: LoRALinear) -> None:
    assert not lora_linear.linear.weight.requires_grad


# ---------------------------------------------------------------------------
# 4. LoRALinear lora_A and lora_B require grad
# ---------------------------------------------------------------------------
def test_lora_linear_lora_params_require_grad(lora_linear: LoRALinear) -> None:
    assert lora_linear.lora_A.requires_grad
    assert lora_linear.lora_B.requires_grad


# ---------------------------------------------------------------------------
# 5. LoRALinear init: lora_B is zero → first forward = same as frozen linear
# ---------------------------------------------------------------------------
def test_lora_linear_zero_init_matches_base(
    base_linear: nn.Linear, sample_input: torch.Tensor
) -> None:
    # Store base output before wrapping (lora_B = 0 so LoRA contributes nothing)
    base_out = base_linear(sample_input).detach()

    lora = LoRALinear(base_linear, r=R, alpha=ALPHA)
    # Ensure lora_B is indeed zeroed
    assert torch.all(lora.lora_B == 0)

    lora_out = lora(sample_input).detach()
    assert torch.allclose(base_out, lora_out, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. LoRALinear merge returns nn.Linear with same output
# ---------------------------------------------------------------------------
def test_lora_linear_merge_same_output(lora_linear: LoRALinear, sample_input: torch.Tensor) -> None:
    # Give lora_B non-zero values to make merge meaningful
    with torch.no_grad():
        lora_linear.lora_B.fill_(0.01)

    lora_out = lora_linear(sample_input).detach()
    merged = lora_linear.merge()

    assert isinstance(merged, nn.Linear)
    merged_out = merged(sample_input).detach()
    assert torch.allclose(lora_out, merged_out, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. LoRALinear forward gradient flows to lora_A and lora_B
# ---------------------------------------------------------------------------
def test_lora_linear_gradients_flow(lora_linear: LoRALinear, sample_input: torch.Tensor) -> None:
    out = lora_linear(sample_input)
    loss = out.sum()
    loss.backward()

    assert lora_linear.lora_A.grad is not None
    assert lora_linear.lora_B.grad is not None
    # Original weight should have no gradient
    assert lora_linear.linear.weight.grad is None


# ---------------------------------------------------------------------------
# 8. apply_lora returns correct count
# ---------------------------------------------------------------------------
def test_apply_lora_returns_correct_count(simple_model: nn.Module) -> None:
    cfg = LoRAConfig(r=R, alpha=ALPHA, target_modules=["q_proj", "v_proj"])
    count = apply_lora(simple_model, cfg)
    assert count == 2


# ---------------------------------------------------------------------------
# 9. apply_lora replaced modules are LoRALinear
# ---------------------------------------------------------------------------
def test_apply_lora_replaced_modules_are_lora_linear(simple_model: nn.Module) -> None:
    cfg = LoRAConfig(r=R, alpha=ALPHA, target_modules=["q_proj", "v_proj"])
    apply_lora(simple_model, cfg)

    assert isinstance(simple_model.q_proj, LoRALinear)
    assert isinstance(simple_model.v_proj, LoRALinear)
    # 'other' should remain a plain nn.Linear
    assert isinstance(simple_model.other, nn.Linear)
    assert not isinstance(simple_model.other, LoRALinear)


# ---------------------------------------------------------------------------
# 10. get_lora_params returns only trainable params
# ---------------------------------------------------------------------------
def test_get_lora_params_only_trainable(simple_model: nn.Module) -> None:
    cfg = LoRAConfig(r=R, alpha=ALPHA, target_modules=["q_proj", "v_proj"])
    apply_lora(simple_model, cfg)

    params = get_lora_params(simple_model)
    assert len(params) > 0
    # All returned params must be LoRA matrices (lora_A or lora_B)
    lora_param_ids = set()
    for name, p in simple_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_param_ids.add(id(p))
    for p in params:
        assert id(p) in lora_param_ids


# ---------------------------------------------------------------------------
# 11. get_lora_param_count keys present, trainable < total
# ---------------------------------------------------------------------------
def test_get_lora_param_count_keys_and_values(simple_model: nn.Module) -> None:
    cfg = LoRAConfig(r=R, alpha=ALPHA, target_modules=["q_proj", "v_proj"])
    apply_lora(simple_model, cfg)

    counts = get_lora_param_count(simple_model)
    assert "trainable" in counts
    assert "frozen" in counts
    assert "total" in counts
    assert counts["trainable"] < counts["total"]
    assert counts["trainable"] + counts["frozen"] == counts["total"]


# ---------------------------------------------------------------------------
# 12. save_lora_weights returns dict with lora params only
# ---------------------------------------------------------------------------
def test_save_lora_weights_lora_only(simple_model: nn.Module) -> None:
    cfg = LoRAConfig(r=R, alpha=ALPHA, target_modules=["q_proj", "v_proj"])
    apply_lora(simple_model, cfg)

    weights = save_lora_weights(simple_model)
    assert len(weights) > 0
    for name in weights:
        assert "lora_A" in name or "lora_B" in name


# ---------------------------------------------------------------------------
# 13. load_lora_weights restores weights correctly
# ---------------------------------------------------------------------------
def test_load_lora_weights_restores(simple_model: nn.Module) -> None:
    cfg = LoRAConfig(r=R, alpha=ALPHA, target_modules=["q_proj", "v_proj"])
    apply_lora(simple_model, cfg)

    # Modify lora weights
    with torch.no_grad():
        simple_model.q_proj.lora_A.fill_(99.0)
        simple_model.q_proj.lora_B.fill_(99.0)

    # Save then restore to known values
    original_weights = {
        "q_proj.lora_A": torch.ones(R, IN_FEATURES) * 3.0,
        "q_proj.lora_B": torch.ones(OUT_FEATURES, R) * 5.0,
    }
    load_lora_weights(simple_model, original_weights)

    assert torch.allclose(simple_model.q_proj.lora_A, torch.ones(R, IN_FEATURES) * 3.0)
    assert torch.allclose(simple_model.q_proj.lora_B, torch.ones(OUT_FEATURES, R) * 5.0)


# ---------------------------------------------------------------------------
# 14. LoRALinear scale property = alpha / r
# ---------------------------------------------------------------------------
def test_lora_linear_scale_property(lora_linear: LoRALinear) -> None:
    assert lora_linear.scale == pytest.approx(ALPHA / R)


# ---------------------------------------------------------------------------
# 15. apply_lora with r=4 gives fewer trainable params than r=8
# ---------------------------------------------------------------------------
def test_apply_lora_smaller_r_fewer_params() -> None:
    torch.manual_seed(0)

    def make_model() -> nn.Module:
        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(IN_FEATURES, OUT_FEATURES)
                self.v_proj = nn.Linear(IN_FEATURES, OUT_FEATURES)

        return TinyModel()

    model_r4 = make_model()
    model_r8 = make_model()

    apply_lora(model_r4, LoRAConfig(r=4, alpha=ALPHA, target_modules=["q_proj", "v_proj"]))
    apply_lora(model_r8, LoRAConfig(r=8, alpha=ALPHA, target_modules=["q_proj", "v_proj"]))

    count_r4 = get_lora_param_count(model_r4)["trainable"]
    count_r8 = get_lora_param_count(model_r8)["trainable"]

    assert count_r4 < count_r8


# ---------------------------------------------------------------------------
# 16. get_lora_params count matches r * (in_features + out_features) * n_replaced
# ---------------------------------------------------------------------------
def test_get_lora_params_count_matches_formula(simple_model: nn.Module) -> None:
    cfg = LoRAConfig(r=R, alpha=ALPHA, target_modules=["q_proj", "v_proj"])
    n_replaced = apply_lora(simple_model, cfg)

    # Each LoRALinear has: lora_A (r x in) + lora_B (out x r)
    expected_elements = n_replaced * R * (IN_FEATURES + OUT_FEATURES)
    counts = get_lora_param_count(simple_model)

    assert counts["trainable"] == expected_elements
