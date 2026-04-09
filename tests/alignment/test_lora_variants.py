"""Tests for LoRA+, IA3, and VeRA efficient fine-tuning adapters."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.alignment.lora_variants import (
    LoRAVariantConfig,
    LoRAPlusAdapter,
    IA3Adapter,
    VeRAAdapter,
    get_adapter_params,
    apply_adapters_to_model,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IN_FEATURES = 64
OUT_FEATURES = 64


def small_model() -> AureliusTransformer:
    """Minimal 2-layer Aurelius model for fast tests."""
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def default_config(variant: str = "lora_plus") -> LoRAVariantConfig:
    return LoRAVariantConfig(rank=8, alpha=16.0, variant=variant)


# ---------------------------------------------------------------------------
# 1. LoRAVariantConfig defaults
# ---------------------------------------------------------------------------

def test_lora_variant_config_defaults():
    cfg = LoRAVariantConfig()
    assert cfg.rank == 8
    assert cfg.alpha == 16.0
    assert cfg.dropout == 0.0
    assert cfg.variant == "lora_plus"
    assert cfg.lora_plus_lr_ratio == 16.0
    assert cfg.vera_shared_dim == 256


# ---------------------------------------------------------------------------
# 2. LoRAPlusAdapter forward output shape
# ---------------------------------------------------------------------------

def test_loraplus_forward_output_shape():
    cfg = default_config("lora_plus")
    adapter = LoRAPlusAdapter(IN_FEATURES, OUT_FEATURES, cfg)
    x = torch.randn(2, 10, IN_FEATURES)
    out = adapter(x)
    assert out.shape == (2, 10, OUT_FEATURES)


# ---------------------------------------------------------------------------
# 3. LoRAPlusAdapter lora_B initialized to zeros → zero output before training
# ---------------------------------------------------------------------------

def test_loraplus_lora_B_initialized_to_zeros():
    cfg = default_config("lora_plus")
    adapter = LoRAPlusAdapter(IN_FEATURES, OUT_FEATURES, cfg)
    assert torch.all(adapter.lora_B == 0), "lora_B must be zero-initialized"

    # Zero lora_B means forward output is also zero
    x = torch.randn(2, 10, IN_FEATURES)
    out = adapter(x)
    assert torch.all(out == 0), "output must be zero when lora_B is zero"


# ---------------------------------------------------------------------------
# 4. LoRAPlusAdapter get_param_groups returns two groups with different LRs
# ---------------------------------------------------------------------------

def test_loraplus_get_param_groups_two_groups_different_lrs():
    cfg = default_config("lora_plus")
    adapter = LoRAPlusAdapter(IN_FEATURES, OUT_FEATURES, cfg)
    base_lr = 1e-4
    groups = adapter.get_param_groups(base_lr)

    assert len(groups) == 2, "must return exactly two param groups"
    lr_a = groups[0]["lr"]
    lr_b = groups[1]["lr"]
    assert lr_a == base_lr, f"A lr should be {base_lr}, got {lr_a}"
    assert lr_b == pytest.approx(base_lr * cfg.lora_plus_lr_ratio), (
        f"B lr should be {base_lr * cfg.lora_plus_lr_ratio}, got {lr_b}"
    )
    assert lr_a != lr_b, "A and B must have different learning rates"

    # Verify correct parameters in each group
    assert adapter.lora_A in groups[0]["params"]
    assert adapter.lora_B in groups[1]["params"]


# ---------------------------------------------------------------------------
# 5. IA3Adapter forward output shape same as input
# ---------------------------------------------------------------------------

def test_ia3_forward_output_shape():
    adapter = IA3Adapter(OUT_FEATURES)
    x = torch.randn(2, 10, OUT_FEATURES)
    out = adapter(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 6. IA3Adapter scale_vector initialized to ones → identity transform initially
# ---------------------------------------------------------------------------

def test_ia3_scale_vector_initialized_to_ones():
    adapter = IA3Adapter(OUT_FEATURES)
    assert torch.all(adapter.scale_vector == 1.0), "scale_vector must be ones at init"

    # When scale_vector is all ones, output equals input
    x = torch.randn(2, 10, OUT_FEATURES)
    out = adapter(x)
    assert torch.allclose(out, x), "identity transform expected when scale_vector=ones"


# ---------------------------------------------------------------------------
# 7. IA3Adapter merge_into_linear creates scaled Linear
# ---------------------------------------------------------------------------

def test_ia3_merge_into_linear():
    adapter = IA3Adapter(OUT_FEATURES)
    # Set non-trivial scale so we can verify the merge
    with torch.no_grad():
        adapter.scale_vector.copy_(torch.rand(OUT_FEATURES) + 0.5)

    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    nn.init.normal_(linear.weight)

    merged = adapter.merge_into_linear(linear)

    assert isinstance(merged, nn.Linear)
    assert merged.weight.shape == linear.weight.shape

    # Verify merged weight = original weight * scale_vector (row-wise)
    expected_weight = linear.weight * adapter.scale_vector.unsqueeze(1)
    assert torch.allclose(merged.weight, expected_weight, atol=1e-6)

    # Verify outputs match
    x = torch.randn(3, IN_FEATURES)
    ia3_out = adapter(linear(x))
    merged_out = merged(x)
    assert torch.allclose(ia3_out, merged_out, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. VeRAAdapter forward output shape correct
# ---------------------------------------------------------------------------

def test_vera_forward_output_shape():
    cfg = default_config("vera")
    adapter = VeRAAdapter(IN_FEATURES, OUT_FEATURES, cfg)
    x = torch.randn(2, 10, IN_FEATURES)
    out = adapter(x)
    assert out.shape == (2, 10, OUT_FEATURES)


# ---------------------------------------------------------------------------
# 9. VeRAAdapter only d and b are parameters; A_frozen and B_frozen are buffers
# ---------------------------------------------------------------------------

def test_vera_only_d_and_b_are_parameters():
    cfg = default_config("vera")
    adapter = VeRAAdapter(IN_FEATURES, OUT_FEATURES, cfg)

    param_names = {n for n, _ in adapter.named_parameters()}
    assert "d" in param_names, "d must be a parameter"
    assert "b" in param_names, "b must be a parameter"
    assert "A_frozen" not in param_names, "A_frozen must NOT be a parameter"
    assert "B_frozen" not in param_names, "B_frozen must NOT be a parameter"

    buffer_names = {n for n, _ in adapter.named_buffers()}
    assert "A_frozen" in buffer_names, "A_frozen must be a buffer"
    assert "B_frozen" in buffer_names, "B_frozen must be a buffer"


# ---------------------------------------------------------------------------
# 10. VeRAAdapter A_frozen and B_frozen do not require grad
# ---------------------------------------------------------------------------

def test_vera_frozen_buffers_no_grad():
    cfg = default_config("vera")
    adapter = VeRAAdapter(IN_FEATURES, OUT_FEATURES, cfg)
    assert not adapter.A_frozen.requires_grad, "A_frozen must not require grad"
    assert not adapter.B_frozen.requires_grad, "B_frozen must not require grad"


# ---------------------------------------------------------------------------
# 11. get_adapter_params returns non-empty list after applying adapters
# ---------------------------------------------------------------------------

def test_get_adapter_params_non_empty_after_apply():
    model = small_model()
    cfg = default_config("lora_plus")
    apply_adapters_to_model(model, cfg)
    params = get_adapter_params(model, "lora_plus")
    assert len(params) > 0, "should find adapter parameters after apply_adapters_to_model"


# ---------------------------------------------------------------------------
# 12. apply_adapters_to_model freezes base model params
# ---------------------------------------------------------------------------

def test_apply_adapters_freezes_base_model_params():
    model = small_model()
    cfg = default_config("lora_plus")

    # Collect original parameter names before applying adapters
    original_param_names = {n for n, _ in model.named_parameters()}

    apply_adapters_to_model(model, cfg)

    # All originally frozen parameters should still be frozen
    for name, param in model.named_parameters():
        if name in original_param_names:
            assert not param.requires_grad, (
                f"Original param '{name}' should be frozen after apply_adapters_to_model"
            )


# ---------------------------------------------------------------------------
# 13. apply_adapters_to_model adds adapter attributes to model
# ---------------------------------------------------------------------------

def test_apply_adapters_adds_adapter_attributes():
    model = small_model()
    cfg = default_config("lora_plus")
    apply_adapters_to_model(model, cfg)

    # Search for any *_adapter attributes on submodules
    adapter_count = 0
    for module in model.modules():
        for attr_name in dir(module):
            if attr_name.endswith("_adapter"):
                attr = getattr(module, attr_name)
                if isinstance(attr, nn.Module):
                    adapter_count += 1

    assert adapter_count > 0, (
        "apply_adapters_to_model must add at least one *_adapter attribute"
    )


# ---------------------------------------------------------------------------
# Bonus: adapter param count sanity for VeRA
# ---------------------------------------------------------------------------

def test_vera_param_count_is_small():
    """VeRA's trainable params (d + b) << LoRA params for same problem size."""
    cfg = LoRAVariantConfig(rank=8, alpha=16.0, variant="vera", vera_shared_dim=256)
    vera = VeRAAdapter(IN_FEATURES, OUT_FEATURES, cfg)
    vera_params = sum(p.numel() for p in vera.parameters() if p.requires_grad)
    # d: 256, b: 64 → 320 total
    assert vera_params == cfg.vera_shared_dim + OUT_FEATURES, (
        f"Expected {cfg.vera_shared_dim + OUT_FEATURES} trainable params, got {vera_params}"
    )
