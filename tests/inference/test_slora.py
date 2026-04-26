"""
Tests for src/inference/slora.py

S-LoRA: Scalable serving of many LoRA adapters (Sheng et al., 2023).

Test configuration: in_features=32, out_features=64, rank=4
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.inference.slora import (
    SLoRALayer,
    SLoRALinear,
    SLoRARegistry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IN_F = 32
OUT_F = 64
RANK = 4
BATCH = 3
SEQ = 8


def make_registry(max_adapters: int = 8) -> SLoRARegistry:
    return SLoRARegistry(max_adapters=max_adapters)


def make_ab(
    in_f: int = IN_F,
    out_f: int = OUT_F,
    rank: int = RANK,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (A, B) tensors with consistent shapes."""
    g = torch.Generator()
    g.manual_seed(seed)
    A = torch.randn(rank, in_f, generator=g)  # (r, d_in)
    B = torch.randn(out_f, rank, generator=g)  # (d_out, r)
    return A, B


def make_layer(registry: SLoRARegistry | None = None) -> SLoRALinear:
    if registry is None:
        registry = make_registry()
    return SLoRALinear(IN_F, OUT_F, registry, bias=False)


def make_input(batch: int = BATCH, seq: int = SEQ, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, seq, IN_F)


# ---------------------------------------------------------------------------
# 1. Output shape matches nn.Linear
# ---------------------------------------------------------------------------


def test_output_shape_matches_linear():
    layer = make_layer()
    x = make_input()
    out = layer(x, [None] * BATCH)
    assert out.shape == (BATCH, SEQ, OUT_F), f"Expected ({BATCH}, {SEQ}, {OUT_F}), got {out.shape}"


# ---------------------------------------------------------------------------
# 2. No adapter (adapter_ids=[None]): output equals base linear
# ---------------------------------------------------------------------------


def test_no_adapter_equals_base_linear():
    layer = make_layer()
    x = make_input(batch=1)
    out_slora = layer(x, [None])
    out_base = layer.W_base(x)
    assert torch.allclose(out_slora, out_base, atol=1e-6), (
        "With adapter_ids=[None], SLoRALinear should equal W_base(x)."
    )


# ---------------------------------------------------------------------------
# 3. With adapter: output = base + lora_delta
# ---------------------------------------------------------------------------


def test_output_equals_base_plus_lora_delta():
    registry = make_registry()
    A, B = make_ab()
    scaling = 0.5
    registry.swap_in("adp0", A, B, RANK, scaling)

    layer = make_layer(registry)
    x = make_input(batch=1)

    out_slora = layer(x, ["adp0"])

    # Manually compute expected delta
    with torch.no_grad():
        expected_base = layer.W_base(x)
        expected_delta = (x @ A.T) @ B.T * scaling
        expected = expected_base + expected_delta

    assert torch.allclose(out_slora, expected, atol=1e-5), "output should equal base + LoRA delta."


# ---------------------------------------------------------------------------
# 4. Different adapters in same batch produce different outputs
# ---------------------------------------------------------------------------


def test_different_adapters_produce_different_outputs():
    registry = make_registry()
    A0, B0 = make_ab(seed=1)
    A1, B1 = make_ab(seed=2)
    registry.swap_in("adp0", A0, B0, RANK)
    registry.swap_in("adp1", A1, B1, RANK)

    layer = make_layer(registry)
    x = make_input(batch=2, seed=10)

    out = layer(x, ["adp0", "adp1"])
    # Each row used a different adapter; deltas differ so outputs must differ.
    assert not torch.allclose(out[0], out[1], atol=1e-6), (
        "Different adapters should yield different per-item outputs."
    )


# ---------------------------------------------------------------------------
# 5. Same adapter full-batch == per-item application (consistency)
# ---------------------------------------------------------------------------


def test_same_adapter_batch_consistency():
    registry = make_registry()
    A, B = make_ab(seed=7)
    registry.swap_in("adp0", A, B, RANK)

    layer = make_layer(registry)
    x = make_input(batch=3, seed=5)

    out_batch = layer(x, ["adp0", "adp0", "adp0"])

    for b in range(3):
        out_single = layer(x[b : b + 1], ["adp0"])
        assert torch.allclose(out_batch[b : b + 1], out_single, atol=1e-5), (
            f"Batch item {b}: batch result should match single-item result."
        )


# ---------------------------------------------------------------------------
# 6. swap_out removes adapter; subsequent access raises KeyError
# ---------------------------------------------------------------------------


def test_swap_out_removes_adapter():
    registry = make_registry()
    A, B = make_ab()
    registry.swap_in("adp0", A, B, RANK)
    assert "adp0" in registry

    registry.swap_out("adp0")
    assert "adp0" not in registry

    with pytest.raises(KeyError):
        registry.get("adp0")


# ---------------------------------------------------------------------------
# 7. max_adapters limit enforced
# ---------------------------------------------------------------------------


def test_max_adapters_limit_raises():
    registry = make_registry(max_adapters=2)
    A, B = make_ab()
    registry.swap_in("adp0", A, B, RANK)
    registry.swap_in("adp1", A, B, RANK)

    with pytest.raises(RuntimeError, match="capacity"):
        registry.swap_in("adp2", A, B, RANK)


# ---------------------------------------------------------------------------
# 8. Mixed None and real adapter_ids — correct selective application
# ---------------------------------------------------------------------------


def test_mixed_none_and_real_adapter_ids():
    registry = make_registry()
    A, B = make_ab(seed=3)
    scaling = 1.0
    registry.swap_in("adp0", A, B, RANK, scaling)

    layer = make_layer(registry)
    x = make_input(batch=3, seed=8)

    out = layer(x, [None, "adp0", None])

    # Items 0 and 2 should equal base only
    for b in [0, 2]:
        expected_base = layer.W_base(x[b : b + 1])
        assert torch.allclose(out[b : b + 1], expected_base, atol=1e-5), (
            f"Batch item {b} (adapter=None) should equal base output."
        )

    # Item 1 should differ from base
    base_1 = layer.W_base(x[1:2])
    assert not torch.allclose(out[1:2], base_1, atol=1e-6), (
        "Batch item 1 (adapter='adp0') should differ from base output."
    )


# ---------------------------------------------------------------------------
# 9. Gradient flows through base weights
# ---------------------------------------------------------------------------


def test_gradient_flows_through_base_weights():
    registry = make_registry()
    A, B = make_ab()
    registry.swap_in("adp0", A, B, RANK)

    layer = make_layer(registry)
    x = make_input(batch=1)
    x.requires_grad_(False)

    out = layer(x, ["adp0"])
    loss = out.sum()
    loss.backward()

    assert layer.W_base.weight.grad is not None, "Gradient should flow through W_base.weight."
    assert layer.W_base.weight.grad.abs().sum() > 0, "W_base.weight gradient should be non-zero."


# ---------------------------------------------------------------------------
# 10. Gradient does NOT flow through A, B in registry (frozen adapters)
# ---------------------------------------------------------------------------


def test_no_gradient_through_adapter_weights():
    registry = make_registry()
    A, B = make_ab()
    A.requires_grad_(True)
    B.requires_grad_(True)
    registry.swap_in("adp0", A, B, RANK)

    layer = make_layer(registry)
    x = make_input(batch=1)

    out = layer(x, ["adp0"])
    loss = out.sum()
    loss.backward()

    # A and B are detached inside forward; their .grad should be None
    assert A.grad is None, "A should have no gradient (inference-frozen adapter)."
    assert B.grad is None, "B should have no gradient (inference-frozen adapter)."


# ---------------------------------------------------------------------------
# 11. Determinism under torch.manual_seed
# ---------------------------------------------------------------------------


def test_determinism_under_manual_seed():
    def run(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        registry = make_registry()
        A, B = make_ab(seed=seed)
        registry.swap_in("adp0", A, B, RANK)
        layer = make_layer(registry)
        torch.manual_seed(0)
        x = torch.randn(2, SEQ, IN_F)
        return layer(x, ["adp0", None]).detach().clone()

    out1 = run(42)
    out2 = run(42)
    assert torch.allclose(out1, out2), "Identical seeds should produce identical outputs."


# ---------------------------------------------------------------------------
# 12. No NaN/Inf on normal inputs
# ---------------------------------------------------------------------------


def test_no_nan_or_inf():
    registry = make_registry()
    A, B = make_ab()
    registry.swap_in("adp0", A, B, RANK)

    layer = make_layer(registry)
    x = make_input(batch=4)
    out = layer(x, ["adp0", None, "adp0", None])

    assert torch.isfinite(out).all(), "Output should contain no NaN or Inf values."


# ---------------------------------------------------------------------------
# 13. scaling=0.0 → output equals base (delta cancelled)
# ---------------------------------------------------------------------------


def test_scaling_zero_equals_base():
    registry = make_registry()
    A, B = make_ab(seed=9)
    registry.swap_in("adp0", A, B, RANK, scaling=0.0)

    layer = make_layer(registry)
    x = make_input(batch=1)

    out_slora = layer(x, ["adp0"])
    out_base = layer.W_base(x)

    assert torch.allclose(out_slora, out_base, atol=1e-6), (
        "scaling=0.0 should zero-out the LoRA delta, giving the base output."
    )


# ---------------------------------------------------------------------------
# 14. rank=1 edge case works
# ---------------------------------------------------------------------------


def test_rank_one_edge_case():
    registry = make_registry()
    rank = 1
    A = torch.randn(rank, IN_F)
    B = torch.randn(OUT_F, rank)
    registry.swap_in("rank1", A, B, rank)

    layer = make_layer(registry)
    x = make_input(batch=2)
    out = layer(x, ["rank1", None])

    assert out.shape == (2, SEQ, OUT_F), "rank=1 output shape mismatch."
    assert torch.isfinite(out).all(), "rank=1 output should be finite."


# ---------------------------------------------------------------------------
# 15. Unknown adapter_id raises KeyError
# ---------------------------------------------------------------------------


def test_unknown_adapter_id_raises_key_error():
    registry = make_registry()
    layer = make_layer(registry)
    x = make_input(batch=1)

    with pytest.raises(KeyError):
        layer(x, ["nonexistent_adapter"])


# ---------------------------------------------------------------------------
# Bonus: SLoRALayer wraps existing nn.Linear correctly
# ---------------------------------------------------------------------------


def test_slora_layer_wrapper_matches_slora_linear():
    torch.manual_seed(0)
    base = nn.Linear(IN_F, OUT_F, bias=False)
    registry = make_registry()
    A, B = make_ab(seed=5)
    registry.swap_in("adp0", A, B, RANK)

    wrapper = SLoRALayer(base, registry)

    x = make_input(batch=2, seed=99)
    out_wrapper = wrapper(x, ["adp0", None])

    # Manually check item 0 has delta applied
    with torch.no_grad():
        expected0 = (base(x[0:1])) + (x[0:1] @ A.T) @ B.T
    assert torch.allclose(out_wrapper[0:1], expected0, atol=1e-5)

    # Item 1 (None) should be pure base
    out_base_1 = base(x[1:2])
    assert torch.allclose(out_wrapper[1:2], out_base_1, atol=1e-6)
