"""Unit and integration tests for CrossLayerKVSharing.

Tiny config used throughout:
    n_layers=4, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
    share_every_n=2

With share_every_n=2 layers 0 and 2 are KV owners; layers 1 and 3 borrow.
"""

from __future__ import annotations

import pytest
import torch

from src.longcontext.cross_layer_kv_sharing import (
    CrossLayerKVAttention,
    CrossLayerKVConfig,
    CrossLayerKVStack,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared tiny config
# ──────────────────────────────────────────────────────────────────────────────

TINY = CrossLayerKVConfig(
    n_layers=4,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    share_every_n=2,
)

B, T = 2, 6  # batch size, sequence length for most tests


def _x(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, TINY.d_model)


# ──────────────────────────────────────────────────────────────────────────────
# 1. test_config_defaults
# ──────────────────────────────────────────────────────────────────────────────

def test_config_defaults() -> None:
    cfg = CrossLayerKVConfig()
    assert cfg.n_layers == 24
    assert cfg.d_model == 2048
    assert cfg.n_heads == 16
    assert cfg.n_kv_heads == 8
    assert cfg.head_dim == 128
    assert cfg.share_every_n == 2


# ──────────────────────────────────────────────────────────────────────────────
# 2. test_kv_owner_has_projections
# ──────────────────────────────────────────────────────────────────────────────

def test_kv_owner_has_projections() -> None:
    owner = CrossLayerKVAttention(TINY, layer_idx=0, is_kv_owner=True)
    assert hasattr(owner, "k_proj") and owner.k_proj is not None
    assert hasattr(owner, "v_proj") and owner.v_proj is not None
    assert isinstance(owner.k_proj, torch.nn.Linear)
    assert isinstance(owner.v_proj, torch.nn.Linear)


# ──────────────────────────────────────────────────────────────────────────────
# 3. test_kv_borrower_no_projections
# ──────────────────────────────────────────────────────────────────────────────

def test_kv_borrower_no_projections() -> None:
    borrower = CrossLayerKVAttention(TINY, layer_idx=1, is_kv_owner=False)
    assert borrower.k_proj is None
    assert borrower.v_proj is None
    # q_proj and out_proj must still be present.
    assert hasattr(borrower, "q_proj") and borrower.q_proj is not None
    assert hasattr(borrower, "out_proj") and borrower.out_proj is not None


# ──────────────────────────────────────────────────────────────────────────────
# 4. test_forward_shape
# ──────────────────────────────────────────────────────────────────────────────

def test_forward_shape() -> None:
    owner = CrossLayerKVAttention(TINY, layer_idx=0, is_kv_owner=True)
    x = _x()
    out, kv = owner(x)
    assert out.shape == (B, T, TINY.d_model), f"Expected {(B, T, TINY.d_model)}, got {out.shape}"
    assert torch.isfinite(out).all()


# ──────────────────────────────────────────────────────────────────────────────
# 5. test_forward_deterministic
# ──────────────────────────────────────────────────────────────────────────────

def test_forward_deterministic() -> None:
    owner = CrossLayerKVAttention(TINY, layer_idx=0, is_kv_owner=True)
    x = _x(42)
    out1, _ = owner(x)
    out2, _ = owner(x)
    assert torch.allclose(out1, out2), "Same input must produce same output"


# ──────────────────────────────────────────────────────────────────────────────
# 6. test_kv_state_shape
# ──────────────────────────────────────────────────────────────────────────────

def test_kv_state_shape() -> None:
    owner = CrossLayerKVAttention(TINY, layer_idx=0, is_kv_owner=True)
    x = _x()
    _, kv_state = owner(x)
    # kv_state packs k and v along last dim: [B, n_kv_heads, T, 2*head_dim]
    expected = (B, TINY.n_kv_heads, T, 2 * TINY.head_dim)
    assert kv_state.shape == expected, f"Expected {expected}, got {kv_state.shape}"


# ──────────────────────────────────────────────────────────────────────────────
# 7. test_borrower_uses_owner_kv
# ──────────────────────────────────────────────────────────────────────────────

def test_borrower_uses_owner_kv() -> None:
    owner = CrossLayerKVAttention(TINY, layer_idx=0, is_kv_owner=True)
    borrower = CrossLayerKVAttention(TINY, layer_idx=1, is_kv_owner=False)

    x = _x(0)

    # Baseline: run owner then borrower with its kv_state.
    _, kv_a = owner(x)
    out_base, _ = borrower(x, kv_cache=kv_a)

    # Provide a different (random) kv_cache — output must differ.
    torch.manual_seed(99)
    kv_b = torch.randn_like(kv_a)
    out_diff_kv, _ = borrower(x, kv_cache=kv_b)
    assert not torch.allclose(out_base, out_diff_kv, atol=1e-5), (
        "Changing the kv_cache should change borrower output"
    )

    # Pass the same kv_cache but build a fresh borrower with different q weights —
    # the output should differ (query matters).
    torch.manual_seed(7)
    borrower2 = CrossLayerKVAttention(TINY, layer_idx=1, is_kv_owner=False)
    out_diff_q, _ = borrower2(x, kv_cache=kv_a)
    assert not torch.allclose(out_base, out_diff_q, atol=1e-5), (
        "Different q weights should change borrower output"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 8. test_stack_forward_shape
# ──────────────────────────────────────────────────────────────────────────────

def test_stack_forward_shape() -> None:
    stack = CrossLayerKVStack(TINY)
    x = _x()
    out = stack(x)
    assert out.shape == x.shape, f"Stack output shape {out.shape} != input shape {x.shape}"
    assert torch.isfinite(out).all()


# ──────────────────────────────────────────────────────────────────────────────
# 9. test_kv_cache_size_ratio
# ──────────────────────────────────────────────────────────────────────────────

def test_kv_cache_size_ratio() -> None:
    stack = CrossLayerKVStack(TINY)
    ratio = stack.kv_cache_size_ratio()
    expected = 1.0 / TINY.share_every_n  # 0.5 for share_every_n=2
    assert abs(ratio - expected) < 1e-9, f"Expected {expected}, got {ratio}"


# ──────────────────────────────────────────────────────────────────────────────
# 10. test_parameter_reduction
# ──────────────────────────────────────────────────────────────────────────────

def test_parameter_reduction() -> None:
    """Borrower layers have strictly fewer parameters than owner layers."""
    owner = CrossLayerKVAttention(TINY, layer_idx=0, is_kv_owner=True)
    borrower = CrossLayerKVAttention(TINY, layer_idx=1, is_kv_owner=False)

    n_owner = sum(p.numel() for p in owner.parameters())
    n_borrower = sum(p.numel() for p in borrower.parameters())

    assert n_borrower < n_owner, (
        f"Borrower ({n_borrower}) should have fewer params than owner ({n_owner})"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 11. test_parameter_count_keys
# ──────────────────────────────────────────────────────────────────────────────

def test_parameter_count_keys() -> None:
    stack = CrossLayerKVStack(TINY)
    counts = stack.parameter_count()
    required = {"total", "kv_params", "q_params", "other_params"}
    assert required.issubset(counts.keys()), (
        f"Missing keys: {required - set(counts.keys())}"
    )
    # Sanity: kv_params + q_params + other_params == total
    parts = counts["kv_params"] + counts["q_params"] + counts["other_params"]
    assert parts == counts["total"], (
        f"Parts sum ({parts}) != total ({counts['total']})"
    )
    # With share_every_n=2 and 4 layers, half the layers own KV projections.
    assert counts["kv_params"] > 0, "There must be some KV parameters"
    assert counts["q_params"] > 0, "There must be some Q parameters"


# ──────────────────────────────────────────────────────────────────────────────
# 12. test_gradient_flows
# ──────────────────────────────────────────────────────────────────────────────

def test_gradient_flows() -> None:
    stack = CrossLayerKVStack(TINY)
    x = _x().requires_grad_(True)
    out = stack(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradient did not flow back to input"
    assert torch.isfinite(x.grad).all(), "Input gradient contains non-finite values"

    # Every parameter that has a weight should have received a gradient.
    for name, p in stack.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"Parameter {name} gradient has non-finite values"


# ──────────────────────────────────────────────────────────────────────────────
# 13. test_share_every_3
# ──────────────────────────────────────────────────────────────────────────────

def test_share_every_3() -> None:
    """share_every_n=3: layers 0, 3, 6, … own KV; 1/3 of layers are owners."""
    cfg = CrossLayerKVConfig(
        n_layers=6,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        share_every_n=3,
    )
    stack = CrossLayerKVStack(cfg)

    owners = [l for l in stack.layers if l.is_kv_owner]
    borrowers = [l for l in stack.layers if not l.is_kv_owner]

    assert len(owners) == 2, f"Expected 2 owners (layers 0, 3), got {len(owners)}"
    assert len(borrowers) == 4, f"Expected 4 borrowers, got {len(borrowers)}"

    ratio = stack.kv_cache_size_ratio()
    assert abs(ratio - 1.0 / 3) < 1e-9, f"Expected 1/3, got {ratio}"

    # Forward should still work.
    x = torch.randn(2, 5, cfg.d_model)
    out = stack(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# ──────────────────────────────────────────────────────────────────────────────
# 14. test_n_layers_4_owners
# ──────────────────────────────────────────────────────────────────────────────

def test_n_layers_4_owners() -> None:
    """n_layers=4, share_every_n=2 → exactly 2 owners at indices 0 and 2."""
    stack = CrossLayerKVStack(TINY)
    owner_indices = [l.layer_idx for l in stack.layers if l.is_kv_owner]
    assert owner_indices == [0, 2], f"Expected owners at [0, 2], got {owner_indices}"
    borrower_indices = [l.layer_idx for l in stack.layers if not l.is_kv_owner]
    assert borrower_indices == [1, 3], f"Expected borrowers at [1, 3], got {borrower_indices}"


# ──────────────────────────────────────────────────────────────────────────────
# Integration test
# ──────────────────────────────────────────────────────────────────────────────

def test_integration_forward_and_backward() -> None:
    """Full integration: n_layers=4 tiny config, B=2 T=8, shape + backward."""
    cfg = CrossLayerKVConfig(
        n_layers=4,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        share_every_n=2,
    )
    stack = CrossLayerKVStack(cfg)

    torch.manual_seed(0)
    x = torch.randn(2, 8, cfg.d_model, requires_grad=True)

    # --- Forward ---
    out = stack(x)

    assert out.shape == (2, 8, cfg.d_model), (
        f"Integration: unexpected output shape {out.shape}"
    )
    assert torch.isfinite(out).all(), "Integration: output contains non-finite values"

    # --- Backward ---
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None, "Integration: no gradient on input"
    assert torch.isfinite(x.grad).all(), "Integration: non-finite gradient on input"

    for name, p in stack.named_parameters():
        assert p.grad is not None, f"Integration: parameter {name} has no gradient"

    # --- KV cache size ratio ---
    assert abs(stack.kv_cache_size_ratio() - 0.5) < 1e-9

    # --- Parameter count sanity ---
    counts = stack.parameter_count()
    assert counts["total"] > 0
    assert counts["kv_params"] > 0

    # Owners have both k and v projections; borrowers have none.
    # Stack has 4 layers, 2 owners → kv_params should equal 2 * (k_proj + v_proj).
    kv_per_owner = cfg.n_kv_heads * cfg.head_dim * cfg.d_model  # one of k or v
    expected_kv = 2 * 2 * kv_per_owner  # 2 owners × 2 projections
    assert counts["kv_params"] == expected_kv, (
        f"Integration: expected kv_params={expected_kv}, got {counts['kv_params']}"
    )
