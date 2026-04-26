"""Tests for src/eval/attention_rollout.py.

Covers AttentionRollout, RolloutAttributor, and AttentionRolloutHook.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from aurelius.eval.attention_rollout import (
    AttentionRollout,
    AttentionRolloutHook,
    RolloutAttributor,
)
from torch import Tensor

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

T = 8  # sequence length
H = 4  # number of attention heads
B = 2  # batch size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_attn(t: int = T) -> Tensor:
    """Return a (T, T) uniform attention matrix (rows sum to 1)."""
    return torch.full((t, t), 1.0 / t)


def _identity_attn(t: int = T) -> Tensor:
    """Return a (T, T) identity attention matrix."""
    return torch.eye(t)


def _rand_attn_batched(b: int = B, h: int = H, t: int = T) -> Tensor:
    """Return a (B, H, T, T) softmax-normalised random attention tensor."""
    raw = torch.rand(b, h, t, t)
    return raw / raw.sum(dim=-1, keepdim=True)


def _rand_attn_unbatched(h: int = H, t: int = T) -> Tensor:
    """Return a (H, T, T) softmax-normalised random attention tensor."""
    raw = torch.rand(h, t, t)
    return raw / raw.sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Test 1 — single-layer identity input produces identity rollout
# ---------------------------------------------------------------------------


def test_single_layer_identity_rollout():
    """If A_l = I, rollout should also be I (after residual + normalise)."""
    rollout = AttentionRollout()
    attn = _identity_attn()  # (T, T)
    result = rollout.compute([attn])  # (T, T)
    assert torch.allclose(result, torch.eye(T), atol=1e-6), (
        "Identity attention should yield identity rollout"
    )


# ---------------------------------------------------------------------------
# Test 2 — single-layer uniform attention → rows sum to 1
# ---------------------------------------------------------------------------


def test_single_layer_uniform_rows_sum_to_one():
    """Rollout of a uniform attention map must have rows summing to 1."""
    rollout = AttentionRollout()
    attn = _uniform_attn()
    result = rollout.compute([attn])
    row_sums = result.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(T), atol=1e-6), (
        f"Row sums should be 1, got {row_sums}"
    )


# ---------------------------------------------------------------------------
# Test 3 — output shape (T, T) for unbatched (T, T) input
# ---------------------------------------------------------------------------


def test_output_shape_unbatched_tt():
    """Single unbatched (T, T) map → output shape (T, T)."""
    rollout = AttentionRollout()
    attn = _uniform_attn()  # (T, T)
    result = rollout.compute([attn])
    assert result.shape == (T, T), f"Expected ({T}, {T}), got {result.shape}"


# ---------------------------------------------------------------------------
# Test 4 — output shape (B, T, T) for batched (B, H, T, T) input
# ---------------------------------------------------------------------------


def test_output_shape_batched():
    """Batched (B, H, T, T) maps → output shape (B, T, T)."""
    rollout = AttentionRollout()
    attn = _rand_attn_batched()  # (B, H, T, T)
    result = rollout.compute([attn])
    assert result.shape == (B, T, T), f"Expected ({B}, {T}, {T}), got {result.shape}"


# ---------------------------------------------------------------------------
# Test 5 — all values in [0, 1]
# ---------------------------------------------------------------------------


def test_values_in_zero_one():
    """Rollout values must lie in [0, 1]."""
    rollout = AttentionRollout()
    maps = [_rand_attn_batched() for _ in range(3)]
    result = rollout.compute(maps)
    assert result.min().item() >= -1e-7, "Rollout contains negative values"
    assert result.max().item() <= 1.0 + 1e-7, "Rollout contains values > 1"


# ---------------------------------------------------------------------------
# Test 6 — rows sum to 1 (normalised)
# ---------------------------------------------------------------------------


def test_rows_sum_to_one():
    """All rows of the rollout matrix must sum to ~1."""
    rollout = AttentionRollout()
    maps = [_rand_attn_batched() for _ in range(4)]
    result = rollout.compute(maps)  # (B, T, T)
    row_sums = result.sum(dim=-1)  # (B, T)
    assert torch.allclose(row_sums, torch.ones(B, T), atol=1e-5), (
        f"Row sums off: min={row_sums.min():.6f} max={row_sums.max():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 7 — multi-layer: 4 random layers → correct output shape
# ---------------------------------------------------------------------------


def test_multi_layer_output_shape():
    """Four batched layers must produce a (B, T, T) rollout."""
    rollout = AttentionRollout()
    maps = [_rand_attn_batched() for _ in range(4)]
    result = rollout.compute(maps)
    assert result.shape == (B, T, T), f"Expected ({B}, {T}, {T}), got {result.shape}"


# ---------------------------------------------------------------------------
# Test 8 — discard_ratio=0.5 zeros out bottom half of weights
# ---------------------------------------------------------------------------


def test_discard_ratio_zeroes_low_weights():
    """discard_ratio=0.5 must produce sparser maps than discard_ratio=0."""
    torch.manual_seed(42)
    rollout_nodiscard = AttentionRollout(discard_ratio=0.0)
    rollout_discard = AttentionRollout(discard_ratio=0.5)

    maps = [_rand_attn_batched() for _ in range(2)]

    # Compute with no discarding and with 50 % discarding
    result_nodiscard = rollout_nodiscard.compute(maps)
    result_discard = rollout_discard.compute(maps)

    # Both must still produce valid shapes
    assert result_discard.shape == (B, T, T)
    # The discarded rollout should differ from the non-discarded one
    assert not torch.allclose(result_nodiscard, result_discard, atol=1e-6), (
        "discard_ratio=0.5 should change the rollout"
    )


# ---------------------------------------------------------------------------
# Test 9 — head_fusion="min" produces valid rollout
# ---------------------------------------------------------------------------


def test_head_fusion_min_valid():
    """head_fusion='min' must produce shape (B, T, T) with rows summing to 1."""
    rollout = AttentionRollout(head_fusion="min")
    maps = [_rand_attn_batched() for _ in range(3)]
    result = rollout.compute(maps)
    assert result.shape == (B, T, T)
    row_sums = result.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(B, T), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 10 — head_fusion="max" produces valid rollout
# ---------------------------------------------------------------------------


def test_head_fusion_max_valid():
    """head_fusion='max' must produce shape (B, T, T) with rows summing to 1."""
    rollout = AttentionRollout(head_fusion="max")
    maps = [_rand_attn_batched() for _ in range(3)]
    result = rollout.compute(maps)
    assert result.shape == (B, T, T)
    row_sums = result.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(B, T), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 11 — RolloutAttributor.attribute output shape (B, T) for batched input
# ---------------------------------------------------------------------------


def test_attributor_output_shape_batched():
    """attribute() must return (B, T) for batched attention maps."""
    rollout = AttentionRollout()
    attributor = RolloutAttributor(rollout)
    maps = [_rand_attn_batched() for _ in range(3)]
    scores = attributor.attribute(maps, target_pos=0)
    assert scores.shape == (B, T), f"Expected ({B}, {T}), got {scores.shape}"


# ---------------------------------------------------------------------------
# Test 12 — RolloutAttributor.attribute sums to 1 per batch item
# ---------------------------------------------------------------------------


def test_attributor_sums_to_one():
    """Importance scores must sum to 1 for each batch item."""
    rollout = AttentionRollout()
    attributor = RolloutAttributor(rollout)
    maps = [_rand_attn_batched() for _ in range(3)]
    scores = attributor.attribute(maps, target_pos=2)
    totals = scores.sum(dim=-1)  # (B,)
    assert torch.allclose(totals, torch.ones(B), atol=1e-6), (
        f"Scores should sum to 1 per batch, got {totals}"
    )


# ---------------------------------------------------------------------------
# Test 13 — RolloutAttributor.attribute uses correct target_pos row
# ---------------------------------------------------------------------------


def test_attributor_target_pos_selects_correct_row():
    """attribute(target_pos=p) must equal rollout[:, p, :] (normalised)."""
    rollout = AttentionRollout()
    attributor = RolloutAttributor(rollout)
    maps = [_rand_attn_batched() for _ in range(2)]

    target = 3
    scores = attributor.attribute(maps, target_pos=target)  # (B, T)
    roll_mat = rollout.compute(maps)  # (B, T, T)
    raw_row = roll_mat[:, target, :]  # (B, T)
    expected = raw_row / raw_row.sum(dim=-1, keepdim=True)

    assert torch.allclose(scores, expected, atol=1e-6), (
        "attributor scores do not match the normalised rollout row"
    )


# ---------------------------------------------------------------------------
# Tiny module for hook tests
# ---------------------------------------------------------------------------


class _AttnModule(nn.Module):
    """Minimal attention-like module that returns (attn_weights, value)."""

    def __init__(self, t: int = T, h: int = H) -> None:
        super().__init__()
        self.t = t
        self.h = h

    def forward(self, x: Tensor) -> tuple:  # type: ignore[override]
        B = x.shape[0]
        attn = torch.softmax(torch.rand(B, self.h, self.t, self.t), dim=-1)
        return attn, x  # first element is attn weights


class _TinyTransformer(nn.Module):
    """Two-layer transformer stub with named 'attn' sub-modules."""

    def __init__(self, t: int = T) -> None:
        super().__init__()
        self.layer0_attn = _AttnModule(t)
        self.layer1_attn = _AttnModule(t)

    def forward(self, x: Tensor) -> Tensor:
        _, x = self.layer0_attn(x)
        _, x = self.layer1_attn(x)
        return x


# ---------------------------------------------------------------------------
# Test 14 — AttentionRolloutHook registers hooks on correct module type
# ---------------------------------------------------------------------------


def test_hook_registers_on_correct_type():
    """Hooks should be registered on all _AttnModule instances."""
    model = _TinyTransformer()
    hook = AttentionRolloutHook(model, attention_module_class=_AttnModule)
    hook.register()

    assert len(hook._hooks) == 2, f"Expected 2 hooks (one per _AttnModule), got {len(hook._hooks)}"
    hook.remove()


# ---------------------------------------------------------------------------
# Test 15 — AttentionRolloutHook.remove cleans up; model still works
# ---------------------------------------------------------------------------


def test_hook_remove_cleans_up():
    """After remove(), hooks list is empty and the model still runs."""
    model = _TinyTransformer()
    hook = AttentionRolloutHook(model, attention_module_class=_AttnModule)
    hook.register()
    hook.remove()

    assert len(hook._hooks) == 0, "Hooks list should be empty after remove()"

    # Model should still run without errors
    x = torch.randn(1, T, 16)
    try:
        model(x)
    except Exception as exc:
        pytest.fail(f"Model raised an exception after hook removal: {exc}")


# ---------------------------------------------------------------------------
# Test 16 — Hook collects maps after a forward pass
# ---------------------------------------------------------------------------


def test_hook_collects_maps():
    """get_maps() must return non-empty list after a forward pass."""
    model = _TinyTransformer()
    hook = AttentionRolloutHook(model, attention_module_class=_AttnModule)
    hook.register()

    x = torch.randn(B, T, 16)
    model(x)

    maps = hook.get_maps()
    assert len(maps) > 0, "No attention maps collected after forward pass"
    # Each map should be a tensor
    for m in maps:
        assert isinstance(m, Tensor), f"Expected Tensor, got {type(m)}"

    hook.remove()


# ---------------------------------------------------------------------------
# Bonus — hook collects maps by name (attention_module_class=None)
# ---------------------------------------------------------------------------


class _NamedAttnTransformer(nn.Module):
    """Transformer stub where attention modules are named exactly 'attn'."""

    def __init__(self) -> None:
        super().__init__()
        # Use nested modules so name ends with 'attn'
        self.layer0 = nn.ModuleDict({"attn": _AttnModule()})
        self.layer1 = nn.ModuleDict({"attn": _AttnModule()})

    def forward(self, x: Tensor) -> Tensor:
        _, x = self.layer0["attn"](x)
        _, x = self.layer1["attn"](x)
        return x


def test_hook_by_name_registers_correctly():
    """With attention_module_class=None, hooks target modules named 'attn'."""
    model = _NamedAttnTransformer()
    hook = AttentionRolloutHook(model)  # no class specified
    hook.register()

    # Should have found 2 modules with leaf name 'attn'
    assert len(hook._hooks) == 2, (
        f"Expected 2 hooks for name-based matching, got {len(hook._hooks)}"
    )

    x = torch.randn(B, T, 16)
    model(x)

    maps = hook.get_maps()
    assert len(maps) == 2, f"Expected 2 maps, got {len(maps)}"
    hook.remove()
