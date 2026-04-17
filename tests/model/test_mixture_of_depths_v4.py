"""
Tests for mixture_of_depths_v4.py — Mixture of Depths (MoD).
All configs use tiny sizes: d_model=16, n_heads=2, seq_len=8, batch=2,
n_layers=2, vocab_size=16.
Every test exercises forward (and/or backward) passes, not just instantiation.
"""

import pytest
import torch
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from model.mixture_of_depths_v4 import (
    TokenRouter,
    MoDBlock,
    MoDTransformer,
    MoDLoss,
    CapacityAnalyzer,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------
D = 16
N_HEADS = 2
T = 8
B = 2
N_LAYERS = 2
V = 16


# ===========================================================================
# 1. TokenRouter — output shapes
# ===========================================================================
def test_token_router_mask_shape_and_dtype():
    router = TokenRouter(D, capacity_factor=0.5)
    x = torch.randn(B, T, D)
    mask, logits = router(x)
    assert mask.shape == (B, T), f"Expected ({B},{T}), got {mask.shape}"
    assert mask.dtype == torch.bool, f"Expected bool, got {mask.dtype}"


def test_token_router_logits_shape_and_dtype():
    router = TokenRouter(D, capacity_factor=0.5)
    x = torch.randn(B, T, D)
    mask, logits = router(x)
    assert logits.shape == (B, T), f"Expected ({B},{T}), got {logits.shape}"
    assert logits.is_floating_point(), "logits should be float"


# ===========================================================================
# 2. TokenRouter — capacity constraint
# ===========================================================================
def test_token_router_capacity_fraction():
    """Exactly int(T * capacity_factor) tokens selected per batch item."""
    cf = 0.5
    k_expected = int(T * cf)
    router = TokenRouter(D, capacity_factor=cf)
    x = torch.randn(B, T, D)
    mask, _ = router(x)
    for b in range(B):
        n_selected = mask[b].sum().item()
        assert n_selected == k_expected, (
            f"Batch {b}: expected {k_expected} selected, got {n_selected}"
        )


def test_token_router_capacity_factor_zero_point_25():
    """capacity_factor=0.25 → 2 tokens selected out of 8."""
    cf = 0.25
    k_expected = max(1, int(T * cf))
    router = TokenRouter(D, capacity_factor=cf)
    x = torch.randn(B, T, D)
    mask, _ = router(x)
    for b in range(B):
        assert mask[b].sum().item() == k_expected


# ===========================================================================
# 3. MoDBlock — output shape preserved
# ===========================================================================
def test_mod_block_output_shape():
    block = MoDBlock(D, N_HEADS, capacity_factor=0.5)
    x = torch.randn(B, T, D)
    out, aux = block(x)
    assert out.shape == (B, T, D), f"Expected ({B},{T},{D}), got {out.shape}"


# ===========================================================================
# 4. MoDBlock — aux_loss is a non-negative scalar
# ===========================================================================
def test_mod_block_aux_loss_scalar_nonneg():
    block = MoDBlock(D, N_HEADS, capacity_factor=0.5)
    x = torch.randn(B, T, D)
    _, aux = block(x)
    assert aux.ndim == 0, "aux_loss must be a scalar tensor"
    assert aux.item() >= 0.0, "aux_loss must be non-negative"


# ===========================================================================
# 5. MoDBlock — with capacity=0.0 all tokens bypass (unchanged)
# ===========================================================================
def test_mod_block_zero_capacity_bypass():
    """capacity_factor=0.0 → k=max(1,0)=1 token still selected, but we can
    check that a block with capacity_factor effectively zero leaves almost all
    tokens at the residual value.  We use capacity_factor=1/T so k=1."""
    # Use tiny epsilon so only 1 token goes through; the rest must equal x
    # Actually the spec says capacity=0.0 → all bypass.  max(1,int(8*0))=1.
    # We monkey-patch k=0 to truly test all-bypass.
    block = MoDBlock(D, N_HEADS, capacity_factor=0.0)
    # Override router to return an all-False mask
    class _ZeroRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(D, 1, bias=False)
        def forward(self, x):
            B, T, D = x.shape
            mask = torch.zeros(B, T, dtype=torch.bool)
            logits = torch.zeros(B, T)
            return mask, logits
    block.router = _ZeroRouter()

    x = torch.randn(B, T, D)
    out, _ = block(x)
    assert torch.allclose(out, x), "With all-bypass, output must equal input exactly"


# ===========================================================================
# 6. MoDTransformer — logits shape
# ===========================================================================
def test_mod_transformer_logits_shape():
    model = MoDTransformer(D, N_LAYERS, N_HEADS, V)
    ids = torch.randint(0, V, (B, T))
    logits, _ = model(ids)
    assert logits.shape == (B, T, V), f"Expected ({B},{T},{V}), got {logits.shape}"


# ===========================================================================
# 7. MoDTransformer — aux_loss is a scalar
# ===========================================================================
def test_mod_transformer_aux_loss_scalar():
    model = MoDTransformer(D, N_LAYERS, N_HEADS, V)
    ids = torch.randint(0, V, (B, T))
    _, aux = model(ids)
    assert aux.ndim == 0, "total_aux_loss must be a scalar"


# ===========================================================================
# 8. MoDTransformer — gradients flow to all parameters
# ===========================================================================
def test_mod_transformer_grad_flows_to_all_params():
    model = MoDTransformer(D, N_LAYERS, N_HEADS, V)
    ids = torch.randint(0, V, (B, T))
    logits, aux = model(ids)
    targets = torch.randint(0, V, (B, T))
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T)
    ) + aux
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for param: {name}"


# ===========================================================================
# 9. MoDLoss — total = ce + aux*weight, correct shapes
# ===========================================================================
def test_mod_loss_values_and_shapes():
    criterion = MoDLoss(aux_weight=0.01)
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    aux = torch.tensor(2.0)
    total, ce, aux_w = criterion(logits, targets, aux)
    assert total.ndim == 0
    assert ce.ndim == 0
    assert aux_w.ndim == 0
    assert torch.allclose(total, ce + aux_w), "total must equal ce + aux_weighted"
    assert torch.allclose(aux_w, torch.tensor(0.02)), "aux_weighted = 0.01 * 2.0 = 0.02"


# ===========================================================================
# 10. CapacityAnalyzer — record + compute_stats keys present
# ===========================================================================
def test_capacity_analyzer_stats_keys():
    analyzer = CapacityAnalyzer()
    mask = torch.ones(B, T, dtype=torch.bool)
    analyzer.record(mask)
    stats = analyzer.compute_stats()
    for key in ("mean_capacity", "std_capacity", "min_capacity", "max_capacity"):
        assert key in stats, f"Missing key: {key}"


# ===========================================================================
# 11. CapacityAnalyzer — reset clears state
# ===========================================================================
def test_capacity_analyzer_reset():
    analyzer = CapacityAnalyzer()
    mask = torch.ones(B, T, dtype=torch.bool)
    analyzer.record(mask)
    analyzer.reset()
    stats = analyzer.compute_stats()
    # After reset, should return zero-valued stats (no records)
    assert stats["mean_capacity"] == 0.0
    assert stats["min_capacity"] == 0.0
    assert stats["max_capacity"] == 0.0


# ===========================================================================
# 12. Variable capacity_factors per layer accepted
# ===========================================================================
def test_mod_transformer_variable_capacity_factors():
    cfs = [0.25, 0.75]
    model = MoDTransformer(D, N_LAYERS, N_HEADS, V, capacity_factors=cfs)
    ids = torch.randint(0, V, (B, T))
    logits, aux = model(ids)
    assert logits.shape == (B, T, V)
    assert aux.ndim == 0


# ===========================================================================
# 13. Backward pass through full model succeeds (no error, loss is finite)
# ===========================================================================
def test_full_model_backward_no_error():
    model = MoDTransformer(D, N_LAYERS, N_HEADS, V)
    criterion = MoDLoss(aux_weight=0.01)
    ids = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))
    logits, aux = model(ids)
    total, _, _ = criterion(logits, targets, aux)
    total.backward()
    assert total.item() == total.item(), "Loss is NaN"  # NaN != NaN
    assert total.item() < float("inf"), "Loss is inf"


# ===========================================================================
# 14. capacity=1.0 output differs from capacity=0.0 (all bypass)
# ===========================================================================
def test_capacity_full_vs_zero_differ():
    """With capacity=1.0 all tokens are processed; with capacity=0.0 (forced
    all-bypass via monkey-patch) output equals input.  They must differ."""
    torch.manual_seed(42)

    # Block with capacity 1.0 (all tokens processed)
    block_full = MoDBlock(D, N_HEADS, capacity_factor=1.0)
    x = torch.randn(B, T, D)
    out_full, _ = block_full(x)

    # Block with forced all-bypass
    block_bypass = MoDBlock(D, N_HEADS, capacity_factor=0.0)
    class _ZeroRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(D, 1, bias=False)
        def forward(self, inp):
            Bi, Ti, Di = inp.shape
            mask = torch.zeros(Bi, Ti, dtype=torch.bool)
            logits = torch.zeros(Bi, Ti)
            return mask, logits
    block_bypass.router = _ZeroRouter()
    out_bypass, _ = block_bypass(x)

    # out_bypass must equal x; out_full must differ from x
    assert torch.allclose(out_bypass, x), "Bypass output must equal input"
    assert not torch.allclose(out_full, x), "Full capacity output must differ from input"


# ===========================================================================
# 15. MoDLoss — ignore_index=-100 is respected (masked tokens ignored)
# ===========================================================================
def test_mod_loss_ignore_index():
    """ignore_index=-100 is threaded through to F.cross_entropy.
    Test that partially-masked targets only score the unmasked positions:
    the loss for the unmasked half should differ from a fully-unmasked loss."""
    criterion = MoDLoss(aux_weight=0.0)
    torch.manual_seed(7)
    logits = torch.randn(B, T, V)
    aux = torch.tensor(0.0)

    # All real targets
    targets_real = torch.randint(0, V, (B, T))
    total_real, ce_real, _ = criterion(logits, targets_real, aux)

    # Mask the first half of tokens → loss computed only on second half
    targets_masked = targets_real.clone()
    targets_masked[:, : T // 2] = -100
    total_masked, ce_masked, _ = criterion(logits, targets_masked, aux)

    # Losses must differ because different tokens are included
    assert ce_real.item() != ce_masked.item(), (
        "CE with half masked should differ from CE with none masked"
    )
    # Both losses must be finite positive numbers
    assert ce_real.item() > 0
    assert ce_masked.item() > 0
