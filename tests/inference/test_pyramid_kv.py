"""
Tests for PyramidKV (src/inference/pyramid_kv.py).

Tiny config used throughout:
    n_layers=4, total_budget=32, B=1, n_heads=2, T=16, head_dim=8

Test coverage (14 tests):
 1.  all_budgets() length == n_layers
 2.  sum(all_budgets()) ≈ total_budget (within rounding)
 3.  Pyramid: budget decreases monotonically (layer 0 ≥ layer L-1)
 4.  No budget < min_budget
 5.  n_layers=1: single layer receives all budget
 6.  compress() output shape correct (k ≤ b_l)
 7.  compress() keys and values have identical shape
 8.  compress() is deterministic under same inputs
 9.  total_budget < n_layers * min_budget → graceful (each gets min_budget)
10.  schedule='linear': budgets decrease layer-by-layer
11.  No NaN/Inf in compressed output under uniform attention
12.  Higher-budget layers retain more tokens than lower-budget layers
13.  batch_size=1 works
14.  window_size portion always retained in compressed output
"""

from __future__ import annotations

import pytest
import torch

from src.inference.pyramid_kv import PyramidKVCache, PyramidKVScheduler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_LAYERS = 4
TOTAL_BUDGET = 32
MIN_BUDGET = 4
B = 1
N_HEADS = 2
T = 16
HEAD_DIM = 8
WINDOW_SIZE = 4


def make_scheduler(
    n_layers: int = N_LAYERS,
    total_budget: int = TOTAL_BUDGET,
    min_budget: int = MIN_BUDGET,
    schedule: str = "pyramid",
) -> PyramidKVScheduler:
    return PyramidKVScheduler(
        n_layers=n_layers,
        total_budget=total_budget,
        min_budget=min_budget,
        schedule=schedule,
    )


def make_cache(
    n_layers: int = N_LAYERS,
    total_budget: int = TOTAL_BUDGET,
    min_budget: int = MIN_BUDGET,
    window_size: int = WINDOW_SIZE,
    schedule: str = "pyramid",
) -> PyramidKVCache:
    return PyramidKVCache(
        n_layers=n_layers,
        total_budget=total_budget,
        min_budget=min_budget,
        window_size=window_size,
        schedule=schedule,
    )


def make_kv_attn(
    b: int = B,
    n_heads: int = N_HEADS,
    t: int = T,
    head_dim: int = HEAD_DIM,
    t_obs: int = WINDOW_SIZE,
    seed: int = 0,
) -> tuple:
    """Return (keys, values, attn_weights) tensors."""
    torch.manual_seed(seed)
    keys = torch.randn(b, n_heads, t, head_dim)
    values = torch.randn(b, n_heads, t, head_dim)
    attn_raw = torch.rand(b, n_heads, t_obs, t)
    # Normalise so rows sum to 1 (attention probabilities)
    attn_weights = attn_raw / attn_raw.sum(dim=-1, keepdim=True)
    return keys, values, attn_weights


# ---------------------------------------------------------------------------
# PyramidKVScheduler tests
# ---------------------------------------------------------------------------

class TestPyramidKVScheduler:

    # Test 1: all_budgets() length == n_layers
    def test_all_budgets_length(self):
        sched = make_scheduler()
        budgets = sched.all_budgets()
        assert len(budgets) == N_LAYERS, (
            f"Expected {N_LAYERS} budgets, got {len(budgets)}"
        )

    # Test 2: sum(all_budgets()) ≈ total_budget (within rounding)
    def test_all_budgets_sum_approximately_total(self):
        sched = make_scheduler()
        budgets = sched.all_budgets()
        total = sum(budgets)
        # Allow off-by-one per layer due to integer rounding
        assert abs(total - TOTAL_BUDGET) <= N_LAYERS, (
            f"Budget sum {total} deviates too far from {TOTAL_BUDGET}"
        )

    # Test 3: Pyramid schedule — budget_for_layer(0) >= budget_for_layer(L-1)
    def test_pyramid_budgets_decrease(self):
        sched = make_scheduler(schedule="pyramid")
        budgets = sched.all_budgets()
        assert budgets[0] >= budgets[-1], (
            f"Expected layer 0 budget {budgets[0]} >= last layer budget {budgets[-1]}"
        )

    # Test 4: No budget below min_budget
    def test_no_budget_below_min(self):
        sched = make_scheduler()
        budgets = sched.all_budgets()
        for i, b in enumerate(budgets):
            assert b >= MIN_BUDGET, (
                f"Layer {i} budget {b} is below min_budget {MIN_BUDGET}"
            )

    # Test 5: n_layers=1 — single layer receives all budget
    def test_single_layer_gets_all_budget(self):
        sched = make_scheduler(n_layers=1, total_budget=20, min_budget=4)
        budgets = sched.all_budgets()
        assert len(budgets) == 1
        assert budgets[0] == 20, f"Expected 20, got {budgets[0]}"

    # Test 9: total_budget < n_layers * min_budget → graceful (each gets min_budget)
    def test_tight_budget_graceful_fallback(self):
        # total_budget=6, n_layers=4, min_budget=4 → impossible to satisfy; give min to all
        sched = make_scheduler(n_layers=4, total_budget=6, min_budget=4)
        budgets = sched.all_budgets()
        for i, b in enumerate(budgets):
            assert b == 4, (
                f"Layer {i} should get min_budget=4 when total is too tight, got {b}"
            )

    # Test 10: schedule='linear' — budgets decrease from layer 0 to L-1
    def test_linear_schedule_decreases(self):
        sched = make_scheduler(schedule="linear", n_layers=6, total_budget=60, min_budget=4)
        budgets = sched.all_budgets()
        # Overall should decrease (first > last)
        assert budgets[0] >= budgets[-1], (
            f"Linear schedule: expected non-increasing, got {budgets}"
        )
        # Verify strictly no large increase between consecutive layers
        # (allow off-by-1 rounding noise between adjacent layers but overall trend down)
        for i in range(1, len(budgets)):
            assert budgets[i] <= budgets[i - 1] + 1, (
                f"Linear schedule budget jumped up at layer {i}: {budgets}"
            )

    # Test 12 (scheduler part): higher-budget layers budget > lower-budget layers
    def test_pyramid_first_layer_budget_greater_than_last(self):
        sched = make_scheduler(n_layers=8, total_budget=64, min_budget=4)
        budgets = sched.all_budgets()
        assert budgets[0] > budgets[-1], (
            f"With 8 layers first={budgets[0]} should be strictly > last={budgets[-1]}"
        )


# ---------------------------------------------------------------------------
# PyramidKVCache tests
# ---------------------------------------------------------------------------

class TestPyramidKVCache:

    # Test 6: compress() output shape correct (k ≤ b_l)
    def test_compress_output_shape(self):
        cache = make_cache()
        keys, values, attn = make_kv_attn()
        for layer_idx in range(N_LAYERS):
            ck, cv = cache.compress(layer_idx, keys, values, attn)
            b_l = cache.scheduler.budget_for_layer(layer_idx)
            assert ck.dim() == 4, f"Layer {layer_idx}: compressed_keys should be 4-D"
            B_out, H_out, k_out, D_out = ck.shape
            assert B_out == B
            assert H_out == N_HEADS
            assert k_out <= b_l, (
                f"Layer {layer_idx}: retained {k_out} > budget {b_l}"
            )
            assert D_out == HEAD_DIM

    # Test 7: compress() keys and values have identical shape
    def test_compress_keys_values_same_shape(self):
        cache = make_cache()
        keys, values, attn = make_kv_attn()
        for layer_idx in range(N_LAYERS):
            ck, cv = cache.compress(layer_idx, keys, values, attn)
            assert ck.shape == cv.shape, (
                f"Layer {layer_idx}: keys shape {ck.shape} != values shape {cv.shape}"
            )

    # Test 8: compress() is deterministic
    def test_compress_deterministic(self):
        cache = make_cache()
        keys, values, attn = make_kv_attn(seed=42)
        ck1, cv1 = cache.compress(0, keys, values, attn)
        ck2, cv2 = cache.compress(0, keys, values, attn)
        assert torch.allclose(ck1, ck2), "Keys differ across calls with same input"
        assert torch.allclose(cv1, cv2), "Values differ across calls with same input"

    # Test 11: No NaN/Inf in compressed output under uniform attention
    def test_no_nan_inf_uniform_attention(self):
        cache = make_cache()
        torch.manual_seed(7)
        keys = torch.randn(B, N_HEADS, T, HEAD_DIM)
        values = torch.randn(B, N_HEADS, T, HEAD_DIM)
        # Perfectly uniform attention
        attn = torch.full((B, N_HEADS, WINDOW_SIZE, T), 1.0 / T)
        for layer_idx in range(N_LAYERS):
            ck, cv = cache.compress(layer_idx, keys, values, attn)
            assert not torch.isnan(ck).any(), f"NaN in keys at layer {layer_idx}"
            assert not torch.isinf(ck).any(), f"Inf in keys at layer {layer_idx}"
            assert not torch.isnan(cv).any(), f"NaN in values at layer {layer_idx}"
            assert not torch.isinf(cv).any(), f"Inf in values at layer {layer_idx}"

    # Test 12: Higher-budget layers retain more tokens than lower-budget layers
    def test_higher_budget_layers_retain_more(self):
        # Use a long sequence so compression is always active
        t_long = 64
        cache = make_cache(n_layers=4, total_budget=80, min_budget=4, window_size=4)
        keys, values, attn = make_kv_attn(t=t_long, t_obs=4, seed=1)
        retained = []
        for layer_idx in range(N_LAYERS):
            ck, _ = cache.compress(layer_idx, keys, values, attn)
            retained.append(ck.shape[2])
        # Layer 0 (highest budget) should retain >= layer L-1 (lowest budget)
        assert retained[0] >= retained[-1], (
            f"Layer 0 retained {retained[0]}, layer {N_LAYERS-1} retained {retained[-1]}; "
            f"expected layer 0 to retain at least as many"
        )

    # Test 13: batch_size=1 works
    def test_batch_size_1(self):
        cache = make_cache()
        keys, values, attn = make_kv_attn(b=1)
        ck, cv = cache.compress(0, keys, values, attn)
        assert ck.shape[0] == 1
        assert cv.shape[0] == 1

    # Test 14: window_size portion is retained in compressed output
    def test_window_tokens_retained(self):
        """The last `window_size` tokens should always appear in compressed output."""
        cache = make_cache(window_size=WINDOW_SIZE)
        torch.manual_seed(99)
        # Construct keys so the last WINDOW_SIZE tokens have a unique fingerprint
        keys = torch.zeros(B, N_HEADS, T, HEAD_DIM)
        values = torch.zeros(B, N_HEADS, T, HEAD_DIM)
        # Give the last `window_size` positions a unique large value
        unique_val = 999.0
        keys[:, :, -WINDOW_SIZE:, :] = unique_val
        values[:, :, -WINDOW_SIZE:, :] = unique_val

        # Uniform attention — importance is spread equally, so historical tokens
        # compete; window tokens should still be retained regardless
        attn = torch.full((B, N_HEADS, WINDOW_SIZE, T), 1.0 / T)

        for layer_idx in range(N_LAYERS):
            b_l = cache.scheduler.budget_for_layer(layer_idx)
            # Only meaningful when sequence > budget
            if T <= b_l:
                continue
            ck, cv = cache.compress(layer_idx, keys, values, attn)
            # Check that the unique_val appears in compressed output
            has_window = (ck == unique_val).all(dim=-1).any()
            assert has_window, (
                f"Layer {layer_idx}: window tokens with value {unique_val} "
                f"not found in compressed keys"
            )
