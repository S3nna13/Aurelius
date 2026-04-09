"""Tests for BigBird-style sparse attention module."""

import torch
import pytest
from torch import Tensor

from src.model.bigbird_attention import (
    BigBirdConfig,
    BigBirdAttention,
    BigBirdBlock,
    compute_attention_sparsity,
    count_attended_positions,
    create_bigbird_mask,
    sparse_attention_with_mask,
)

# Common test hyperparameters
T = 16
WINDOW = 3
N_GLOBAL = 2
N_RANDOM = 1
B = 2
H = 2
D = 32  # head_dim

D_MODEL = 64  # n_heads * head_dim


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------
def test_config_defaults():
    cfg = BigBirdConfig()
    assert cfg.window_size == 3
    assert cfg.n_global_tokens == 2


# ---------------------------------------------------------------------------
# 2. test_create_bigbird_mask_shape
# ---------------------------------------------------------------------------
def test_create_bigbird_mask_shape():
    mask = create_bigbird_mask(T, WINDOW, N_GLOBAL, N_RANDOM, seed=0)
    assert mask.shape == (T, T)
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 3. test_bigbird_mask_global_tokens_attend_all
# ---------------------------------------------------------------------------
def test_bigbird_mask_global_tokens_attend_all():
    mask = create_bigbird_mask(T, WINDOW, N_GLOBAL, N_RANDOM, seed=0)
    # Rows 0 and 1 (global tokens) must be all True
    for g in range(N_GLOBAL):
        assert mask[g].all(), f"Global token row {g} should attend all positions"


# ---------------------------------------------------------------------------
# 4. test_bigbird_mask_global_tokens_attended_by_all
# ---------------------------------------------------------------------------
def test_bigbird_mask_global_tokens_attended_by_all():
    mask = create_bigbird_mask(T, WINDOW, N_GLOBAL, N_RANDOM, seed=0)
    # Columns 0 and 1 must be all True
    for g in range(N_GLOBAL):
        assert mask[:, g].all(), f"All tokens should attend to global token {g}"


# ---------------------------------------------------------------------------
# 5. test_bigbird_mask_local_window
# ---------------------------------------------------------------------------
def test_bigbird_mask_local_window():
    # With window_size=3, token 5 should attend tokens 2..8 (5-3=2, 5+3=8)
    mask = create_bigbird_mask(T, WINDOW, n_global_tokens=0, n_random_keys=0, seed=0)
    for j in range(2, 9):  # 2 through 8 inclusive
        assert mask[5, j], f"Token 5 should attend token {j} (window_size=3)"


# ---------------------------------------------------------------------------
# 6. test_bigbird_mask_diagonal
# ---------------------------------------------------------------------------
def test_bigbird_mask_diagonal():
    mask = create_bigbird_mask(T, WINDOW, N_GLOBAL, N_RANDOM, seed=0)
    diag = mask.diagonal()
    assert diag.all(), "All diagonal (self-attention) positions should be True"


# ---------------------------------------------------------------------------
# 7. test_compute_sparsity_range
# ---------------------------------------------------------------------------
def test_compute_sparsity_range():
    mask = create_bigbird_mask(T, WINDOW, N_GLOBAL, N_RANDOM, seed=0)
    sparsity = compute_attention_sparsity(mask)
    assert 0.0 <= sparsity <= 1.0, f"Sparsity {sparsity} out of [0, 1]"


# ---------------------------------------------------------------------------
# 8. test_compute_sparsity_full
# ---------------------------------------------------------------------------
def test_compute_sparsity_full():
    full_mask = torch.ones(T, T, dtype=torch.bool)
    sparsity = compute_attention_sparsity(full_mask)
    assert sparsity == 0.0, f"Full mask should have sparsity 0.0, got {sparsity}"


# ---------------------------------------------------------------------------
# 9. test_count_attended_positions_shape
# ---------------------------------------------------------------------------
def test_count_attended_positions_shape():
    mask = create_bigbird_mask(T, WINDOW, N_GLOBAL, N_RANDOM, seed=0)
    counts = count_attended_positions(mask)
    assert counts.shape == (T,), f"Expected shape ({T},), got {counts.shape}"


# ---------------------------------------------------------------------------
# 10. test_count_attended_positions_global
# ---------------------------------------------------------------------------
def test_count_attended_positions_global():
    mask = create_bigbird_mask(T, WINDOW, N_GLOBAL, N_RANDOM, seed=0)
    counts = count_attended_positions(mask)
    # Global tokens should attend to all T positions
    for g in range(N_GLOBAL):
        assert counts[g].item() == T, (
            f"Global token {g} should attend {T} positions, got {counts[g].item()}"
        )


# ---------------------------------------------------------------------------
# 11. test_sparse_attention_shape
# ---------------------------------------------------------------------------
def test_sparse_attention_shape():
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    mask = torch.ones(T, T, dtype=torch.bool)
    out = sparse_attention_with_mask(q, k, v, mask)
    assert out.shape == (B, H, T, D), f"Expected shape ({B}, {H}, {T}, {D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 12. test_sparse_attention_with_full_mask_equals_standard
# ---------------------------------------------------------------------------
def test_sparse_attention_with_full_mask_equals_standard():
    torch.manual_seed(42)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    full_mask = torch.ones(T, T, dtype=torch.bool)
    scale = D ** -0.5

    # sparse_attention_with_mask with full mask
    out_sparse = sparse_attention_with_mask(q, k, v, full_mask, scale=scale)

    # Standard explicit attention
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    out_standard = torch.matmul(attn_weights, v)

    assert torch.allclose(out_sparse, out_standard, atol=1e-4), (
        "sparse_attention with full mask should match standard attention"
    )


# ---------------------------------------------------------------------------
# 13. test_bigbird_attention_shape
# ---------------------------------------------------------------------------
def test_bigbird_attention_shape():
    cfg = BigBirdConfig(
        window_size=WINDOW,
        n_global_tokens=N_GLOBAL,
        n_random_keys=N_RANDOM,
        d_model=D_MODEL,
        n_heads=H,
        head_dim=D,
    )
    model = BigBirdAttention(cfg)
    x = torch.randn(B, T, D_MODEL)
    out = model(x, seed=0)
    assert out.shape == (B, T, D_MODEL), (
        f"Expected shape ({B}, {T}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 14. test_bigbird_attention_different_seeds
# ---------------------------------------------------------------------------
def test_bigbird_attention_different_seeds():
    cfg = BigBirdConfig(
        window_size=WINDOW,
        n_global_tokens=N_GLOBAL,
        n_random_keys=N_RANDOM,
        d_model=D_MODEL,
        n_heads=H,
        head_dim=D,
    )
    mask0 = create_bigbird_mask(T, WINDOW, N_GLOBAL, N_RANDOM, seed=0)
    mask1 = create_bigbird_mask(T, WINDOW, N_GLOBAL, N_RANDOM, seed=999)
    # Different seeds should produce different random key assignments
    assert not torch.equal(mask0, mask1), (
        "Different seeds should produce different BigBird masks (random keys differ)"
    )


# ---------------------------------------------------------------------------
# 15. test_bigbird_block_shape
# ---------------------------------------------------------------------------
def test_bigbird_block_shape():
    cfg = BigBirdConfig(
        window_size=WINDOW,
        n_global_tokens=N_GLOBAL,
        n_random_keys=N_RANDOM,
        d_model=D_MODEL,
        n_heads=H,
        head_dim=D,
    )
    block = BigBirdBlock(cfg)
    x = torch.randn(B, T, D_MODEL)
    out = block(x)
    assert out.shape == (B, T, D_MODEL), (
        f"Expected shape ({B}, {T}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 16. test_bigbird_block_residual
# ---------------------------------------------------------------------------
def test_bigbird_block_residual():
    cfg = BigBirdConfig(
        window_size=WINDOW,
        n_global_tokens=N_GLOBAL,
        n_random_keys=N_RANDOM,
        d_model=D_MODEL,
        n_heads=H,
        head_dim=D,
    )
    block = BigBirdBlock(cfg)
    x = torch.randn(B, T, D_MODEL)
    out = block(x)
    # Output should differ from input (non-trivial transform)
    assert not torch.allclose(out, x, atol=1e-6), (
        "BigBirdBlock output should differ from input (non-trivial residual transform)"
    )
